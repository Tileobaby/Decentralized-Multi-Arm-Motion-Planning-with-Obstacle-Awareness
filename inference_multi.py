# inference_multi.py - N-robot motion planning with configurable base positions
# Supports arbitrary number of robots with user-defined spatial configurations

import os
import sys
from math import ceil
from pathlib import Path

# Check if render is enabled BEFORE importing anything that imports torch
_render_enabled = False
if '--render' in sys.argv:
    try:
        render_idx = sys.argv.index('--render')
        if render_idx + 1 < len(sys.argv):
            _render_enabled = sys.argv[render_idx + 1].lower() == 'true'
    except (ValueError, IndexError):
        pass

# Import IsaacGym FIRST if rendering is enabled
if _render_enabled:
    _original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]
    from torch_robotics.isaac_gym_envs.motion_planning_envs import PandaMotionPlanningIsaacGymEnv, MotionPlanningController
    sys.argv = _original_argv

# Now safe to import torch and other dependencies
import torch
import matplotlib.pyplot as plt
import einops
from einops._torch_specific import allow_ops_in_compiled_graph

from experiment_launcher import single_experiment_yaml, run_experiment

from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml

from torch_robotics.robots import RobotPanda
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionWorkspaceBoundariesDistanceField
)
from torch_robotics.trajectory.utils import interpolate_traj_via_points
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

from multi_robot_utils import (
    setup_robot_with_transform,
    build_costs_for_robot,
    create_guide_with_other_traj_support,
    multi_robot_gibbs_sample,
    compute_pairwise_inter_robot_metrics
)

import os, random, git
from torch_robotics import environments, robots
from torch_robotics.tasks.tasks import PlanningTask

allow_ops_in_compiled_graph()

TRAINED_MODELS_DIR = '../../data_trained_models/'


def build_env_robot_task_from_random_shard(dataset_subdir: str,
                                           *,
                                           seed=None,
                                           use_extra_objects=False,
                                           obstacle_cutoff_margin=None,
                                           tensor_args=None):
    repo_root = git.Repo('.', search_parent_directories=True).working_dir
    base_dir  = os.path.join(repo_root, 'data_trajectories', dataset_subdir)

    shards = []
    for d in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, d)
        if os.path.isdir(p) and \
           os.path.exists(os.path.join(p, 'args.yaml')) and \
           os.path.exists(os.path.join(p, 'metadata.yaml')):
            shards.append(d)
    if not shards:
        raise FileNotFoundError(f"No shards with args.yaml & metadata.yaml in {base_dir}")

    rng = random.Random(seed)
    chosen = rng.choice(shards)

    args_path = os.path.join(base_dir, chosen, 'args.yaml')
    md_path   = os.path.join(base_dir, chosen, 'metadata.yaml')
    args2     = load_params_from_yaml(args_path)
    meta      = load_params_from_yaml(md_path)

    if obstacle_cutoff_margin is not None:
        args2['obstacle_cutoff_margin'] = obstacle_cutoff_margin

    env_cls_name = meta['env_id'] + ('ExtraObjects' if use_extra_objects else '')
    env_cls  = getattr(environments, env_cls_name)
    env      = env_cls(tensor_args=tensor_args)

    robot_cls = getattr(robots, meta['robot_id'])
    robot_env = robot_cls(tensor_args=tensor_args)

    task = PlanningTask(env=env, robot=robot_env, tensor_args=tensor_args, **args2)
    return env, robot_env, task, chosen


def parse_robot_positions(pos_str: str):
    """Parse robot positions from string format.
    
    Format: "x1,y1,z1,yaw1|x2,y2,z2,yaw2|..." (use | as separator to avoid shell issues)
    Alternative: "x1,y1,z1,yaw1;x2,y2,z2,yaw2;..." (need quotes in shell)
    Example: "-0.49,0,0,0|0.49,0,0,0" for 2 robots at x=-0.49 and x=0.49
    
    Returns:
        List of tuples (x, y, z, yaw)
    """
    positions = []
    # Support both | and ; as separators
    separator = '|' if '|' in pos_str else ';'
    for robot_pos in pos_str.split(separator):
        robot_pos = robot_pos.strip()
        if not robot_pos:
            continue
        coords = [float(x.strip()) for x in robot_pos.split(',')]
        if len(coords) != 4:
            raise ValueError(f"Each robot position must have 4 values (x,y,z,yaw), got {len(coords)}")
        positions.append(tuple(coords))
    return positions


def generate_default_positions(n_robots: int, spacing: float = 0.5):
    """Generate default robot positions in a line along x-axis.
    
    Args:
        n_robots: Number of robots
        spacing: Spacing between robots (default 0.5m)
    
    Returns:
        List of tuples (x, y, z, yaw)
    """
    positions = []
    if n_robots == 1:
        positions.append((0.0, 0.0, 0.0, 0.0))
    else:
        # Center the line at origin
        total_width = (n_robots - 1) * spacing
        start_x = -total_width / 2
        for i in range(n_robots):
            x = start_x + i * spacing
            positions.append((x, 0.0, 0.0, 0.0))
    return positions


@single_experiment_yaml
def experiment(
    model_id: str = 'EnvSpheres3D-RobotPanda',
    
    # Multi-robot configuration
    n_robots: int = 2,
    robot_positions: str = "",  # Format: "x1,y1,z1,yaw1|x2,y2,z2,yaw2|..." Leave empty for auto-generated positions
    robot_spacing: float = 0.5,  # Spacing between robots when using auto-generated positions (m)
    
    n_samples: int = 20,
    
    weight_grad_cost_collision: float = 2e-2,
    weight_grad_cost_smoothness: float = 1e-7,
    weight_grad_cost_inter_robot: float = 10.0,
    
    planner_alg: str = 'mpd',
    use_guide_on_extra_objects_only: bool = False,
    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 5,
    n_diffusion_steps_without_noise: int = 5,
    factor_num_interpolated_points_for_collision: float = 1.5,
    trajectory_duration: float = 5.0,
    
    device: str = 'cuda',
    render: bool = False,
    seed: int = 0,
    results_dir: str = 'logs',
    
    **kwargs
):
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # Parse or generate robot positions
    if robot_positions:
        positions = parse_robot_positions(robot_positions)
    else:
        positions = generate_default_positions(n_robots, robot_spacing)
    
    if len(positions) != n_robots:
        raise ValueError(f"Number of positions ({len(positions)}) must match n_robots ({n_robots})")

    print(f'==================== MULTI-ROBOT INFERENCE ====================')
    print(f'Model: {model_id}')
    print(f'Number of robots: {n_robots}')
    print(f'Robot positions:')
    for i, (x, y, z, yaw) in enumerate(positions):
        print(f'  Robot {i}: x={x:.3f}, y={y:.3f}, z={z:.3f}, yaw={yaw:.3f}')
    print(f'Seed: {seed}, Samples: {n_samples}')

    # Setup directories
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(model_dir, f'results_inference_multi_{n_robots}robots', str(seed))
    os.makedirs(results_dir, exist_ok=True)

    # Load args and dataset
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,
        **args,
        tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    robot = dataset.robot
    
    # Build environment
    env, robot_meta, task, chosen_shard = build_env_robot_task_from_random_shard(
        args['dataset_subdir'],
        seed=seed,
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args
    )
    print(f"[DATA] Using shard '{chosen_shard}'")

    # Create N robots using utility function
    dt = trajectory_duration / n_support_points
    
    robots = []
    for i, (x, y, z, yaw) in enumerate(positions):
        robot_i = setup_robot_with_transform(
            robot,
            offset_x=x, offset_y=y, offset_z=z, yaw=yaw,
            dt=dt,
            tensor_args=tensor_args
        )
        robots.append(robot_i)
    
    print(f"[ROBOTS] Created {n_robots} Panda robots with configured positions")

    # Load and compile model
    diffusion_configs = dict(
        variance_schedule=args['variance_schedule'],
        n_diffusion_steps=args['n_diffusion_steps'],
        predict_epsilon=args['predict_epsilon'],
    )
    unet_configs = dict(
        state_dim=dataset.state_dim,
        n_support_points=dataset.n_support_points,
        unet_input_dim=args['unet_input_dim'],
        dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']],
    )
    diffusion_model = get_model(
        model_class=args['diffusion_model_class'],
        model=TemporalUnet(**unet_configs),
        tensor_args=tensor_args,
        **diffusion_configs,
        **unet_configs
    )
    diffusion_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, 'checkpoints', 
                        'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'),
            map_location=device
        )
    )
    diffusion_model.eval()
    
    freeze_torch_model_params(diffusion_model)
    diffusion_model = torch.compile(diffusion_model)
    diffusion_model.warmup(horizon=n_support_points, device=device)
    
    models = [diffusion_model] * n_robots  # Same model for all robots
    print(f"[MODEL] Loaded and compiled")

    # Sample start/goal for all robots
    def sample_start_goal():
        n_tries = 100
        for _ in range(n_tries):
            q_free = task.random_coll_free_q(n_samples=2)
            start, goal = q_free[0], q_free[1]
            if torch.linalg.norm(start - goal) > dataset.threshold_start_goal_pos:
                return start, goal
        raise ValueError("No collision-free start/goal found.")

    starts = []
    goals = []
    for i in range(n_robots):
        start, goal = sample_start_goal()
        starts.append(start)
        goals.append(goal)
        print(f'[START/GOAL] Robot {i}: {start.cpu().numpy()} -> {goal.cpu().numpy()}')

    # Create hard conditions for all robots
    hard_conds_list = []
    for i in range(n_robots):
        hard_conds = dataset.get_hard_conditions(torch.vstack((starts[i], goals[i])), normalize=True)
        hard_conds = {k: einops.repeat(v, 'd -> b d', b=n_samples) for k, v in hard_conds.items()}
        hard_conds_list.append(hard_conds)

    # Build costs and guides
    if use_guide_on_extra_objects_only:
        collision_fields = task.get_collision_fields_extra_objects()
    else:
        collision_fields = task.get_collision_fields()
    
    # Remove workspace boundaries (offset robots cause false positives)
    collision_fields = [
        cf for cf in collision_fields 
        if not isinstance(cf, CollisionWorkspaceBoundariesDistanceField)
    ]
    print(f"[COSTS] Using {len(collision_fields)} collision fields (workspace boundaries removed)")

    # Build costs for each robot using utility function
    costs = []
    for i in range(n_robots):
        other_robots = [robots[j] for j in range(n_robots) if j != i]
        cost = build_costs_for_robot(
            robots[i], other_robots, collision_fields, n_support_points, dt,
            weight_grad_cost_collision, weight_grad_cost_smoothness, weight_grad_cost_inter_robot,
            tensor_args
        )
        costs.append(cost)
    
    # Create guides using utility function
    guides = []
    for i in range(n_robots):
        guide = create_guide_with_other_traj_support(
            dataset, costs[i], n_support_points, 
            factor_num_interpolated_points_for_collision,
            tensor_args
        )
        guides.append(guide)
    
    print(f"[GUIDE] Created guides with inter-robot awareness for {n_robots} robots")

    # Setup sampling kwargs
    t_start_guide = ceil(start_guide_steps_fraction * diffusion_model.n_diffusion_steps)
    
    sample_kwargs_list = []
    for i in range(n_robots):
        sample_kwargs = dict(
            guide=guides[i],
            n_guide_steps=n_guide_steps,
            t_start_guide=t_start_guide,
            noise_std_extra_schedule_fn=lambda x: 0.5,
        )
        sample_kwargs_list.append(sample_kwargs)

    # Multi-robot Gibbs sampling
    print(f"\n[SAMPLING] Starting {n_robots}-robot Gibbs sampling...")
    
    with TimerCUDA() as timer:
        trajs_final_list = multi_robot_gibbs_sample(
            models=models,
            hard_conds_list=hard_conds_list,
            sample_kwargs_list=sample_kwargs_list,
            horizon=n_support_points,
            state_dim=dataset.state_dim,
            n_diffusion_steps_without_noise=n_diffusion_steps_without_noise,
            return_chain=False
        )
    
    print(f"[TIMING] Sampling took {timer.elapsed:.3f}s")
    
    # Unnormalize trajectories
    trajs_final_list_unnorm = []
    for i in range(n_robots):
        trajs_unnorm = dataset.unnormalize_trajectories(trajs_final_list[i])
        trajs_final_list_unnorm.append(trajs_unnorm)
    
    print(f"[OUTPUT] Generated trajectories: {[f'Robot{i}={trajs.shape}' for i, trajs in enumerate(trajs_final_list_unnorm)]}")

    # Compute metrics for each robot
    print(f"\n[METRICS] Environment Collisions:")
    
    # Backup original task settings
    _ws_backup = task.df_collision_ws_boundaries
    _fk_backup = task.robot.fk_map_collision
    
    collision_free_masks = []
    try:
        task.df_collision_ws_boundaries = None
        
        for i in range(n_robots):
            # Check collision for robot i
            task.robot.fk_map_collision = robots[i].fk_map_collision
            res = task.collision_breakdown_for_trajs(
                trajs_final_list_unnorm[i], margin=0.01, num_interpolation=5,
                override_fk=robots[i].fk_map_collision, return_waypoint_masks=True
            )
            
            any_collision = res['any']
            n_free = (~any_collision).sum().item()
            
            # Debug: check collision breakdown
            #print(f"  Robot {i} collision breakdown:")
            #for key in res.keys():
            #    if key != 'any' and key != 'waypoint_masks':
            #        coll_count = res[key].sum().item() if isinstance(res[key], torch.Tensor) else res[key]
            #        print(f"    {key}: {coll_count}")
            
            collision_free_masks.append(~any_collision)
            print(f"  Robot {i}: {n_free}/{n_samples} collision-free ({100*n_free/n_samples:.1f}%)")
    
    finally:
        task.df_collision_ws_boundaries = _ws_backup
        task.robot.fk_map_collision = _fk_backup

    # Compute pairwise inter-robot collision metrics
    print(f"\n[METRICS] Inter-Robot Collisions:")
    inter_metrics = compute_pairwise_inter_robot_metrics(
        robots, trajs_final_list_unnorm,
        margin=0.02, upsample_k=5
    )
    
    # Print per-pair statistics
    for pair_info in inter_metrics['pairs']:
        i, j = pair_info['pair']
        print(f"\n  Pair Robot {i} - Robot {j}:")
        print(f"    Min distance (mean): {pair_info['min_distance_mean']:.4f}m")
        print(f"    Min clearance (mean): {pair_info['min_clearance_mean']:.4f}m")
        print(f"    Penetrations: {pair_info['penetration_count']}/{n_samples} ({100*pair_info['penetration_count']/n_samples:.1f}%)")
        penetration_indices = torch.where(pair_info['penetration'])[0].tolist()
        if len(penetration_indices) > 0:
            print(f"    Penetration samples: {penetration_indices}")
    
    # Print aggregate statistics
    print(f"\n  Aggregate (all pairs):")
    print(f"    Min distance: {inter_metrics['min_distance_mean']:.4f}m (min: {inter_metrics['min_distance_min']:.4f}m)")
    print(f"    Min clearance: {inter_metrics['min_clearance_mean']:.4f}m (min: {inter_metrics['min_clearance_min']:.4f}m)")
    print(f"    Penetration rate: {inter_metrics['penetration_rate']*100:.2f}%")
    
    # Debug: show which samples have penetration
    penetration_indices = torch.where(inter_metrics['any_penetration'])[0]
    print(f"    Samples with ANY inter-robot penetration: {penetration_indices.tolist()}")

    # Combined success: all robots collision-free AND no inter-robot collisions
    all_robots_free = torch.ones(n_samples, dtype=torch.bool, device=device)
    for mask in collision_free_masks:
        all_robots_free = all_robots_free & mask
    
    n_all_robots_free = all_robots_free.sum().item()
    print(f"\n[DEBUG] All robots env-collision-free: {n_all_robots_free}/{n_samples}")
    print(f"[DEBUG] Env-free samples: {torch.where(all_robots_free)[0].tolist()}")
    
    all_success = all_robots_free & (~inter_metrics['any_penetration'])
    n_success = all_success.sum().item()
    
    print(f"\n[SUCCESS] All {n_robots} robots free + no inter-collision: {n_success}/{n_samples} ({100*n_success/n_samples:.1f}%)")
    if n_success > 0:
        print(f"[SUCCESS] Success samples: {torch.where(all_success)[0].tolist()}")

    # Rendering (only for Panda robots)
    if render and all(isinstance(r, RobotPanda) for r in robots):
        print(f"\n[RENDER] Starting visualization...")
        planner_visualizer = PlanningVisualizer(task=task)
        base_file_name = Path(os.path.basename(__file__)).stem
        
        # Find a successful trajectory
        idx_vis = 0
        if n_success > 0:
            success_indices = torch.where(all_success)[0]
            idx_vis = success_indices[0].item()
        
        # Get position trajectories for all robots
        trajs_pos_list = []
        for i in range(n_robots):
            q_traj = robots[i].get_position(trajs_final_list_unnorm[i][idx_vis:idx_vis+1]).squeeze(0)
            trajs_pos_list.append(q_traj)
        
        # Stack for multi-robot visualization [n_robots, horizon, dim]
        trajs_pos_all = torch.stack(trajs_pos_list, dim=0)
        
        # Interpolate for smoother visualization
        _tr = interpolate_traj_via_points(trajs_pos_all, 2)
        trajs_pos_all = _tr.contiguous()
        
        # Transpose to [horizon, n_robots, dim]
        trajs_pos_all = trajs_pos_all.transpose(0, 1)
        
        # Create IsaacGym environment
        _original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]
        motion_planning_isaac_env = PandaMotionPlanningIsaacGymEnv(
            env, robots[0], task,
            asset_root="../../deps/isaacgym/assets",
            franka_asset_file="urdf/franka_description/robots/franka_panda.urdf",
            controller_type='position',
            num_envs=n_robots,
            all_robots_in_one_env=True,
            color_robots=True,
            show_goal_configuration=False,
            sync_with_real_time=True,
            show_collision_spheres=False,
            dt=dt,
        )
        sys.argv = _original_argv
        
        from isaacgym import gymapi, gymtorch
        
        # Set robot positions
        gym = motion_planning_isaac_env.gym
        sim = motion_planning_isaac_env.sim
        envh = motion_planning_isaac_env.envs[0]
        
        gym.refresh_actor_root_state_tensor(sim)
        root = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
        
        num_actors = gym.get_actor_count(envh)
        robot_handles = []
        for i in range(num_actors):
            ah = gym.get_actor_handle(envh, i)
            name = gym.get_actor_name(envh, ah)
            if name and "franka" in name.lower():
                robot_handles.append(ah)
        
        # Set positions for all robots
        for i in range(min(n_robots, len(robot_handles))):
            actor_sim_idx = gym.get_actor_index(envh, robot_handles[i], gymapi.DOMAIN_SIM)
            x, y, z, _ = positions[i]
            root[actor_sim_idx, 0:3] = torch.tensor([x, y, z], device=root.device, dtype=root.dtype)
        
        gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(root))
        
        # Build goal vector (concatenate all robot goals)
        goal_vec = torch.cat([goals[i] for i in range(n_robots)])
        
        # Run visualization
        motion_planning_controller = MotionPlanningController(motion_planning_isaac_env)
        
        motion_planning_controller.run_trajectories(
            trajs_pos_all,
            start_states_joint_pos=trajs_pos_all[0],
            goal_state_joint_pos=goal_vec,
            n_first_steps=10,
            n_last_steps=10,
            visualize=True,
            render_viewer_camera=True,
            make_video=True,
            video_path=os.path.join(results_dir, f'{base_file_name}-{n_robots}robots-isaac.mp4'),
            make_gif=False
        )
        
        print(f"[RENDER] Video saved to {results_dir}/{base_file_name}-{n_robots}robots-isaac.mp4")
    
    plt.show()
    
    print(f"\n==================== TEST COMPLETE ====================")
    
    # Return metrics
    result = {
        'n_samples': n_samples,
        'n_success': n_success,
        'success_rate': n_success / n_samples,
        'time': timer.elapsed,
    }
    
    # Add per-robot collision-free counts
    for i in range(n_robots):
        result[f'n_free_robot_{i}'] = collision_free_masks[i].sum().item()
    
    return result


if __name__ == '__main__':
    run_experiment(experiment)
