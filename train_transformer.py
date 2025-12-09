"""
训练 Transformer Motion Planning Policy

直接从 waypoints 学习:
- 输入: 当前关节配置 + 目标关节配置  
- 输出: 关节增量 (action)

Usage:
    python train_transformer.py \
        --waypoints_dir expert_200 \
        --tasks_dir tasks_200 \
        --output_dir runs/transformer_v1 \
        --epochs 100 \
        --batch_size 256
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import pickle
from datetime import datetime

from transformer_policy import TransformerMotionPolicy, TransformerMotionPolicyV2


class WaypointDataset(Dataset):
    """
    从 waypoints 直接构建训练数据集
    
    每个样本:
        - current_state: 当前关节配置 [6]
        - goal_state: 目标关节配置 [6]  
        - action: 下一步关节增量 [6]
    
    注意: 多臂场景 (12, 18, 24 关节) 会被拆分成单臂样本
    """
    def __init__(self, waypoints_dir, tasks_dir=None, action_scale=1.0):
        self.samples = []
        self.action_scale = action_scale
        
        waypoints_path = Path(waypoints_dir)
        waypoint_files = sorted(list(waypoints_path.glob('*.npy')))
        
        print(f"Loading {len(waypoint_files)} waypoint files...")
        
        for wp_file in tqdm(waypoint_files, desc="Loading waypoints"):
            try:
                waypoints = np.load(wp_file, allow_pickle=True)
                
                if waypoints.ndim != 2:
                    continue
                
                T, total_dim = waypoints.shape
                
                # 检查是否是 6 的倍数 (单臂或多臂)
                if total_dim % 6 != 0:
                    print(f"Skipping {wp_file.name}: dim={total_dim} not multiple of 6")
                    continue
                
                num_arms = total_dim // 6
                
                if T < 2:
                    continue
                
                # 为每个机械臂分别处理
                for arm_idx in range(num_arms):
                    start_idx = arm_idx * 6
                    end_idx = (arm_idx + 1) * 6
                    arm_waypoints = waypoints[:, start_idx:end_idx]
                    
                    # 目标是最后一个waypoint
                    goal = arm_waypoints[-1]
                    
                    # 为每一步创建训练样本
                    for t in range(T - 1):
                        current = arm_waypoints[t]
                        next_state = arm_waypoints[t + 1]
                        action = next_state - current  # 关节增量
                        
                        self.samples.append({
                            'current': current.astype(np.float32),
                            'goal': goal.astype(np.float32),
                            'action': action.astype(np.float32),
                            'progress': t / (T - 1),  # 进度信息 (可选)
                        })
                    
            except Exception as e:
                print(f"Error loading {wp_file}: {e}")
                continue
        
        print(f"Created {len(self.samples)} training samples")
        
        # 计算动作统计
        all_actions = np.array([s['action'] for s in self.samples])
        self.action_mean = all_actions.mean(axis=0)
        self.action_std = all_actions.std(axis=0) + 1e-8
        print(f"Action stats - mean: {self.action_mean}, std: {self.action_std}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['current']),
            torch.FloatTensor(sample['goal']),
            torch.FloatTensor(sample['action']),
        )


class SequenceWaypointDataset(Dataset):
    """
    序列版本 - 输入整个轨迹预测整个动作序列
    """
    def __init__(self, waypoints_dir, max_len=256, pad_value=0.0):
        self.trajectories = []
        self.max_len = max_len
        self.pad_value = pad_value
        
        waypoints_path = Path(waypoints_dir)
        waypoint_files = sorted(list(waypoints_path.glob('*.npy')))
        
        print(f"Loading {len(waypoint_files)} waypoint files...")
        
        for wp_file in tqdm(waypoint_files, desc="Loading waypoints"):
            try:
                waypoints = np.load(wp_file, allow_pickle=True)
                
                if waypoints.ndim != 2 or waypoints.shape[1] != 6:
                    continue
                
                T = len(waypoints)
                if T < 2:
                    continue
                
                # 计算actions (关节增量)
                actions = np.diff(waypoints, axis=0)  # [T-1, 6]
                states = waypoints[:-1]               # [T-1, 6]
                goal = waypoints[-1]                  # [6]
                
                self.trajectories.append({
                    'states': states.astype(np.float32),
                    'actions': actions.astype(np.float32),
                    'goal': goal.astype(np.float32),
                    'length': len(states),
                })
                    
            except Exception as e:
                print(f"Error loading {wp_file}: {e}")
                continue
        
        print(f"Loaded {len(self.trajectories)} trajectories")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        T = traj['length']
        
        # Padding
        states = np.zeros((self.max_len, 6), dtype=np.float32)
        actions = np.zeros((self.max_len, 6), dtype=np.float32)
        mask = np.ones(self.max_len, dtype=bool)  # True = masked (padding)
        
        states[:T] = traj['states']
        actions[:T] = traj['actions']
        mask[:T] = False
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(traj['goal']),
            torch.FloatTensor(actions),
            torch.BoolTensor(mask),
            T
        )


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    model.train()
    total_loss = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        current, goal, action = [x.to(device) for x in batch]
        
        optimizer.zero_grad()
        
        # 前向传播
        pred_action = model(current, goal)
        
        # 计算损失
        loss = criterion(pred_action, action)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item() * len(current)
        total_samples += len(current)
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_errors = []
    
    with torch.no_grad():
        for batch in dataloader:
            current, goal, action = [x.to(device) for x in batch]
            
            pred_action = model(current, goal)
            loss = criterion(pred_action, action)
            
            total_loss += loss.item() * len(current)
            total_samples += len(current)
            
            # 计算每个样本的L2误差
            errors = torch.norm(pred_action - action, dim=-1)
            all_errors.extend(errors.cpu().numpy())
    
    return {
        'loss': total_loss / total_samples,
        'mean_error': np.mean(all_errors),
        'max_error': np.max(all_errors),
        'std_error': np.std(all_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints_dir', type=str, required=True)
    parser.add_argument('--tasks_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/transformer')
    parser.add_argument('--model_type', type=str, default='v2', choices=['v1', 'v2'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 加载数据
    print("Loading dataset...")
    dataset = WaypointDataset(args.waypoints_dir, args.tasks_dir)
    
    # 划分训练/验证集
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 创建模型
    if args.model_type == 'v2':
        model = TransformerMotionPolicyV2(
            state_dim=6,
            action_dim=6,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    else:
        model = TransformerMotionPolicy(
            state_dim=6,
            action_dim=6,
            d_model=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_error': []}
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_error'].append(val_metrics['mean_error'])
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Val Error: {val_metrics['mean_error']:.6f}")
        
        # 保存最佳模型
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_error': val_metrics['mean_error'],
                'config': config,
            }, output_dir / 'best_model.pth')
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    # 保存最终模型和训练历史
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
    }, output_dir / 'final_model.pth')
    
    # 转换为 JSON 可序列化的格式
    history_json = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history_json, f)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    print(f"Models saved to {output_dir}")


if __name__ == '__main__':
    main()
