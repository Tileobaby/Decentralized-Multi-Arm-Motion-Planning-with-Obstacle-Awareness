"""
Multi-Arm Aware Transformer Motion Planning Policy (V3)

关键改进:
1. 每个臂能感知其他臂的状态
2. 使用 Transformer Attention 机制进行臂间协调
3. 支持可变数量的机械臂

输入: 所有臂的当前状态 + 所有臂的目标状态
输出: 所有臂的动作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class MultiArmTransformer(nn.Module):
    """
    多臂感知 Transformer
    
    架构:
    1. 每个臂编码成一个 token
    2. Transformer 让各臂之间交换信息
    3. 每个臂输出自己的动作
    """
    def __init__(
        self,
        joint_dim=6,           # 每个臂的关节维度
        action_dim=6,          # 每个臂的动作维度
        d_model=256,           # Transformer 维度
        nhead=8,               # 注意力头数
        num_layers=6,          # Transformer 层数
        dim_feedforward=1024,  # FFN 维度
        dropout=0.1,
        max_arms=8,            # 最大机械臂数量
    ):
        super().__init__()
        
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_arms = max_arms
        
        # 每个臂的输入: current[6] + goal[6] + diff[6] = 18
        arm_input_dim = joint_dim * 3
        
        # 臂编码器: 将每个臂的状态编码成 d_model 维
        self.arm_encoder = nn.Sequential(
            nn.Linear(arm_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # 位置编码 (区分不同的臂)
        self.arm_position_embedding = nn.Embedding(max_arms, d_model)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh(),
        )
        
        # 可学习的动作缩放
        self.action_scale = nn.Parameter(torch.ones(action_dim) * 0.01)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, current_states, goal_states, arm_mask=None):
        """
        Args:
            current_states: [B, N, joint_dim] - 所有臂的当前状态
            goal_states: [B, N, joint_dim] - 所有臂的目标状态
            arm_mask: [B, N] - 可选，标记哪些臂是有效的 (True=masked/无效)
        
        Returns:
            actions: [B, N, action_dim] - 所有臂的动作
        """
        B, N, _ = current_states.shape
        
        # 计算差值特征
        diff = goal_states - current_states
        
        # 拼接每个臂的特征
        arm_features = torch.cat([current_states, goal_states, diff], dim=-1)  # [B, N, 18]
        
        # 编码每个臂
        x = self.arm_encoder(arm_features)  # [B, N, d_model]
        
        # 添加位置编码
        positions = torch.arange(N, device=x.device)
        pos_embed = self.arm_position_embedding(positions)  # [N, d_model]
        x = x + pos_embed.unsqueeze(0)
        
        # Transformer 处理 (臂间交互)
        if arm_mask is not None:
            x = self.transformer(x, src_key_padding_mask=arm_mask)
        else:
            x = self.transformer(x)
        
        # 解码动作
        actions = self.action_decoder(x) * self.action_scale  # [B, N, action_dim]
        
        return actions
    
    def forward_single_arm(self, current_state, goal_state, other_current=None, other_goal=None):
        """
        单臂推理接口 (兼容旧接口)
        
        Args:
            current_state: [B, 6] - 当前臂状态
            goal_state: [B, 6] - 当前臂目标
            other_current: [B, M, 6] - 其他臂的当前状态 (可选)
            other_goal: [B, M, 6] - 其他臂的目标 (可选)
        
        Returns:
            action: [B, 6] - 当前臂的动作
        """
        B = current_state.shape[0]
        
        if other_current is not None and other_goal is not None:
            # 将当前臂放在第一个位置
            all_current = torch.cat([current_state.unsqueeze(1), other_current], dim=1)
            all_goal = torch.cat([goal_state.unsqueeze(1), other_goal], dim=1)
        else:
            # 只有单臂
            all_current = current_state.unsqueeze(1)  # [B, 1, 6]
            all_goal = goal_state.unsqueeze(1)        # [B, 1, 6]
        
        actions = self.forward(all_current, all_goal)  # [B, N, 6]
        
        # 返回第一个臂的动作
        return actions[:, 0, :]


class MultiArmTransformerV2(nn.Module):
    """
    改进版: 使用 Cross-Attention 让每个臂关注其他臂
    
    更明确地分离 "自己的状态" 和 "环境(其他臂)"
    """
    def __init__(
        self,
        joint_dim=6,
        action_dim=6,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        max_arms=8,
    ):
        super().__init__()
        
        self.joint_dim = joint_dim
        self.action_dim = action_dim
        self.d_model = d_model
        
        # 自身状态编码
        self.self_encoder = nn.Sequential(
            nn.Linear(joint_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # 其他臂状态编码
        self.other_encoder = nn.Sequential(
            nn.Linear(joint_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Cross-Attention layers
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 2)
        ])
        
        # 动作输出
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, action_dim),
            nn.Tanh(),
        )
        
        self.action_scale = nn.Parameter(torch.ones(action_dim) * 0.01)
    
    def forward(self, current_state, goal_state, other_states=None):
        """
        Args:
            current_state: [B, 6] - 当前臂状态
            goal_state: [B, 6] - 目标状态
            other_states: [B, M, 6] - 其他臂的当前状态 (可选)
        
        Returns:
            action: [B, 6]
        """
        B = current_state.shape[0]
        
        # 编码自身
        diff = goal_state - current_state
        self_feat = torch.cat([current_state, goal_state, diff], dim=-1)
        x = self.self_encoder(self_feat)  # [B, d_model]
        x = x.unsqueeze(1)  # [B, 1, d_model] 作为 query
        
        # 编码其他臂
        if other_states is not None and other_states.shape[1] > 0:
            other_feat = self.other_encoder(other_states)  # [B, M, d_model]
            
            # Cross-Attention
            for i, (attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
                # Attention
                x_norm = self.layer_norms[i*2](x)
                attn_out, _ = attn(x_norm, other_feat, other_feat)
                x = x + attn_out
                
                # FFN
                x_norm = self.layer_norms[i*2+1](x)
                x = x + ffn(x_norm)
        else:
            # 没有其他臂，只用 FFN
            for i, ffn in enumerate(self.ffn_layers):
                x_norm = self.layer_norms[i*2+1](x)
                x = x + ffn(x_norm)
        
        x = x.squeeze(1)  # [B, d_model]
        action = self.action_head(x) * self.action_scale
        
        return action


def create_model(model_type='multi_arm', **kwargs):
    """工厂函数"""
    if model_type == 'multi_arm':
        return MultiArmTransformer(**kwargs)
    elif model_type == 'multi_arm_v2':
        return MultiArmTransformerV2(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    print("Testing MultiArmTransformer...")
    model = MultiArmTransformer(d_model=256, nhead=8, num_layers=6)
    
    B, N = 4, 3  # 4 batch, 3 arms
    current = torch.randn(B, N, 6)
    goal = torch.randn(B, N, 6)
    
    actions = model(current, goal)
    print(f"  Input: current {current.shape}, goal {goal.shape}")
    print(f"  Output: {actions.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting single arm interface...")
    action = model.forward_single_arm(current[:, 0], goal[:, 0], 
                                       current[:, 1:], goal[:, 1:])
    print(f"  Output: {action.shape}")
    
    print("\nTesting MultiArmTransformerV2...")
    model_v2 = MultiArmTransformerV2(d_model=256, nhead=8, num_layers=4)
    action = model_v2(current[:, 0], goal[:, 0], current[:, 1:])
    print(f"  Output: {action.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model_v2.parameters()):,}")
