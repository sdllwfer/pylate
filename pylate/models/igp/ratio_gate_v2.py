"""
Ratio Gate Module V2 - 改进版门控模块

增加参数量版本，使用 MLP 计算 gate 值
"""

import torch
import torch.nn as nn


class RatioGateV2(nn.Module):
    """
    比例门控 V2 (Ratio Gate V2)
    
    改进版本，使用 MLP 计算 gate 值，增加参数量。
    
    参数:
        hidden_size (int): 隐藏层维度
        max_ratio (float): 最大比例值
    """
    
    def __init__(self, hidden_size: int = 768, max_ratio: float = 0.2):
        super().__init__()
        
        self.max_ratio = max_ratio
        
        # 使用 MLP 计算 gate 值
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, query_vec, inst_vec):
        """
        前向传播
        
        参数:
            query_vec: [batch_size, hidden_size]
            inst_vec: [batch_size, hidden_size]
        
        返回:
            gate: [batch_size]
        """
        # 拼接 query 和 instruction
        combined = torch.cat([query_vec, inst_vec], dim=-1)
        
        # 计算 gate 值
        gate = self.gate_mlp(combined)
        
        # 限制最大比例
        gate = gate * self.max_ratio
        
        return gate.squeeze(-1)
