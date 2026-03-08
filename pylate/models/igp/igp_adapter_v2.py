"""
IGP Adapter Module V2 - 改进版指令引导适配器模块

增加参数量版本，包含：
- 输入投影层
- 多层瓶颈结构
- 输出投影层
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class IGPAdapterV2(nn.Module):
    """
    IGP 适配器 V2 (Instruction-Guided Probe Adapter V2)
    
    改进版本，增加参数量，使用多层瓶颈结构。
    
    参数:
        hidden_size (int): 隐藏层维度
        bottleneck_dim (int): 瓶颈层维度
        num_layers (int): 瓶颈层数
        dropout (float): Dropout 概率
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        bottleneck_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        
        # 输入投影
        self.input_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 多层瓶颈结构
        layers = []
        in_dim = hidden_size
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, bottleneck_dim),
                nn.LayerNorm(bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = bottleneck_dim
        
        self.bottleneck = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query_vec, inst_vec, concat_dim: str = "hidden"):
        """
        前向传播
        
        参数:
            query_vec: [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]
            inst_vec: [batch_size, hidden_size]
            concat_dim: 拼接维度，"hidden" 表示在 hidden 维度拼接（每个 token 都能看到指令）
        
        返回:
            output: [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]
            delta: 偏移向量（与 query_vec 维度相同）
        """
        # 记录原始输入用于计算 delta
        residual = query_vec
        
        # 处理 3D 输入（[batch_size, seq_len, hidden_size]）
        if query_vec.dim() == 3:
            batch_size, seq_len, hidden_size = query_vec.shape
            if concat_dim == "hidden":
                # 在 hidden 维度拼接
                # inst_vec: [batch_size, hidden_size] -> [batch_size, seq_len, hidden_size]
                inst_vec_expanded = inst_vec.unsqueeze(1).expand(-1, seq_len, -1)
                combined = torch.cat([query_vec, inst_vec_expanded], dim=-1)
            else:
                # 在 seq_len 维度拼接
                inst_vec_expanded = inst_vec.unsqueeze(1)
                combined = torch.cat([query_vec, inst_vec_expanded], dim=1)
        else:
            # 2D 输入（[batch_size, hidden_size]）
            combined = torch.cat([query_vec, inst_vec], dim=-1)
        
        # 输入投影
        x = self.input_proj(combined)
        
        # 瓶颈层
        delta = self.bottleneck(x)
        
        # 输出投影
        delta = self.output_proj(delta)
        
        # 残差连接
        output = self.layer_norm(x + self.dropout(delta))
        
        # 计算 delta（偏移向量）
        delta_out = output - residual
        
        return output, delta_out
