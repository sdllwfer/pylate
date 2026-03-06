"""
IGP Adapter Module - 指令引导适配器模块

该模块用于在Transformer层之间注入指令相关的可训练参数，
增强模型对指令的感知能力。

设计规范:
- 使用瓶颈结构 (bottleneck) 减少参数量
- 残差连接保持原始表示
- 可通过配置启用/禁用
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class IGPAdapter(nn.Module):
    """
    IGP 适配器 (Instruction-Guided Probe Adapter)
    
    使用瓶颈结构，在保持原始表示的同时注入指令相关知识。
    
    参数:
        hidden_size (int): 隐藏层维度
        bottleneck_dim (int): 瓶颈层维度 (通常远小于 hidden_size)
        dropout (float): Dropout 概率
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        input_dim: int = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        
        # 如果没有指定 input_dim，默认是 hidden_size (用于残差连接模式)
        # 如果指定了 input_dim，使用 input_dim * 2 (用于 [Query, Inst] 拼接模式)
        effective_input_dim = input_dim if input_dim is not None else hidden_size
        
        # 下投影层: input_dim * 2 -> bottleneck_dim (支持 [Query, Inst] 拼接)
        self.down_project = nn.Linear(effective_input_dim * 2, bottleneck_dim)
        
        # 上投影层: bottleneck_dim -> hidden_size
        self.up_project = nn.Linear(bottleneck_dim, hidden_size)
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.xavier_uniform_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        instruction_vector: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        按照 IGP 方案设计:
        - 输入: [Query_vec, Inst_vec] 拼接
        - 输出: delta 向量（与 Query 维度相同）
        
        参数:
            hidden_states: 隐藏状态 [batch_size, seq_len, hidden_size]
            instruction_vector: 指令向量 [batch_size, hidden_size]，可选
        
        返回:
            output: 输出 [batch_size, seq_len, hidden_size]
            delta: 偏移向量 [batch_size, seq_len, hidden_size]
        """
        # 记录原始残差
        residual = hidden_states
        
        # 如果提供了指令向量，按照方案拼接 [Query, Inst]
        if instruction_vector is not None:
            # 扩展指令向量到序列维度
            # [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
            inst_vec_expanded = instruction_vector.unsqueeze(1)
            # 拼接: [batch_size, seq_len+1, hidden_size]
            combined = torch.cat([hidden_states, inst_vec_expanded], dim=1)
        else:
            combined = hidden_states
        
        # 下投影
        combined_bottleneck = self.down_project(combined)
        combined_bottleneck = self.activation(combined_bottleneck)
        combined_bottleneck = self.dropout(combined_bottleneck)
        
        # 上投影 (只取前 seq_len 个位置)
        combined_output = self.up_project(combined_bottleneck)
        combined_output = self.dropout(combined_output)
        
        # 只取原始序列长度的输出
        output = combined_output[:, :hidden_states.size(1), :]
        
        # 计算 delta（偏移向量）
        delta = output - residual
        
        # LayerNorm 和残差连接
        output = self.layer_norm(output + residual)
        
        return output, delta


class IGPAdapterConfig:
    """IGPAdapter 配置类"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
    ):
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.dropout = dropout
    
    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'bottleneck_dim': self.bottleneck_dim,
            'dropout': self.dropout,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
