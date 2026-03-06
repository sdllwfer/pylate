"""
Ratio Gate Module - 门控机制模块

该模块用于控制原始表示和指令增强表示的融合比例。
通过可学习的门控参数，动态调整指令信息对最终表示的贡献程度。

设计规范:
- 限制门控最大值，防止指令破坏原语义
- 支持动态门控和静态门控两种模式
- 可通过配置启用/禁用
"""

import torch
import torch.nn as nn
from typing import Optional


class RatioGate(nn.Module):
    """
    比率门控 (Ratio Gate)
    
    用于控制原始表示和指令增强表示的融合比例。
    
    参数:
        hidden_size (int): 隐藏层维度
        max_ratio (float): 门控最大比率，默认 0.2，防止指令破坏原语义
        use_dynamic (bool): 是否使用动态门控 (基于输入预测)
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        max_ratio: float = 0.2,
        use_dynamic: bool = False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_ratio = max_ratio
        self.use_dynamic = use_dynamic
        
        # 可学习的门控参数 (初始值为 0，sigmoid(0) = 0.5，初始比率 = 0.5 * max_ratio)
        self.ratio_gate = nn.Parameter(torch.tensor(0.0))
        
        if use_dynamic:
            # 动态门控: 基于输入预测门控值
            self.gate_predictor = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )
    
    def forward(
        self,
        original_vec: torch.Tensor,
        instruction_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            original_vec: 原始表示 [batch_size, ..., hidden_size]
            instruction_vec: 指令增强表示 [batch_size, ..., hidden_size]
        
        返回:
            fused_vec: 融合后的表示 [batch_size, ..., hidden_size]
            current_ratio: 当前使用的门控比率 (标量或张量)
        """
        if self.use_dynamic:
            # 动态门控: 拼接原始表示和指令表示，预测门控值
            combined = torch.cat([original_vec, instruction_vec], dim=-1)
            dynamic_gate = self.gate_predictor(combined)  # [batch_size, ..., 1]
            # 限制动态门控范围
            current_ratio = self.max_ratio * torch.sigmoid(dynamic_gate)
        else:
            # 静态门控: 使用可学习参数
            current_ratio = self.max_ratio * torch.sigmoid(self.ratio_gate)
        
        # 确保 current_ratio 可以广播
        if current_ratio.dim() < original_vec.dim():
            # 添加缺失的维度以支持广播
            for _ in range(original_vec.dim() - current_ratio.dim()):
                current_ratio = current_ratio.unsqueeze(1)
        
        # 弹性门控融合: Q_hat = Q_origin + ratio * inst_vec
        # 扩展 instruction_vec 到与 original_vec 相同的维度
        if instruction_vec.dim() < original_vec.dim():
            # 如果 instruction_vec 缺少序列维度
            if instruction_vec.dim() == original_vec.dim() - 1:
                instruction_vec = instruction_vec.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        fused_vec = original_vec + current_ratio * instruction_vec
        
        # 分离梯度，返回.detach() 的 ratio 用于记录
        current_ratio_detached = current_ratio.detach()
        
        return fused_vec, current_ratio_detached
    
    def get_current_ratio(self) -> float:
        """获取当前门控比率 (用于日志记录)"""
        return float(self.max_ratio * torch.sigmoid(self.ratio_gate).item())


class RatioGateConfig:
    """RatioGate 配置类"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        max_ratio: float = 0.2,
        use_dynamic: bool = False,
    ):
        self.hidden_size = hidden_size
        self.max_ratio = max_ratio
        self.use_dynamic = use_dynamic
    
    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'max_ratio': self.max_ratio,
            'use_dynamic': self.use_dynamic,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
