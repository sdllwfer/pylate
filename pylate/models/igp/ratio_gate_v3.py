"""
Ratio Gate Module V3 - 动态感知门控 (Context-Aware Dynamic Gating)

带L1稀疏正则化的动态门控模块，实现"按需开启"机制。
"""

import torch
import torch.nn as nn


class RatioGateV3(nn.Module):
    """
    动态感知门控 V3 (Context-Aware Dynamic Gating)
    
    核心改进：
    1. 使用轻量级MLP根据输入动态计算门控值（Instance-level）
    2. 每个样本有独立的门控值，实现"按需开启"
    3. 配合L1稀疏正则化，强制模型在无指令时关闭门控
    
    参数:
        hidden_size (int): 隐藏层维度
        max_ratio (float): 最大比例值
    """
    
    def __init__(self, hidden_size: int = 768, max_ratio: float = 0.2):
        super().__init__()
        
        self.max_ratio = max_ratio
        self.hidden_size = hidden_size
        
        # 轻量级动态门控网络（2层MLP）
        # 输入：instruction向量（或query+instruction拼接）
        # 输出：单个标量门控值
        self.dynamic_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, query_vec, inst_vec):
        """
        前向传播，计算动态门控值
        
        参数:
            query_vec: [batch_size, hidden_size] - Query表征（可选，用于辅助判断）
            inst_vec: [batch_size, hidden_size] - Instruction向量
        
        返回:
            gate_ratio: [batch_size] - 每个样本的门控值
        """
        # 使用instruction向量作为输入（也可以考虑拼接query_vec）
        # 让网络根据instruction的质量决定是否开启门控
        gate_logit = self.dynamic_gate(inst_vec).squeeze(-1)  # [batch_size]
        
        # Sigmoid约束到(0,1)，然后缩放到(0, max_ratio)
        gate_ratio = self.max_ratio * torch.sigmoid(gate_logit)
        
        return gate_ratio
    
    def get_l1_penalty(self, query_vec, inst_vec):
        """
        计算L1稀疏正则化惩罚项
        
        用于强制门控在无指令时趋向于0
        
        返回:
            l1_penalty: scalar - 当前batch的门控L1惩罚
        """
        gate_ratio = self.forward(query_vec, inst_vec)
        # L1惩罚：门控值的绝对值均值
        l1_penalty = gate_ratio.abs().mean()
        return l1_penalty


class RatioGateV3WithQuery(nn.Module):
    """
    动态感知门控 V3 变体 - 使用Query+Instruction拼接作为输入
    
    如果希望利用Query的语义信息辅助判断，可以使用这个版本
    """
    
    def __init__(self, hidden_size: int = 768, max_ratio: float = 0.2):
        super().__init__()
        
        self.max_ratio = max_ratio
        self.hidden_size = hidden_size
        
        # 输入维度是hidden_size * 2（query + instruction拼接）
        self.dynamic_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, query_vec, inst_vec):
        """
        前向传播
        
        参数:
            query_vec: [batch_size, hidden_size]
            inst_vec: [batch_size, hidden_size]
        """
        # 拼接query和instruction向量
        combined = torch.cat([query_vec, inst_vec], dim=-1)  # [batch_size, hidden_size * 2]
        
        gate_logit = self.dynamic_gate(combined).squeeze(-1)  # [batch_size]
        gate_ratio = self.max_ratio * torch.sigmoid(gate_logit)
        
        return gate_ratio
    
    def get_l1_penalty(self, query_vec, inst_vec):
        """计算L1稀疏正则化惩罚项"""
        gate_ratio = self.forward(query_vec, inst_vec)
        l1_penalty = gate_ratio.abs().mean()
        return l1_penalty
