"""
Ratio Gate Module V3 - 全局感知门控 (Global-Aware Dynamic Gating)

使用 Query 的全局表示来动态预测门控比例。
配合 L1 稀疏正则化，实现"按需开启"机制。
"""

import torch
import torch.nn as nn


class RatioGateV3(nn.Module):
    """
    全局感知门控 V3 (Global-Aware Dynamic Gating)
    
    核心改进：
    1. 使用轻量级 MLP 根据 Query 全局表示动态计算门控 logit
    2. 每个样本有独立的门控值，实现"按需开启"
    3. 门控 logit 在 wrapper 中经过 sigmoid 和 max_ratio 缩放
    4. 配合 L1 稀疏正则化，强制模型在无指令时关闭门控
    
    参数:
        hidden_size (int): 隐藏层维度
        max_ratio (float): 最大比例值 (在 wrapper 中使用)
    """
    
    def __init__(self, hidden_size: int = 768, max_ratio: float = 0.2):
        super().__init__()
        
        self.max_ratio = max_ratio
        self.hidden_size = hidden_size
        
        # 轻量级动态门控网络（2层MLP）
        # 输入：Q_global (Query 全局表示)
        # 输出：gate_logits (尚未经过 sigmoid)
        self.dynamic_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, Q_global):
        """
        前向传播，计算门控 logit
        
        参数:
            Q_global: [batch_size, hidden_size] - Query 的全局表示
        
        返回:
            gate_logits: [batch_size, 1] - 门控 logit (尚未缩放)
        """
        # 使用 Q_global 预测门控 logit
        gate_logits = self.dynamic_gate(Q_global)  # [batch, 1]
        
        return gate_logits


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
