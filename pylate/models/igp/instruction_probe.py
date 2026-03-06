"""
InstructionProbe Module - 指令引导探针模块

该模块用于从输入序列中提取指令特征，通过可学习的探针 token 与上下文进行注意力交互，
生成指令向量 (inst_vec)。使用 Sigmoid 激活而非 Softmax，以保留多词指令的概率分布。

设计规范:
- 不使用 nn.MultiheadAttention (Softmax 会稀释多词指令概率)
- 手动计算点积注意力
- 返回 inst_vec, attn_logits, attn_weights 三个值
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class InstructionProbe(nn.Module):
    """
    指令引导探针 (Instruction-Guided Probe)
    
    通过可学习的探针 token 与上下文进行注意力交互，提取指令特征。
    
    参数:
        hidden_size (int): 隐藏层维度
        num_heads (int): 注意力头数
        dropout (float): Dropout 概率
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 验证维度兼容性
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) 必须能被 num_heads ({num_heads}) 整除"
        
        # 可学习的探针 token [1, 1, hidden_size]
        self.probe_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # 简单的上下文编码器 (使用 TransformerEncoderLayer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN 架构，更利于训练稳定性
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(hidden_size),
        )
        
        # 输出投影和 LayerNorm
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.probe_token, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            query_embeddings: 查询的 token embeddings [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len], 1 表示有效位置, 0 表示 padding
        
        返回:
            inst_vec: 指令向量 [batch_size, hidden_size]
            attn_logits: 未归一化的注意力分数 [batch_size, seq_len]
            attn_weights: Sigmoid 注意力权重 [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_size = query_embeddings.shape
        
        # 1. 扩展探针 token 到 batch 大小 [batch_size, 1, hidden_size]
        probe = self.probe_token.expand(batch_size, -1, -1)
        
        # 2. 使用上下文编码器处理输入
        # 将探针作为第一个 token，与输入序列拼接
        # [batch_size, 1+seq_len, hidden_size]
        context_input = torch.cat([probe, query_embeddings], dim=1)
        
        # 扩展 attention_mask 以包含探针位置 (探针位置始终可见)
        # [batch_size, 1+seq_len]
        extended_mask = torch.cat(
            [torch.ones(batch_size, 1, device=attention_mask.device), attention_mask],
            dim=1
        )
        
        # 创建 padding mask: True 表示需要 mask 掉的位置
        padding_mask = ~extended_mask.bool()  # [batch_size, 1+seq_len]
        
        # 通过上下文编码器
        encoded = self.context_encoder(
            context_input,
            src_key_padding_mask=padding_mask
        )  # [batch_size, 1+seq_len, hidden_size]
        
        # 提取探针位置的输出
        probe_output = encoded[:, 0, :]  # [batch_size, hidden_size]
        
        # 3. 手动计算点积注意力
        # 使用探针输出作为 Query，输入上下文作为 Key 和 Value
        # 由于已经过 Transformer 编码，我们直接使用编码后的上下文
        context_encoded = encoded[:, 1:, :]  # [batch_size, seq_len, hidden_size]
        
        # 计算 Query-Key 点积并归一化
        # Q: [batch_size, 1, head_dim * num_heads] -> 扩展 probe_output
        # K: [batch_size, seq_len, head_dim * num_heads]
        
        Q = probe_output.unsqueeze(1)  # [batch_size, 1, hidden_size]
        K = context_encoded  # [batch_size, seq_len, hidden_size]
        V = context_encoded  # [batch_size, seq_len, hidden_size]
        
        # 点积注意力分数
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)  # [batch_size, 1, seq_len]
        
        # 4. Mask 处理：将 padding 位置的分数设为 -inf
        # 扩展 attention_mask 以匹配 scores
        mask = attention_mask.unsqueeze(1).float()  # [batch_size, 1, seq_len]
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. 获取未归一化的 logits
        attn_logits = scores.squeeze(1)  # [batch_size, seq_len]
        
        # 6. Sigmoid 激活获取权重 (不使用 Softmax，以保留多词指令概率)
        attn_weights = torch.sigmoid(attn_logits)  # [batch_size, seq_len]
        
        # 7. 加权求和获取指令向量
        # 扩展权重以匹配 V 的维度
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        inst_vec = torch.sum(attn_weights_expanded * V, dim=1)  # [batch_size, hidden_size]
        
        # 8. 残差连接和 LayerNorm
        inst_vec = self.layer_norm(probe_output + self.output_proj(inst_vec))
        
        return inst_vec, attn_logits, attn_weights


class InstructionProbeConfig:
    """InstructionProbe 配置类"""
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
    
    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'dropout': self.dropout,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
