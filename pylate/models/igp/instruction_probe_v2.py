"""
InstructionProbe Module V2 - 改进版指令引导探针模块

增加参数量版本，包含：
- 输入/输出投影层
- 可配置的编码器层数
- 手动实现的多头注意力
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class InstructionProbeV2(nn.Module):
    """
    指令引导探针 V2 (Instruction-Guided Probe V2)
    
    改进版本，增加参数量，提高表达能力。
    
    参数:
        hidden_size (int): 隐藏层维度
        num_heads (int): 注意力头数
        num_layers (int): Transformer编码器层数
        dropout (float): Dropout 概率
    """
    
    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
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
        
        # 输入投影层（增加参数量）
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        # 上下文编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        
        # 输出投影层（增加参数量）
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 多头注意力参数（手动实现）
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # 温度参数，控制注意力分布的锐度
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, query_embeddings, attention_mask):
        """
        前向传播
        
        参数:
            query_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        
        返回:
            inst_vec: [batch_size, hidden_size]
            attn_logits: [batch_size, seq_len]
            attn_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, hidden_size = query_embeddings.shape
        
        # 输入投影
        query_embeddings = self.input_proj(query_embeddings)
        
        # 扩展探针 token
        probe = self.probe_token.expand(batch_size, -1, -1).to(query_embeddings.device)
        
        # 拼接探针和输入
        context_input = torch.cat([probe, query_embeddings], dim=1)
        extended_mask = torch.cat(
            [torch.ones(batch_size, 1, device=attention_mask.device), attention_mask],
            dim=1
        )
        padding_mask = ~extended_mask.bool()
        
        # 上下文编码
        encoded = self.context_encoder(context_input, src_key_padding_mask=padding_mask)
        probe_output = encoded[:, 0, :]
        context_encoded = encoded[:, 1:, :]
        
        # 手动计算多头注意力
        Q = self.q_proj(probe_output).unsqueeze(1)
        K = self.k_proj(context_encoded)
        V = self.v_proj(context_encoded)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
        mask = attention_mask.unsqueeze(1).float()
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_logits = scores.squeeze(1)
        # 使用带温度的 softmax，温度越小分布越尖锐
        attn_weights = torch.softmax(attn_logits / self.temperature.abs(), dim=-1)
        
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        inst_vec = torch.sum(attn_weights_expanded * V, dim=1)
        
        # 输出投影
        inst_vec = self.output_proj(inst_vec)
        inst_vec = self.layer_norm(probe_output + inst_vec)
        
        return inst_vec, attn_logits, attn_weights
