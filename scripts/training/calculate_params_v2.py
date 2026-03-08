"""
计算改进版 IGP 模块的参数量
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
import math

class ImprovedInstructionProbe(nn.Module):
    """改进的 InstructionProbe - 增加参数量"""
    
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
        
        # 可学习的探针 token
        self.probe_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # 输入投影层（增加参数量）
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
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
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 多头注意力参数（手动实现）
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, query_embeddings, attention_mask):
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
        attn_weights = torch.sigmoid(attn_logits)
        
        attn_weights_expanded = attn_weights.unsqueeze(-1)
        inst_vec = torch.sum(attn_weights_expanded * V, dim=1)
        
        # 输出投影
        inst_vec = self.output_proj(inst_vec)
        inst_vec = self.layer_norm(probe_output + inst_vec)
        
        return inst_vec, attn_logits, attn_weights


class ImprovedIGPAdapter(nn.Module):
    """改进的 IGPAdapter - 增加参数量"""
    
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
        
    def forward(self, query_vec, inst_vec):
        # 拼接 query 和 instruction
        combined = torch.cat([query_vec, inst_vec], dim=-1)
        
        # 输入投影
        x = self.input_proj(combined)
        
        # 瓶颈层
        delta = self.bottleneck(x)
        
        # 输出投影
        delta = self.output_proj(delta)
        
        # 残差连接
        output = self.layer_norm(x + self.dropout(delta))
        
        return output


class ImprovedRatioGate(nn.Module):
    """改进的 RatioGate - 增加参数量"""
    
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
        # 拼接 query 和 instruction
        combined = torch.cat([query_vec, inst_vec], dim=-1)
        
        # 计算 gate 值
        gate = self.gate_mlp(combined)
        
        # 限制最大比例
        gate = gate * self.max_ratio
        
        return gate.squeeze(-1)


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())


def calculate_improved_params():
    """计算改进版 IGP 参数量"""
    print("=" * 60)
    print("改进版 IGP 模块参数量分析")
    print("=" * 60)
    
    hidden_size = 768
    
    # 1. 改进版 InstructionProbe
    print("\n1. 改进版 InstructionProbe")
    print("-" * 60)
    
    for num_layers in [2, 4, 6, 8]:
        probe = ImprovedInstructionProbe(
            hidden_size=hidden_size,
            num_heads=8,
            num_layers=num_layers,
        )
        params = count_parameters(probe)
        print(f"  num_layers={num_layers}: {params:,} 参数")
    
    # 2. 改进版 IGPAdapter
    print("\n2. 改进版 IGPAdapter")
    print("-" * 60)
    
    for bottleneck_dim in [128, 256, 512]:
        for num_layers in [2, 3, 4]:
            adapter = ImprovedIGPAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                num_layers=num_layers,
            )
            params = count_parameters(adapter)
            print(f"  bottleneck={bottleneck_dim}, layers={num_layers}: {params:,} 参数")
    
    # 3. 改进版 RatioGate
    print("\n3. 改进版 RatioGate")
    print("-" * 60)
    gate = ImprovedRatioGate(hidden_size=hidden_size)
    params = count_parameters(gate)
    print(f"  ImprovedRatioGate: {params:,} 参数")
    
    # 4. 总参数量对比
    print("\n4. 总参数量对比")
    print("-" * 60)
    
    configs = [
        {"probe_layers": 4, "bottleneck": 256, "adapter_layers": 2, "name": "中等配置"},
        {"probe_layers": 6, "bottleneck": 512, "adapter_layers": 3, "name": "大配置"},
        {"probe_layers": 8, "bottleneck": 512, "adapter_layers": 4, "name": "超大配置"},
    ]
    
    bert_params = 110_000_000
    
    for config in configs:
        probe = ImprovedInstructionProbe(hidden_size=hidden_size, num_layers=config["probe_layers"])
        adapter = ImprovedIGPAdapter(hidden_size=hidden_size, bottleneck_dim=config["bottleneck"], num_layers=config["adapter_layers"])
        gate = ImprovedRatioGate(hidden_size=hidden_size)
        
        total = count_parameters(probe) + count_parameters(adapter) + count_parameters(gate)
        ratio = total / bert_params * 100
        print(f"  {config['name']}: {total:,} 参数 ({ratio:.2f}% of BERT)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    calculate_improved_params()
