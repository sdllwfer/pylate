"""
测试大参数量配置
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from pylate import models
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate


def test_large_params():
    """测试大参数量配置"""
    print("=" * 60)
    print("测试大参数量配置")
    print("=" * 60)
    
    print("\n1. 加载基础模型...")
    base_model = models.ColBERT(model_name_or_path='answerdotai/ModernBERT-base')
    hidden_size = base_model[0].get_word_embedding_dimension()
    print(f"   ✅ 基础模型加载成功")
    print(f"   hidden_size: {hidden_size}")
    
    print("\n2. 创建大参数量 IGP 模块...")
    
    # Probe: 6 层
    probe = InstructionProbe(
        hidden_size=hidden_size,
        num_heads=8,
        num_layers=6,  # 大参数量
        dropout=0.1,
    )
    probe_params = sum(p.numel() for p in probe.parameters())
    print(f"   ✅ InstructionProbe 创建成功 (num_layers=6)")
    print(f"      参数量: {probe_params:,}")
    
    # Adapter: bottleneck_dim=512
    adapter = IGPAdapter(
        hidden_size=hidden_size,
        bottleneck_dim=512,  # 大参数量
        dropout=0.1,
        input_dim=hidden_size * 2,
    )
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"   ✅ IGPAdapter 创建成功 (bottleneck_dim=512)")
    print(f"      参数量: {adapter_params:,}")
    
    # Gate
    gate = RatioGate(
        hidden_size=hidden_size,
        max_ratio=0.2,
        use_dynamic=False,
    )
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"   ✅ RatioGate 创建成功")
    print(f"      参数量: {gate_params:,}")
    
    total_igp = probe_params + adapter_params + gate_params
    print(f"\n   📊 IGP 总参数量: {total_igp:,}")
    
    # 与 BERT 对比
    bert_params = 110_000_000
    ratio = total_igp / bert_params * 100
    print(f"   📊 占 BERT 比例: {ratio:.2f}%")
    
    print("\n3. 测试前向传播...")
    # 创建测试输入
    test_embeddings = torch.randn(2, 10, hidden_size)
    test_mask = torch.ones(2, 10)
    
    inst_vec, attn_logits, attn_weights = probe(test_embeddings, test_mask)
    print(f"   ✅ Probe 前向传播成功")
    print(f"      inst_vec shape: {inst_vec.shape}")
    
    # 测试 Adapter (需要 3D 输入)
    query_states = torch.randn(2, 10, hidden_size)  # [batch, seq_len, hidden]
    output, delta = adapter(query_states, inst_vec)
    print(f"   ✅ Adapter 前向传播成功")
    print(f"      output shape: {output.shape}")
    print(f"      delta shape: {delta.shape}")
    
    # 测试 Gate (需要 3D 输入)
    gate_output, gate_ratio = gate(query_states, inst_vec.unsqueeze(1).expand(-1, query_states.size(1), -1))
    print(f"   ✅ Gate 前向传播成功")
    print(f"      gate_output shape: {gate_output.shape}")
    print(f"      gate_ratio: {gate_ratio.mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_large_params()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
