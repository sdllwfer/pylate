"""
测试 V2 版本训练流程
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from pylate import models
from pylate.models.igp import InstructionProbeV2, IGPAdapterV2, RatioGateV2
from igp_losses import IGPLoss
from data_utils import IGPColBERTCollator
from datasets import Dataset


class IGPColBERTModelV2(nn.Module):
    """IGP ColBERT Model V2"""
    
    def __init__(
        self,
        base_model,
        probe_num_layers: int = 6,
        bottleneck_dim: int = 512,
        adapter_num_layers: int = 3,
        max_ratio: float = 0.2,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.hidden_size = base_model[0].get_word_embedding_dimension()
        
        # 创建 V2 版本的 IGP 模块
        self.probe = InstructionProbeV2(
            hidden_size=self.hidden_size,
            num_heads=8,
            num_layers=probe_num_layers,
            dropout=0.1,
        )
        
        self.adapter = IGPAdapterV2(
            hidden_size=self.hidden_size,
            bottleneck_dim=bottleneck_dim,
            num_layers=adapter_num_layers,
            dropout=0.1,
        )
        
        self.gate = RatioGateV2(
            hidden_size=self.hidden_size,
            max_ratio=max_ratio,
        )
        
        # 保存 tokenizer 引用
        self.tokenizer = base_model.tokenizer
        
    def forward(self, features):
        """前向传播"""
        return self.base_model(features)
    
    def encode(self, sentences, is_query=True, **kwargs):
        """编码句子"""
        return self.base_model.encode(sentences, is_query=is_query, **kwargs)


def test_v2_training():
    """测试 V2 版本训练流程"""
    print("=" * 60)
    print("测试 V2 版本训练流程")
    print("=" * 60)
    
    print("\n1. 加载基础模型...")
    base_model = models.ColBERT(model_name_or_path='answerdotai/ModernBERT-base')
    print(f"   ✅ 基础模型加载成功")
    
    print("\n2. 创建 IGP V2 模型...")
    model = IGPColBERTModelV2(
        base_model=base_model,
        probe_num_layers=6,
        bottleneck_dim=512,
        adapter_num_layers=3,
        max_ratio=0.2,
    )
    print(f"   ✅ IGP V2 模型创建成功")
    
    # 计算参数量
    probe_params = sum(p.numel() for p in model.probe.parameters())
    adapter_params = sum(p.numel() for p in model.adapter.parameters())
    gate_params = sum(p.numel() for p in model.gate.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n   📊 参数量统计:")
    print(f"      Probe: {probe_params:,}")
    print(f"      Adapter: {adapter_params:,}")
    print(f"      Gate: {gate_params:,}")
    print(f"      IGP 总计: {probe_params + adapter_params + gate_params:,}")
    print(f"      总参数量: {total_params:,}")
    
    print("\n3. 测试模型编码...")
    test_sentences = ['This is a test query', 'Another test query']
    output = model.encode(test_sentences, is_query=True)
    print(f"   ✅ 编码成功，输出形状: {output[0].shape}")
    
    print("\n4. 创建损失函数...")
    loss_fn = IGPLoss(model=model, aux_loss_weight=0.0)
    print(f"   ✅ 损失函数创建成功")
    
    print("\n5. 测试损失计算...")
    # 创建测试数据
    collator = IGPColBERTCollator(tokenizer=model.tokenizer, max_query_length=32)
    test_data = [
        {'query': 'test query 1', 'positive': 'positive doc 1', 'negative': 'negative doc 1'},
        {'query': 'test query 2', 'positive': 'positive doc 2', 'negative': 'negative doc 2'},
    ]
    batch = collator(test_data)
    
    # 计算损失
    loss = loss_fn(batch, labels=None)
    print(f"   ✅ 损失计算成功: {loss.item():.4f}")
    
    print("\n6. 测试反向传播...")
    loss.backward()
    print(f"   ✅ 反向传播成功")
    
    # 检查梯度
    has_grad = False
    for name, param in model.probe.named_parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    if has_grad:
        print(f"   ✅ Probe 模块有梯度")
    else:
        print(f"   ❌ Probe 模块没有梯度")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_v2_training()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
