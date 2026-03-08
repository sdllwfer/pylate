"""
计算 IGP 模块的参数量
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())

def calculate_igp_params():
    """计算 IGP 各模块参数量"""
    print("=" * 60)
    print("IGP 模块参数量分析")
    print("=" * 60)
    
    hidden_size = 768
    
    # 1. InstructionProbe 参数量
    print("\n1. InstructionProbe")
    print("-" * 60)
    
    for num_heads in [4, 8, 16]:
        for num_layers in [2, 4, 6]:
            probe = InstructionProbe(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=0.1
            )
            # 修改层数
            probe.context_encoder.num_layers = num_layers
            
            params = count_parameters(probe)
            print(f"  num_heads={num_heads}, num_layers={num_layers}: {params:,} 参数")
    
    # 2. IGPAdapter 参数量
    print("\n2. IGPAdapter")
    print("-" * 60)
    
    for bottleneck_dim in [64, 128, 256, 512]:
        adapter = IGPAdapter(
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            input_dim=hidden_size*2
        )
        params = count_parameters(adapter)
        print(f"  bottleneck_dim={bottleneck_dim}: {params:,} 参数")
    
    # 3. RatioGate 参数量
    print("\n3. RatioGate")
    print("-" * 60)
    gate = RatioGate(hidden_size=hidden_size, max_ratio=0.2)
    params = count_parameters(gate)
    print(f"  RatioGate: {params:,} 参数")
    
    # 4. 总参数量（当前配置）
    print("\n4. 当前配置总参数量")
    print("-" * 60)
    
    configs = [
        {"probe_heads": 4, "probe_layers": 2, "bottleneck": 64, "name": "当前配置"},
        {"probe_heads": 8, "probe_layers": 4, "bottleneck": 128, "name": "中等配置"},
        {"probe_heads": 16, "probe_layers": 6, "bottleneck": 256, "name": "大配置"},
        {"probe_heads": 16, "probe_layers": 8, "bottleneck": 512, "name": "超大配置"},
    ]
    
    for config in configs:
        probe = InstructionProbe(hidden_size=hidden_size, num_heads=config["probe_heads"])
        probe.context_encoder.num_layers = config["probe_layers"]
        adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=config["bottleneck"], input_dim=hidden_size*2)
        gate = RatioGate(hidden_size=hidden_size)
        
        total = count_parameters(probe) + count_parameters(adapter) + count_parameters(gate)
        print(f"  {config['name']}: {total:,} 参数")
    
    # 5. 与 BERT 对比
    print("\n5. 与 BERT 参数量对比")
    print("-" * 60)
    bert_params = 110_000_000  # BERT-base 约 110M
    print(f"  BERT-base: {bert_params:,} 参数")
    
    for config in configs:
        probe = InstructionProbe(hidden_size=hidden_size, num_heads=config["probe_heads"])
        probe.context_encoder.num_layers = config["probe_layers"]
        adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=config["bottleneck"], input_dim=hidden_size*2)
        gate = RatioGate(hidden_size=hidden_size)
        
        total = count_parameters(probe) + count_parameters(adapter) + count_parameters(gate)
        ratio = total / bert_params * 100
        print(f"  {config['name']}: {total:,} 参数 ({ratio:.2f}% of BERT)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    calculate_igp_params()
