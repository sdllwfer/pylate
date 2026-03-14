"""
测试 GPU 显存占用情况
"""
import os
import torch
import torch.nn as nn

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda:0'

print("=" * 60)
print("测试 GPU 显存占用")
print("=" * 60)

# 检查 GPU 可用性
if not torch.cuda.is_available():
    print("❌ CUDA 不可用")
    exit(1)

print(f"✅ CUDA 可用")
print(f"   当前设备: {torch.cuda.current_device()}")
print(f"   设备名称: {torch.cuda.get_device_name(0)}")

# 获取初始显存占用
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
initial_memory = torch.cuda.memory_allocated() / 1024**2
print(f"   初始显存占用: {initial_memory:.2f} MB")

# 导入模型
print("\n📥 加载模型...")
from pylate import models
from pylate.models.igp import IGPColBERTWrapper, InstructionProbeV2, IGPAdapterV2, RatioGateV3

base_model = models.ColBERT(model_name_or_path="lightonai/ColBERT-Zero", device=device)
print(f"✅ 基础模型加载完成")
print(f"   显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 创建 IGP 模块
print("\n🔧 创建 IGP 模块...")
probe = InstructionProbeV2(hidden_size=768).to(device)
adapter = IGPAdapterV2(hidden_size=768).to(device)
gate = RatioGateV3(hidden_size=768, max_ratio=0.2).to(device)
print(f"✅ IGP 模块创建完成")
print(f"   显存占用: {torch.cuda.memory_allocated() / 1024**1024:.2f} MB")

# 创建 Wrapper
print("\n📦 创建 IGPColBERTWrapper...")
igp_model = IGPColBERTWrapper(
    base_model=base_model,
    probe=probe,
    adapter=adapter,
    gate=gate,
)
print(f"✅ Wrapper 创建完成")
print(f"   显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 创建测试数据
print("\n📝 创建测试数据...")
batch_size = 64
seq_len = 32

query_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
query_attention_mask = torch.ones(batch_size, seq_len).to(device)
instruction_mask = torch.zeros(batch_size, seq_len).to(device)

print(f"   Batch size: {batch_size}")
print(f"   Sequence length: {seq_len}")
print(f"   输入数据显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 前向传播
print("\n🚀 执行前向传播...")
result = igp_model(
    query_input_ids=query_input_ids,
    query_attention_mask=query_attention_mask,
    instruction_mask=instruction_mask,
)

print(f"✅ 前向传播完成")
print(f"   输出 shape: {result['token_embeddings'].shape}")
print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"   峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# 模拟反向传播
print("\n🔄 执行反向传播...")
loss = result['token_embeddings'].mean()
loss.backward()

print(f"✅ 反向传播完成")
print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"   峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
