"""
测试 train_followir.py 的显存占用
"""
import os
import torch
import torch.nn as nn

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda:0'

print("=" * 60)
print("测试 train_followir.py 的显存占用")
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
print("\n📥 加载模型 (lightonai/GTE-ModernColBERT-v1)...")
from pylate import models, utils, losses

model = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
print(f"✅ 模型加载完成")
print(f"   模型设备: {model.device}")
print(f"   显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 创建损失函数
print("\n🔧 创建 Contrastive 损失...")
train_loss = losses.Contrastive(model=model)
print(f"✅ 损失函数创建完成")
print(f"   显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 创建测试数据
print("\n📝 创建测试数据...")
batch_size = 32
query_texts = ["This is a test query"] * batch_size
pos_texts = ["This is a positive document"] * batch_size
neg_texts = ["This is a negative document"] * batch_size

# Tokenize
query_tokens = model.tokenize(query_texts, is_query=True, pad=True)
pos_tokens = model.tokenize(pos_texts, is_query=False, pad=True)
neg_tokens = model.tokenize(neg_texts, is_query=False, pad=True)

# 将数据移到 GPU
query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
pos_tokens = {k: v.to(device) for k, v in pos_tokens.items()}
neg_tokens = {k: v.to(device) for k, v in neg_tokens.items()}

print(f"   Batch size: {batch_size}")
print(f"   Query shape: {query_tokens['input_ids'].shape}")
print(f"   Pos shape: {pos_tokens['input_ids'].shape}")
print(f"   Neg shape: {neg_tokens['input_ids'].shape}")
print(f"   显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# 前向传播
print("\n🚀 执行前向传播 (Contrastive Loss)...")
sentence_features = [query_tokens, pos_tokens, neg_tokens]
loss = train_loss(sentence_features=sentence_features)

print(f"✅ 前向传播完成")
print(f"   Loss: {loss.item():.6f}")
print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"   峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

# 反向传播
print("\n🔄 执行反向传播...")
loss.backward()

print(f"✅ 反向传播完成")
print(f"   当前显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"   峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
