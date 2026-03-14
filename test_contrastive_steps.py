"""
测试 Contrastive 损失的各个步骤的显存占用
"""
import os
import torch
import torch.nn.functional as F

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda:0'

print("=" * 60)
print("测试 Contrastive 损失各步骤的显存占用")
print("=" * 60)

# 检查 GPU 可用性
if not torch.cuda.is_available():
    print("❌ CUDA 不可用")
    exit(1)

print(f"✅ CUDA 可用")

# 获取初始显存占用
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

def print_memory(step_name):
    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"   [{step_name}] 当前: {allocated:.2f} MB, 峰值: {peak:.2f} MB")

print_memory("初始")

# 导入模型
print("\n📥 加载模型...")
from pylate import models

model = models.ColBERT(model_name_or_path="lightonai/GTE-ModernColBERT-v1")
print_memory("模型加载")

# 创建测试数据
print("\n📝 创建测试数据...")
batch_size = 32
query_texts = ["This is a test query"] * batch_size
pos_texts = ["This is a positive document"] * batch_size
neg_texts = ["This is a negative document"] * batch_size

query_tokens = model.tokenize(query_texts, is_query=True, pad=True)
pos_tokens = model.tokenize(pos_texts, is_query=False, pad=True)
neg_tokens = model.tokenize(neg_texts, is_query=False, pad=True)

query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
pos_tokens = {k: v.to(device) for k, v in pos_tokens.items()}
neg_tokens = {k: v.to(device) for k, v in neg_tokens.items()}

sentence_features = [query_tokens, pos_tokens, neg_tokens]
print_memory("数据创建")

# Step 1: 获取 embeddings
print("\n🚀 Step 1: 获取 embeddings...")
embeddings = [
    F.normalize(model(sf)["token_embeddings"], p=2, dim=-1)
    for sf in sentence_features
]
print(f"   Query embeddings shape: {embeddings[0].shape}")
print(f"   Pos embeddings shape: {embeddings[1].shape}")
print(f"   Neg embeddings shape: {embeddings[2].shape}")
print_memory("获取 embeddings")

# Step 2: 计算分数
print("\n🚀 Step 2: 计算 colbert_scores...")
from pylate.scores import colbert_scores

scores_pos = colbert_scores(embeddings[0], embeddings[1])
print(f"   Pos scores shape: {scores_pos.shape}")
print_memory("计算 pos scores")

scores_neg = colbert_scores(embeddings[0], embeddings[2])
print(f"   Neg scores shape: {scores_neg.shape}")
print_memory("计算 neg scores")

# Step 3: 拼接分数
print("\n🚀 Step 3: 拼接分数...")
scores = torch.cat([scores_pos, scores_neg], dim=1)
print(f"   Combined scores shape: {scores.shape}")
print_memory("拼接分数")

# Step 4: 计算损失
print("\n🚀 Step 4: 计算 CrossEntropy 损失...")
batch_size = embeddings[0].size(0)
labels = torch.arange(0, batch_size, device=embeddings[0].device)
loss = F.cross_entropy(scores / 0.05, labels)
print(f"   Loss: {loss.item():.6f}")
print_memory("计算损失")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
