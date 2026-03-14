import torch
import gc
import sys
sys.path.insert(0, '/home/luwa/Documents/pylate/scripts/training')

def reset_gpu():
    torch.cuda.empty_cache()
    gc.collect()

device = torch.device("cuda:2")
reset_gpu()

print("=" * 60)
print("1. 测试 train_followir.py 前向传播")
print("=" * 60)

from pylate import models

reset_gpu()
model1 = models.ColBERT(model_name_or_path="lightonai/ColBERT-Zero", device="cuda:2")

batch_size = 32
queries = ["test query " + str(i) for i in range(batch_size)]
positives = ["test positive " + str(i) for i in range(batch_size)]
negatives = ["test negative " + str(i) for i in range(batch_size)]

q_tokens = model1.tokenize(queries, is_query=True)
p_tokens = model1.tokenize(positives, is_query=False)
n_tokens = model1.tokenize(negatives, is_query=False)

q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
p_tokens = {k: v.to(device) for k, v in p_tokens.items()}
n_tokens = {k: v.to(device) for k, v in n_tokens.items()}

q_emb = model1(q_tokens)["token_embeddings"]
p_emb = model1(p_tokens)["token_embeddings"]
n_emb = model1(n_tokens)["token_embeddings"]

from pylate.scores import colbert_scores
scores_pos = colbert_scores(q_emb, p_emb)
scores_neg = colbert_scores(q_emb, n_emb)

mem1 = torch.cuda.memory_allocated(device) / 1024**3
print(f"train_followir 前向: {mem1:.2f}GB")
print(f"train_followir 分数 shape: pos={scores_pos.shape}, neg={scores_neg.shape}")

del model1, q_emb, p_emb, n_emb, scores_pos, scores_neg
reset_gpu()

print("\n" + "=" * 60)
print("2. 测试 train_colbert_igp.py 方式 (简化版)")
print("=" * 60)

reset_gpu()
model2 = models.ColBERT(model_name_or_path="lightonai/ColBERT-Zero", device="cuda:2")

from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGateV3
hidden_size = model2[0].get_word_embedding_dimension()
probe = InstructionProbe(hidden_size=hidden_size, num_heads=8, dropout=0.1).to(device)
adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=128, dropout=0.1, input_dim=hidden_size).to(device)
gate = RatioGateV3(hidden_size=hidden_size, max_ratio=0.2).to(device)

print(f"IGP 模块参数量: Probe={sum(p.numel() for p in probe.parameters()):,}, Adapter={sum(p.numel() for p in adapter.parameters()):,}, Gate={sum(p.numel() for p in gate.parameters()):,}")

# 使用与 train_followir 相同的方式处理数据
test_queries = ["test query " + str(i) for i in range(batch_size)]
test_docs = ["test doc " + str(i) for i in range(batch_size)]

q_tokens2 = model2.tokenize(test_queries, is_query=True)
d_tokens = model2.tokenize(test_docs, is_query=False)

q_tokens2 = {k: v.to(device) for k, v in q_tokens2.items()}
d_tokens = {k: v.to(device) for k, v in d_tokens.items()}

q_emb2 = model2(q_tokens2)["token_embeddings"]
d_emb2 = model2(d_tokens)["token_embeddings"]

scores2 = colbert_scores(q_emb2, d_emb2)

mem2 = torch.cuda.memory_allocated(device) / 1024**3
print(f"train_colbert_igp 前向: {mem2:.2f}GB")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print(f"train_followir: {mem1:.2f}GB")
print(f"train_colbert_igp (简化): {mem2:.2f}GB")
print(f"差异: {mem1 - mem2:.2f}GB")
