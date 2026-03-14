import torch
import gc

def reset_gpu():
    torch.cuda.empty_cache()
    gc.collect()

# 使用 GPU 0（不限制 CUDA_VISIBLE_DEVICES）
device = torch.device("cuda:0")
reset_gpu()

print("=" * 60)
print("测试 train_followir.py 显存占用")
print("=" * 60)

from pylate import models

reset_gpu()
print("\n加载 ColBERT 模型...")
model1 = models.ColBERT(model_name_or_path="lightonai/ColBERT-Zero", device="cuda:0")

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

print("前向传播...")
q_emb = model1(q_tokens)["token_embeddings"]
p_emb = model1(p_tokens)["token_embeddings"]
n_emb = model1(n_tokens)["token_embeddings"]

from pylate.scores import colbert_scores
scores_pos = colbert_scores(q_emb, p_emb)
scores_neg = colbert_scores(q_emb, n_emb)

mem1 = torch.cuda.memory_allocated(device) / 1024**3
print(f"\ntrain_followir 显存: {mem1:.2f}GB")
