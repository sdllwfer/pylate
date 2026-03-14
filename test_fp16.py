import torch
import gc

def reset_gpu():
    torch.cuda.empty_cache()
    gc.collect()

device = torch.device("cuda:0")
reset_gpu()

print("=" * 60)
print("测试 train_followir.py (fp16=True)")
print("=" * 60)

from pylate import models

reset_gpu()
model = models.ColBERT(model_name_or_path="lightonai/ColBERT-Zero", device="cuda:0")

batch_size = 16
queries = ["test query " + str(i) for i in range(batch_size)]
positives = ["test positive " + str(i) for i in range(batch_size)]
negatives = ["test negative " + str(i) for i in range(batch_size)]

q_tokens = model.tokenize(queries, is_query=True)
p_tokens = model.tokenize(positives, is_query=False)
n_tokens = model.tokenize(negatives, is_query=False)

q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
p_tokens = {k: v.to(device) for k, v in p_tokens.items()}
n_tokens = {k: v.to(device) for k, v in n_tokens.items()}

print("前向传播 (fp32)...")
q_emb = model(q_tokens)["token_embeddings"]
p_emb = model(p_tokens)["token_embeddings"]
n_emb = model(n_tokens)["token_embeddings"]

from pylate.scores import colbert_scores
scores_pos = colbert_scores(q_emb, p_emb)
scores_neg = colbert_scores(q_emb, n_emb)

mem_fp32 = torch.cuda.memory_allocated(device) / 1024**3
print(f"fp32 显存: {mem_fp32:.2f}GB")

del q_emb, p_emb, n_emb, scores_pos, scores_neg
reset_gpu()

print("\n前向传播 (fp16)...")
with torch.cuda.amp.autocast(dtype=torch.float16):
    q_emb = model(q_tokens)["token_embeddings"]
    p_emb = model(p_tokens)["token_embeddings"]
    n_emb = model(n_tokens)["token_embeddings"]
    
    scores_pos = colbert_scores(q_emb, p_emb)
    scores_neg = colbert_scores(q_emb, n_emb)

mem_fp16 = torch.cuda.memory_allocated(device) / 1024**3
print(f"fp16 显存: {mem_fp16:.2f}GB")

print(f"\n差异: fp16 比 fp32 多用 {mem_fp16 - mem_fp32:.2f}GB")
