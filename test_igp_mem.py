import torch
import gc
import sys
sys.path.insert(0, '/home/luwa/Documents/pylate/scripts/training')

def reset_gpu():
    torch.cuda.empty_cache()
    gc.collect()

device = torch.device("cuda:0")
reset_gpu()

print("=" * 60)
print("测试 train_colbert_igp.py 显存占用")
print("=" * 60)

from pylate import models

reset_gpu()
print("\n加载 ColBERT 模型...")
model = models.ColBERT(model_name_or_path="lightonai/ColBERT-Zero", device="cuda:0")

print("加载 IGP 模块...")
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGateV3
hidden_size = model[0].get_word_embedding_dimension()
probe = InstructionProbe(hidden_size=hidden_size, num_heads=8, dropout=0.1).to(device)
adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=128, dropout=0.1, input_dim=hidden_size).to(device)
gate = RatioGateV3(hidden_size=hidden_size, max_ratio=0.2).to(device)

batch_size = 32
test_queries = ["test query " + str(i) for i in range(batch_size)]
test_docs = ["test doc " + str(i) for i in range(batch_size)]

q_tokens = model.tokenize(test_queries, is_query=True)
d_tokens = model.tokenize(test_docs, is_query=False)

q_tokens = {k: v.to(device) for k, v in q_tokens.items()}
d_tokens = {k: v.to(device) for k, v in d_tokens.items()}

print("前向传播...")
q_emb = model(q_tokens)["token_embeddings"]
d_emb = model(d_tokens)["token_embeddings"]

from pylate.scores import colbert_scores
scores = colbert_scores(q_emb, d_emb)

mem = torch.cuda.memory_allocated(device) / 1024**3
print(f"\ntrain_colbert_igp 显存: {mem:.2f}GB")
