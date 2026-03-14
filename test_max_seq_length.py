#!/usr/bin/env python
"""测试 max_seq_length 是否正确设置"""

import sys
sys.path.insert(0, '/home/luwa/Documents/pylate')

from pylate import models

# 加载模型
model_path = "lightonai/ColBERT-Zero"
print(f"加载模型: {model_path}")

model = models.ColBERT(
    model_name_or_path=model_path,
    device='cpu',
    query_length=512,
    document_length=2048,
)

print(f"\n加载后:")
print(f"  query_length: {model.query_length}")
print(f"  document_length: {model.document_length}")
print(f"  _first_module().max_seq_length: {model._first_module().max_seq_length}")

# 设置 max_seq_length
max_seq_length = max(512, 2048) + 10
model._first_module().max_seq_length = max_seq_length
print(f"\n设置后:")
print(f"  _first_module().max_seq_length: {model._first_module().max_seq_length}")

# 测试 tokenize
query = "This is a test query with some instruction. Please follow this instruction carefully."
doc = "This is a test document. " * 50  # 长文档

print(f"\n测试 tokenize:")
print(f"  Query length: {len(query.split())} words")
print(f"  Doc length: {len(doc.split())} words")

# Tokenize query
query_tokens = model.tokenize([query], is_query=True)
print(f"  Query tokens shape: {query_tokens['input_ids'].shape}")

# Tokenize doc
doc_tokens = model.tokenize([doc], is_query=False)
print(f"  Doc tokens shape: {doc_tokens['input_ids'].shape}")

print("\n✅ 测试完成!")
