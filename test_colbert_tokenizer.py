#!/usr/bin/env python
"""测试 ColBERT tokenizer"""

import sys
sys.path.insert(0, '/home/luwa/Documents/pylate')

from pylate import models

# 加载模型
model_path = "lightonai/ColBERT-Zero"
print(f"加载模型: {model_path}")

print("\n=== 加载前 ===")
from transformers import AutoTokenizer
tokenizer_before = AutoTokenizer.from_pretrained(model_path)
print(f"  tokenizer.model_max_length: {tokenizer_before.model_max_length}")

model = models.ColBERT(
    model_name_or_path=model_path,
    device='cpu',
    query_length=512,
    document_length=2048,
)

print(f"\n=== 加载后 ===")
print(f"  model.query_length: {model.query_length}")
print(f"  model.document_length: {model.document_length}")
print(f"  model.tokenizer.model_max_length: {model.tokenizer.model_max_length}")

# 检查 _first_module
print(f"\n=== _first_module ===")
transformer = model._first_module()
print(f"  transformer.max_seq_length: {transformer.max_seq_length}")
print(f"  transformer.tokenizer.model_max_length: {transformer.tokenizer.model_max_length}")

# 设置 max_seq_length
max_seq_length = max(512, 2048) + 10
transformer.max_seq_length = max_seq_length
print(f"\n=== 设置 max_seq_length = {max_seq_length} 后 ===")
print(f"  transformer.max_seq_length: {transformer.max_seq_length}")
print(f"  transformer.tokenizer.model_max_length: {transformer.tokenizer.model_max_length}")
print(f"  model.tokenizer.model_max_length: {model.tokenizer.model_max_length}")

print("\n✅ 测试完成!")
