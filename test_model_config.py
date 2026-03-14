#!/usr/bin/env python
"""测试模型配置"""

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

# 检查模型配置
transformer = model._first_module()
print(f"\n模型配置:")
print(f"  max_seq_length: {transformer.max_seq_length}")
print(f"  auto_model.config.max_position_embeddings: {transformer.auto_model.config.max_position_embeddings}")
print(f"  tokenizer.model_max_length: {transformer.tokenizer.model_max_length}")

# 设置 max_seq_length
max_seq_length = max(512, 2048) + 10
transformer.max_seq_length = max_seq_length
print(f"\n设置后:")
print(f"  max_seq_length: {transformer.max_seq_length}")

# 测试 forward
test_input = "This is a test query with some instruction. Please follow this instruction carefully."
tokens = model.tokenize([test_input], is_query=True)
print(f"\nTokenize 结果:")
print(f"  input_ids shape: {tokens['input_ids'].shape}")

# 直接调用 Transformer.forward
features = {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}
output = transformer(features)
print(f"  output token_embeddings shape: {output['token_embeddings'].shape}")

print("\n✅ 测试完成!")
