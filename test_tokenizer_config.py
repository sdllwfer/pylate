#!/usr/bin/env python
"""测试 tokenizer 配置"""

from transformers import AutoTokenizer

# 加载 tokenizer
model_path = "lightonai/ColBERT-Zero"
print(f"加载 tokenizer: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"\nTokenizer 配置:")
print(f"  model_max_length: {tokenizer.model_max_length}")
print(f"  max_len_single_sentence: {tokenizer.max_len_single_sentence if hasattr(tokenizer, 'max_len_single_sentence') else 'N/A'}")
print(f"  max_len_sentences_pair: {tokenizer.max_len_sentences_pair if hasattr(tokenizer, 'max_len_sentences_pair') else 'N/A'}")

# 测试 encode
test_input = "This is a test query with some instruction. Please follow this instruction carefully."
encoded = tokenizer.encode(test_input, add_special_tokens=True)
print(f"\n编码结果:")
print(f"  输入长度: {len(test_input.split())} words")
print(f"  token 数量: {len(encoded)}")

# 使用不同的 max_length 测试
for max_len in [38, 512, 1024]:
    encoded = tokenizer.encode(test_input, add_special_tokens=True, max_length=max_len, truncation=True)
    print(f"  max_length={max_len}: {len(encoded)} tokens")

print("\n✅ 测试完成!")
