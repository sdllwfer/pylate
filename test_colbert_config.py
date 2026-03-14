#!/usr/bin/env python
"""测试 ColBERT 模型配置"""

import sys
sys.path.insert(0, '/home/luwa/Documents/pylate')

from transformers import AutoConfig, AutoTokenizer

# 加载模型配置
model_path = "lightonai/ColBERT-Zero"
print(f"加载模型配置: {model_path}")

config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f"\n模型配置:")
print(f"  max_position_embeddings: {config.max_position_embeddings}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_hidden_layers: {config.num_hidden_layers}")

print(f"\nTokenizer 配置:")
print(f"  model_max_length: {tokenizer.model_max_length}")
print(f"  max_len_single_sentence: {tokenizer.max_len_single_sentence if hasattr(tokenizer, 'max_len_single_sentence') else 'N/A'}")

# 检查 sentence_bert_config.json
import json
import os
from huggingface_hub import hf_hub_download

try:
    config_path = hf_hub_download(repo_id=model_path, filename="sentence_bert_config.json")
    with open(config_path, 'r') as f:
        sb_config = json.load(f)
    print(f"\nsentence_bert_config.json:")
    print(f"  {sb_config}")
except Exception as e:
    print(f"\nsentence_bert_config.json: 未找到 ({e})")

# 检查 config_sentence_transformers.json
try:
    config_path = hf_hub_download(repo_id=model_path, filename="config_sentence_transformers.json")
    with open(config_path, 'r') as f:
        cst_config = json.load(f)
    print(f"\nconfig_sentence_transformers.json:")
    print(f"  {cst_config}")
except Exception as e:
    print(f"\nconfig_sentence_transformers.json: 未找到 ({e})")

print("\n✅ 测试完成!")
