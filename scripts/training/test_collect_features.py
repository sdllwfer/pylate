"""
测试 collect_features 的返回格式
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from data_utils import IGPColBERTCollator
from pylate import models

def test_collect_features():
    """测试 collect_features 的返回格式"""
    print("=" * 60)
    print("测试 collect_features 返回格式")
    print("=" * 60)
    
    # 创建 tokenizer
    model_name = "answerdotai/ModernBERT-base"
    print(f"\n加载模型: {model_name}")
    base_model = models.ColBERT(model_name_or_path=model_name)
    tokenizer = base_model.tokenizer
    
    # 创建 collator
    collator = IGPColBERTCollator(tokenizer=tokenizer, max_query_length=32)
    
    # 创建测试数据
    features = [
        {
            'query': 'What is machine learning?',
            'instruction': 'Focus on definitions',
            'positive': 'Machine learning is a subset of AI.',
            'negative': 'The weather is nice today.',
        },
        {
            'query': 'How does neural network work?',
            'instruction': 'Explain the mechanism',
            'positive': 'Neural networks consist of layers of neurons.',
            'negative': 'I like eating pizza.',
        }
    ]
    
    # 使用 collator 处理数据
    print("\n使用 collator 处理数据...")
    batch = collator(features)
    
    print("\nCollator 返回的 batch 键:")
    for key in sorted(batch.keys()):
        print(f"  {key}: {batch[key].shape}")
    
    # 模拟 collect_features 的行为
    print("\n模拟 collect_features 的行为...")
    
    # 根据 SentenceTransformerTrainer.collect_features 的实现
    # 它会查找以 _input_ids, _sentence_embedding, _pixel_values 结尾的键
    features_list = []
    for column in batch:
        if column.endswith("_input_ids"):
            prefix = column[: -len("input_ids")]
        elif column.endswith("_sentence_embedding"):
            prefix = column[: -len("sentence_embedding")]
        elif column.endswith("_pixel_values"):
            prefix = column[: -len("pixel_values")]
        else:
            continue
        
        feature_dict = {key[len(prefix):]: value for key, value in batch.items() if key.startswith(prefix)}
        features_list.append(feature_dict)
        print(f"\n  Prefix '{prefix}' -> 特征: {list(feature_dict.keys())}")
    
    print(f"\n最终 features 列表长度: {len(features_list)}")
    for i, feat in enumerate(features_list):
        print(f"  features[{i}] 键: {list(feat.keys())}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_collect_features()
