"""
测试 query 分割逻辑
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from data_utils import IGPColBERTCollator
from pylate import models

def test_query_split():
    """测试 query 分割逻辑"""
    print("=" * 60)
    print("测试 Query 分割逻辑")
    print("=" * 60)
    
    # 加载模型和 tokenizer
    model_name = "answerdotai/ModernBERT-base"
    print(f"\n加载模型: {model_name}")
    base_model = models.ColBERT(model_name_or_path=model_name)
    tokenizer = base_model.tokenizer
    
    # 创建 collator
    collator = IGPColBERTCollator(tokenizer=tokenizer, max_query_length=128)
    
    # 测试数据
    test_queries = [
        "A relevant document should include the countries or individuals who oppose the use of the euro and the reason(s) for their opposition to its use. Identify documents that discuss opposition to the introduction of the euro, the European currency.",
        "Given the query about the benefits of exercise, the relevant document should discuss the positive effects of physical activity on health. Find documents that explain how exercise improves cardiovascular health.",
        "This is a simple query without clear instruction markers. It just asks about machine learning.",
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}:")
        print(f"{'='*60}")
        print(f"原始 Query ({len(query.split())} 词):")
        print(f"  {query[:100]}...")
        
        # 创建测试数据
        features = [{
            'query': query,
            'pos': ['This is a positive document.'],
            'neg': ['This is a negative document.'],
        }]
        
        # 使用 collator 处理
        batch = collator(features)
        
        # 解码查看结果
        input_ids = batch['sentence_0_input_ids'][0]
        token_labels = batch['sentence_0_token_labels'][0]
        
        # 找到实际的 token（非 padding）
        mask = input_ids != tokenizer.pad_token_id
        actual_tokens = input_ids[mask]
        actual_labels = token_labels[mask]
        
        decoded = tokenizer.decode(actual_tokens, skip_special_tokens=False)
        
        # 统计 query 和 instruction 的 token 数量
        query_tokens = (actual_labels == 0).sum().item()
        instruction_tokens = (actual_labels == 1).sum().item()
        
        print(f"\n分割结果:")
        print(f"  Query tokens (label=0): {query_tokens}")
        print(f"  Instruction tokens (label=1): {instruction_tokens}")
        print(f"  总 tokens: {query_tokens + instruction_tokens}")
        print(f"\n解码结果:")
        print(f"  {decoded[:150]}...")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_query_split()
