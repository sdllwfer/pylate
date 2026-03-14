#!/usr/bin/env python
"""使用 tokenizer 分析数据集的 token 长度"""

import json
import os
import sys
sys.path.insert(0, '/home/luwa/Documents/pylate')

from transformers import AutoTokenizer
import statistics

def analyze_dataset_tokens(file_path, model_name="lightonai/ColBERT-Zero"):
    """使用 tokenizer 分析数据集的 token 长度"""
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"📊 分析数据集: {file_path}")
    print(f"🤖 使用 tokenizer: {model_name}")
    print("=" * 60)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"   Tokenizer vocab size: {len(tokenizer)}")
    print(f"   Tokenizer model_max_length: {tokenizer.model_max_length}")
    
    # 统计数据
    query_tokens = []
    instruction_tokens = []
    doc_tokens = []
    combined_tokens = []  # query + instruction
    
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            try:
                item = json.loads(line.strip())
                
                # Query token 数量
                query = item.get('query', '')
                query_tok = tokenizer.encode(query, add_special_tokens=False)
                query_tokens.append(len(query_tok))
                
                # Instruction token 数量
                instruction = item.get('instruction', '')
                if instruction:
                    inst_tok = tokenizer.encode(instruction, add_special_tokens=False)
                    instruction_tokens.append(len(inst_tok))
                    
                    # Query + Instruction 组合
                    combined = f"{query} {instruction}"
                    combined_tok = tokenizer.encode(combined, add_special_tokens=False)
                    combined_tokens.append(len(combined_tok))
                
                # 文档 token 数量
                positive = item.get('positive', item.get('pos', ''))
                if isinstance(positive, list):
                    positive = positive[0] if positive else ''
                doc_tok = tokenizer.encode(positive, add_special_tokens=False)
                doc_tokens.append(len(doc_tok))
                
            except json.JSONDecodeError:
                print(f"   ⚠️  JSON解析错误，跳过第 {line_count} 行")
                continue
    
    print(f"\n📈 数据统计:")
    print(f"   总样本数: {line_count}")
    print(f"   有效样本数: {len(query_tokens)}")
    
    # Query token 统计
    print(f"\n🔍 Query Token 统计:")
    print(f"   平均值: {statistics.mean(query_tokens):.2f}")
    print(f"   中位数: {statistics.median(query_tokens):.2f}")
    print(f"   最大值: {max(query_tokens)}")
    print(f"   最小值: {min(query_tokens)}")
    print(f"   标准差: {statistics.stdev(query_tokens):.2f}")
    
    # Query token 分布
    query_ranges = [
        (0, 10), (10, 20), (20, 32), (32, 50), 
        (50, 100), (100, 200), (200, float('inf'))
    ]
    print(f"\n   Query Token 分布:")
    for min_len, max_len in query_ranges:
        count = sum(1 for l in query_tokens if min_len <= l < max_len)
        percentage = count / len(query_tokens) * 100
        if max_len == float('inf'):
            print(f"      ≥{min_len} tokens: {count} ({percentage:.1f}%)")
        else:
            print(f"      {min_len}-{max_len} tokens: {count} ({percentage:.1f}%)")
    
    # Instruction token 统计
    if instruction_tokens:
        print(f"\n📝 Instruction Token 统计:")
        print(f"   平均值: {statistics.mean(instruction_tokens):.2f}")
        print(f"   中位数: {statistics.median(instruction_tokens):.2f}")
        print(f"   最大值: {max(instruction_tokens)}")
        print(f"   最小值: {min(instruction_tokens)}")
        
        # Instruction token 分布
        inst_ranges = [
            (0, 50), (50, 100), (100, 200), (200, 300), 
            (300, 500), (500, float('inf'))
        ]
        print(f"\n   Instruction Token 分布:")
        for min_len, max_len in inst_ranges:
            count = sum(1 for l in instruction_tokens if min_len <= l < max_len)
            percentage = count / len(instruction_tokens) * 100
            if max_len == float('inf'):
                print(f"      ≥{min_len} tokens: {count} ({percentage:.1f}%)")
            else:
                print(f"      {min_len}-{max_len} tokens: {count} ({percentage:.1f}%)")
    
    # Query + Instruction 组合统计
    if combined_tokens:
        print(f"\n🔗 Query + Instruction Token 统计:")
        print(f"   平均值: {statistics.mean(combined_tokens):.2f}")
        print(f"   中位数: {statistics.median(combined_tokens):.2f}")
        print(f"   最大值: {max(combined_tokens)}")
        print(f"   最小值: {min(combined_tokens)}")
        
        # 超过 512 的样本数
        over_512 = sum(1 for l in combined_tokens if l > 512)
        print(f"\n   ⚠️  超过 512 tokens 的样本: {over_512} ({over_512/len(combined_tokens)*100:.1f}%)")
    
    # 文档 token 统计
    print(f"\n📄 文档 Token 统计:")
    print(f"   平均值: {statistics.mean(doc_tokens):.2f}")
    print(f"   中位数: {statistics.median(doc_tokens):.2f}")
    print(f"   最大值: {max(doc_tokens)}")
    print(f"   最小值: {min(doc_tokens)}")
    print(f"   标准差: {statistics.stdev(doc_tokens):.2f}")
    
    # 超过 2048 的样本数
    over_2048 = sum(1 for l in doc_tokens if l > 2048)
    print(f"\n   ⚠️  超过 2048 tokens 的样本: {over_2048} ({over_2048/len(doc_tokens)*100:.1f}%)")
    
    # 文档 token 分布
    doc_ranges = [
        (0, 128), (128, 256), (256, 512), (512, 1000),
        (1000, 1500), (1500, 2048), (2048, float('inf'))
    ]
    print(f"\n   文档 Token 分布:")
    for min_len, max_len in doc_ranges:
        count = sum(1 for l in doc_tokens if min_len <= l < max_len)
        percentage = count / len(doc_tokens) * 100
        if max_len == float('inf'):
            print(f"      ≥{min_len} tokens: {count} ({percentage:.1f}%)")
        else:
            print(f"      {min_len}-{max_len} tokens: {count} ({percentage:.1f}%)")
    
    # 显示一些示例
    print(f"\n💡 示例数据:")
    print(f"   最长 Query ({max(query_tokens)} tokens):")
    longest_query_idx = query_tokens.index(max(query_tokens))
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == longest_query_idx:
                item = json.loads(line.strip())
                query = item.get('query', '')
                print(f"      {query[:150]}...")
                break
    
    print(f"\n   最长文档 ({max(doc_tokens)} tokens):")
    longest_doc_idx = doc_tokens.index(max(doc_tokens))
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == longest_doc_idx:
                item = json.loads(line.strip())
                positive = item.get('positive', item.get('pos', ''))
                if isinstance(positive, list):
                    positive = positive[0] if positive else ''
                print(f"      {positive[:150]}...")
                break
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")
    
    # 建议
    print(f"\n💡 配置建议:")
    print(f"   - Query + Instruction 最大长度: 建议 ≥ {max(combined_tokens) if combined_tokens else max(query_tokens)} tokens")
    print(f"   - 文档最大长度: 建议 ≥ {max(doc_tokens)} tokens")

if __name__ == "__main__":
    dataset_path = "/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/train_data_igp.jsonl"
    analyze_dataset_tokens(dataset_path)
