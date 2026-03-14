#!/usr/bin/env python
"""分析数据集的长度和文档长度"""

import json
import os
from collections import defaultdict
import statistics

def analyze_dataset(file_path):
    """分析数据集的长度分布"""
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print(f"📊 分析数据集: {file_path}")
    print("=" * 60)
    
    # 统计数据
    query_lengths = []
    doc_lengths = []
    instruction_lengths = []
    
    query_with_instruction = 0
    query_without_instruction = 0
    
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            try:
                item = json.loads(line.strip())
                
                # Query 长度（按字符数）
                query = item.get('query', '')
                query_len = len(query)
                query_lengths.append(query_len)
                
                # Instruction 长度
                instruction = item.get('instruction', '')
                if instruction:
                    instruction_len = len(instruction)
                    instruction_lengths.append(instruction_len)
                    query_with_instruction += 1
                else:
                    query_without_instruction += 1
                
                # 文档长度
                positive = item.get('positive', item.get('pos', ''))
                if isinstance(positive, list):
                    positive = positive[0] if positive else ''
                doc_len = len(positive)
                doc_lengths.append(doc_len)
                
            except json.JSONDecodeError:
                print(f"   ⚠️  JSON解析错误，跳过第 {line_count} 行")
                continue
    
    print(f"\n📈 数据统计:")
    print(f"   总样本数: {line_count}")
    print(f"   有效样本数: {len(query_lengths)}")
    
    # Query 长度统计
    print(f"\n🔍 Query 长度统计（字符数）:")
    print(f"   平均值: {statistics.mean(query_lengths):.2f}")
    print(f"   中位数: {statistics.median(query_lengths):.2f}")
    print(f"   最大值: {max(query_lengths)}")
    print(f"   最小值: {min(query_lengths)}")
    print(f"   标准差: {statistics.stdev(query_lengths):.2f}")
    
    # Query 长度分布
    query_ranges = [
        (0, 50), (50, 100), (100, 200), (200, 500), 
        (500, 1000), (1000, 2000), (2000, float('inf'))
    ]
    print(f"\n   Query 长度分布:")
    for min_len, max_len in query_ranges:
        count = sum(1 for l in query_lengths if min_len <= l < max_len)
        percentage = count / len(query_lengths) * 100
        if max_len == float('inf'):
            print(f"      ≥{min_len} chars: {count} ({percentage:.1f}%)")
        else:
            print(f"      {min_len}-{max_len} chars: {count} ({percentage:.1f}%)")
    
    # Instruction 统计
    print(f"\n📝 Instruction 统计:")
    print(f"   有 instruction 的样本: {query_with_instruction} ({query_with_instruction/len(query_lengths)*100:.1f}%)")
    print(f"   无 instruction 的样本: {query_without_instruction} ({query_without_instruction/len(query_lengths)*100:.1f}%)")
    
    if instruction_lengths:
        print(f"\n   Instruction 长度统计（字符数）:")
        print(f"      平均值: {statistics.mean(instruction_lengths):.2f}")
        print(f"      中位数: {statistics.median(instruction_lengths):.2f}")
        print(f"      最大值: {max(instruction_lengths)}")
        print(f"      最小值: {min(instruction_lengths)}")
    
    # 文档长度统计
    print(f"\n📄 文档长度统计（字符数）:")
    print(f"   平均值: {statistics.mean(doc_lengths):.2f}")
    print(f"   中位数: {statistics.median(doc_lengths):.2f}")
    print(f"   最大值: {max(doc_lengths)}")
    print(f"   最小值: {min(doc_lengths)}")
    print(f"   标准差: {statistics.stdev(doc_lengths):.2f}")
    
    # 文档长度分布
    doc_ranges = [
        (0, 500), (500, 1000), (1000, 2000), (2000, 5000),
        (5000, 10000), (10000, 20000), (20000, float('inf'))
    ]
    print(f"\n   文档长度分布:")
    for min_len, max_len in doc_ranges:
        count = sum(1 for l in doc_lengths if min_len <= l < max_len)
        percentage = count / len(doc_lengths) * 100
        if max_len == float('inf'):
            print(f"      ≥{min_len} chars: {count} ({percentage:.1f}%)")
        else:
            print(f"      {min_len}-{max_len} chars: {count} ({percentage:.1f}%)")
    
    # 显示一些示例
    print(f"\n💡 示例数据:")
    print(f"   最长 Query ({max(query_lengths)} chars):")
    longest_query_idx = query_lengths.index(max(query_lengths))
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == longest_query_idx:
                item = json.loads(line.strip())
                query = item.get('query', '')
                print(f"      {query[:200]}...")
                break
    
    print(f"\n   最长文档 ({max(doc_lengths)} chars):")
    longest_doc_idx = doc_lengths.index(max(doc_lengths))
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == longest_doc_idx:
                item = json.loads(line.strip())
                positive = item.get('positive', item.get('pos', ''))
                if isinstance(positive, list):
                    positive = positive[0] if positive else ''
                print(f"      {positive[:200]}...")
                break
    
    print("\n" + "=" * 60)
    print("✅ 分析完成!")

if __name__ == "__main__":
    dataset_path = "/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/train_data_igp.jsonl"
    analyze_dataset(dataset_path)
