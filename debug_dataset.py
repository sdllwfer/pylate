#!/usr/bin/env python3
"""
调试数据集列顺序问题
"""
import sys
sys.path.insert(0, '/home/luwa/Documents/pylate')

from datasets import Dataset

# 模拟数据加载
data_list = []
with open('/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/colbert_train_final.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:  # 只加载前10条
            break
        item = eval(line.strip())
        query = item.get('query', '')
        positives = item.get('pos', [])
        negatives = item.get('neg', [])
        
        if positives and negatives:
            for pos_doc in positives[:1]:  # 只取第一个正样本
                for neg_doc in negatives[:1]:  # 只取第一个负样本
                    data_list.append({
                        'anchor': query,
                        'positive': pos_doc,
                        'negative': neg_doc,
                    })

print("=" * 70)
print("🔍 调试数据集列顺序")
print("=" * 70)

# 创建数据集
dataset = Dataset.from_list(data_list)

print(f"\n[1] 原始数据集列: {dataset.column_names}")
print(f"   列数: {len(dataset.column_names)}")

# 检查列顺序
for i, col in enumerate(dataset.column_names):
    print(f"   索引 {i}: {col}")

# 应用 select_columns
dataset_fixed = dataset.select_columns(['anchor', 'positive', 'negative'])
print(f"\n[2] 修复后数据集列: {dataset_fixed.column_names}")

# 检查数据内容
print(f"\n[3] 第一条数据:")
print(f"   anchor: {dataset_fixed[0]['anchor'][:50]}...")
print(f"   positive: {dataset_fixed[0]['positive'][:50]}...")
print(f"   negative: {dataset_fixed[0]['negative'][:50]}...")

print("\n" + "=" * 70)
print("✅ 调试完成")
print("=" * 70)
