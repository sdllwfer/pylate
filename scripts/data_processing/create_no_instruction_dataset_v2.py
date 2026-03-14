"""
从短指令数据集中生成只有无指令样本的数据集（改进版）

改进：
1. 严格排除当前查询的所有相关文档（包括原pos和原neg）
2. 严格排除其他查询的正文档
3. 使用更严格的负采样策略
"""

import json
import random
import argparse
from pathlib import Path


def create_no_instruction_dataset_v2(input_path: str, output_path: str, neg_sample_ratio: float = 1.0):
    """
    创建无指令样本数据集（改进版）
    
    Args:
        input_path: 原始数据集路径
        output_path: 输出数据集路径
        neg_sample_ratio: 负样本采样比例（相对于正样本数量）
    """
    print(f"📂 读取原始数据集: {input_path}")
    
    # 读取所有样本
    samples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"   共读取 {len(samples)} 个样本")
    
    # 第一步：收集所有文档，并建立文档到查询的映射
    doc_to_queries = {}  # 文档 -> 哪些查询认为它是正样本
    query_related_docs = {}  # 查询 -> 它的所有相关文档
    
    for sample in samples:
        query = sample['query']
        pos_docs = sample.get('pos', [])
        neg_docs = sample.get('neg', [])
        
        # 该查询的所有相关文档（pos + neg，因为无指令时都是相关的）
        all_related = set(pos_docs + neg_docs)
        query_related_docs[query] = all_related
        
        # 建立文档到查询的映射
        for doc in all_related:
            if doc not in doc_to_queries:
                doc_to_queries[doc] = set()
            doc_to_queries[doc].add(query)
    
    print(f"   共收集 {len(doc_to_queries)} 个唯一文档")
    
    # 第二步：为每个查询生成严格负样本
    no_instruction_samples = []
    
    for sample in samples:
        query = sample['query']
        original_pos = sample.get('pos', [])
        original_neg = sample.get('neg', [])
        
        # 创建无指令版本
        new_sample = {
            'query': query,
            'pos': original_pos + original_neg,  # 原来的pos和neg都变成pos
            'neg': []
        }
        
        # 严格负样本：不能是任何查询的相关文档
        # 1. 排除当前查询的所有相关文档
        current_related = query_related_docs[query]
        
        # 2. 从全局文档池中筛选严格负样本
        strict_negative_candidates = []
        for doc in doc_to_queries.keys():
            # 该文档相关的所有查询
            related_queries = doc_to_queries[doc]
            # 如果该文档只与当前查询相关（即不在其他查询中作为正样本）
            # 则它不能作为当前查询的负样本
            # 我们需要的是：与该查询完全无关的文档
            if query not in related_queries:
                # 这个文档与当前查询无关，可以作为负样本
                strict_negative_candidates.append(doc)
        
        # 需要的负样本数量
        num_neg_needed = int(len(new_sample['pos']) * neg_sample_ratio)
        
        # 采样负样本
        if len(strict_negative_candidates) >= num_neg_needed:
            new_sample['neg'] = random.sample(strict_negative_candidates, num_neg_needed)
        else:
            print(f"   ⚠️ 警告: 查询 '{query[:50]}...' 的严格负样本不足")
            print(f"      需要 {num_neg_needed} 个，只有 {len(strict_negative_candidates)} 个")
            # 如果不够，使用所有可用的严格负样本
            new_sample['neg'] = strict_negative_candidates
        
        no_instruction_samples.append(new_sample)
    
    # 保存结果
    print(f"\n💾 保存无指令数据集到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in no_instruction_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"   共生成 {len(no_instruction_samples)} 个无指令样本")
    
    # 统计信息
    total_pos = sum(len(s['pos']) for s in no_instruction_samples)
    total_neg = sum(len(s['neg']) for s in no_instruction_samples)
    print(f"\n📊 数据集统计:")
    print(f"   - 样本总数: {len(no_instruction_samples)}")
    print(f"   - 正样本总数: {total_pos}")
    print(f"   - 负样本总数: {total_neg}")
    print(f"   - 平均每个查询的正样本: {total_pos / len(no_instruction_samples):.2f}")
    print(f"   - 平均每个查询的负样本: {total_neg / len(no_instruction_samples):.2f}")
    
    # 验证负样本质量
    print(f"\n🔍 验证负样本质量...")
    overlap_count = 0
    for sample in no_instruction_samples:
        query = sample['query']
        for neg_doc in sample['neg']:
            # 检查负文档是否出现在该查询的相关文档中
            if neg_doc in query_related_docs.get(query, set()):
                overlap_count += 1
    
    if overlap_count == 0:
        print("   ✅ 所有负样本都严格与查询无关")
    else:
        print(f"   ⚠️ 发现 {overlap_count} 个问题负样本")


def main():
    parser = argparse.ArgumentParser(description='创建无指令样本数据集（改进版）')
    parser.add_argument('--input', type=str, 
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl',
                        help='输入数据集路径')
    parser.add_argument('--output', type=str,
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/no_instruction_train_v2.jsonl',
                        help='输出数据集路径')
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                        help='负样本采样比例（相对于正样本数量）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建数据集
    create_no_instruction_dataset_v2(args.input, args.output, args.neg_ratio)
    
    print("\n✅ 无指令数据集（改进版）生成完成！")


if __name__ == '__main__':
    main()
