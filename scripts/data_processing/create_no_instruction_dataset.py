"""
从短指令数据集中生成只有无指令样本的数据集

处理逻辑：
1. 读取原始数据集
2. 移除 instruction 字段
3. 将原来的 neg 样本合并到 pos 样本中（因为无指令时它们都是相关的）
4. 从其他样本中随机采样作为新的 neg 样本
"""

import json
import random
import argparse
from pathlib import Path


def create_no_instruction_dataset(input_path: str, output_path: str, neg_sample_ratio: float = 1.0):
    """
    创建无指令样本数据集
    
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
    
    # 收集所有文档用于负样本采样
    all_docs = []
    for sample in samples:
        all_docs.extend(sample.get('pos', []))
        all_docs.extend(sample.get('neg', []))
    
    print(f"   共收集 {len(all_docs)} 个文档用于负样本采样")
    
    # 生成无指令样本
    no_instruction_samples = []
    for sample in samples:
        # 创建无指令版本
        new_sample = {
            'query': sample['query'],
            # 移除 instruction 相关字段
            'pos': [],
            'neg': []
        }
        
        # 原来的 pos 和 neg 都变成 pos（因为无指令约束时都是相关的）
        original_pos = sample.get('pos', [])
        original_neg = sample.get('neg', [])
        new_sample['pos'] = original_pos + original_neg
        
        # 从其他样本中随机采样作为 neg
        num_neg_needed = int(len(new_sample['pos']) * neg_sample_ratio)
        
        # 排除当前样本的文档
        other_docs = [d for d in all_docs if d not in original_pos and d not in original_neg]
        
        if len(other_docs) >= num_neg_needed:
            new_sample['neg'] = random.sample(other_docs, num_neg_needed)
        else:
            # 如果不够，允许重复采样
            new_sample['neg'] = random.choices(other_docs, k=num_neg_needed) if other_docs else []
        
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


def main():
    parser = argparse.ArgumentParser(description='创建无指令样本数据集')
    parser.add_argument('--input', type=str, 
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl',
                        help='输入数据集路径')
    parser.add_argument('--output', type=str,
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/no_instruction_train.jsonl',
                        help='输出数据集路径')
    parser.add_argument('--neg_ratio', type=float, default=1.0,
                        help='负样本采样比例（相对于正样本数量）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建数据集
    create_no_instruction_dataset(args.input, args.output, args.neg_ratio)
    
    print("\n✅ 无指令数据集生成完成！")


if __name__ == '__main__':
    main()
