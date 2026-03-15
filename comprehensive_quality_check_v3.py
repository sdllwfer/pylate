#!/usr/bin/env python3
"""
全面质量检查过拟合训练集 v3
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Any

random.seed(42)


def load_data(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def check_basic_stats(data: List[Dict]):
    print("=" * 80)
    print("1. 基本统计信息")
    print("=" * 80)
    
    total = len(data)
    with_inst = [item for item in data if item.get('type') == 'with_instruction']
    without_inst = [item for item in data if item.get('type') == 'without_instruction']
    
    print(f"总样本数: {total}")
    print(f"有指令样本: {len(with_inst)} ({len(with_inst)/total*100:.1f}%)")
    print(f"无指令样本: {len(without_inst)} ({len(without_inst)/total*100:.1f}%)")
    
    by_dataset = defaultdict(lambda: {'total': 0, 'with_inst': 0, 'without_inst': 0})
    for item in data:
        ds = item.get('dataset', 'unknown')
        by_dataset[ds]['total'] += 1
        if item.get('type') == 'with_instruction':
            by_dataset[ds]['with_inst'] += 1
        else:
            by_dataset[ds]['without_inst'] += 1
    
    print("\n按数据集分布:")
    for ds, stats in sorted(by_dataset.items()):
        print(f"  {ds}: 总计 {stats['total']} (有指令: {stats['with_inst']}, 无指令: {stats['without_inst']})")


def check_document_quality(data: List[Dict]):
    print("\n" + "=" * 80)
    print("2. 文档质量检查")
    print("=" * 80)
    
    issues = []
    
    for i, item in enumerate(data):
        pos_docs = item.get('pos', [])
        if not pos_docs:
            issues.append(f"样本 {i}: 没有正样本")
        
        for j, doc in enumerate(pos_docs):
            if not doc or not doc.strip():
                issues.append(f"样本 {i}, 正样本 {j}: 文档为空")
        
        neg_docs = item.get('neg', [])
        if not neg_docs:
            issues.append(f"样本 {i}: 没有负样本")
        elif len(neg_docs) < 3:
            issues.append(f"样本 {i}: 负样本数量不足 ({len(neg_docs)} < 3)")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:10]:
            print(f"  - {issue}")
    else:
        print("✅ 所有文档质量检查通过")
    
    pos_lengths = []
    neg_lengths = []
    
    for item in data:
        for doc in item.get('pos', []):
            if doc:
                pos_lengths.append(len(doc))
        for doc in item.get('neg', []):
            if doc:
                neg_lengths.append(len(doc))
    
    if pos_lengths:
        print(f"\n正样本长度统计:")
        print(f"  平均: {sum(pos_lengths)/len(pos_lengths):.0f} 字符")
        print(f"  中位数: {sorted(pos_lengths)[len(pos_lengths)//2]} 字符")
        print(f"  最小: {min(pos_lengths)} 字符, 最大: {max(pos_lengths)} 字符")
    
    if neg_lengths:
        print(f"\n负样本长度统计:")
        print(f"  平均: {sum(neg_lengths)/len(neg_lengths):.0f} 字符")
        print(f"  中位数: {sorted(neg_lengths)[len(neg_lengths)//2]} 字符")
        print(f"  最小: {min(neg_lengths)} 字符, 最大: {max(neg_lengths)} 字符")


def check_data_balance(data: List[Dict]):
    print("\n" + "=" * 80)
    print("3. 数据平衡性检查")
    print("=" * 80)
    
    pos_counts = []
    neg_counts = []
    
    for item in data:
        pos_counts.append(len(item.get('pos', [])))
        neg_counts.append(len(item.get('neg', [])))
    
    print("正样本数量分布:")
    pos_dist = defaultdict(int)
    for c in pos_counts:
        pos_dist[c] += 1
    for c in sorted(pos_dist.keys()):
        print(f"  {c} 个: {pos_dist[c]} 样本 ({pos_dist[c]/len(data)*100:.1f}%)")
    
    print("\n负样本数量分布:")
    neg_dist = defaultdict(int)
    for c in neg_counts:
        neg_dist[c] += 1
    for c in sorted(neg_dist.keys()):
        print(f"  {c} 个: {neg_dist[c]} 样本 ({neg_dist[c]/len(data)*100:.1f}%)")
    
    ratios = [n/p for p, n in zip(pos_counts, neg_counts) if p > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    print(f"\n正负样本比例:")
    print(f"  平均比例: 1:{avg_ratio:.1f}")
    print(f"  推荐比例: 1:3 到 1:10")
    
    if 3 <= avg_ratio <= 10:
        print(f"  ✅ 正负样本比例合理")
    else:
        print(f"  ⚠️ 正负样本比例可能需要调整")


def check_duplicate_documents(data: List[Dict]):
    print("\n" + "=" * 80)
    print("4. 重复文档检查")
    print("=" * 80)
    
    all_docs = set()
    duplicates = 0
    duplicate_details = []
    
    for i, item in enumerate(data):
        for doc in item.get('pos', []):
            doc_hash = hash(doc[:100])
            if doc_hash in all_docs:
                duplicates += 1
                duplicate_details.append(f"样本 {i} 正样本")
            all_docs.add(doc_hash)
        
        for doc in item.get('neg', []):
            doc_hash = hash(doc[:100])
            if doc_hash in all_docs:
                duplicates += 1
                duplicate_details.append(f"样本 {i} 负样本")
            all_docs.add(doc_hash)
    
    if duplicates > 0:
        print(f"⚠️ 发现 {duplicates} 个重复文档")
        print("重复示例:")
        for detail in duplicate_details[:5]:
            print(f"  - {detail}")
    else:
        print("✅ 未发现重复文档")
    
    print(f"总唯一文档数: {len(all_docs)}")


def check_hard_negatives_quality(data: List[Dict]):
    print("\n" + "=" * 80)
    print("5. 难负样本质量检查")
    print("=" * 80)
    
    with_inst = [item for item in data if item.get('type') == 'with_instruction']
    
    print(f"有指令样本数: {len(with_inst)}")
    print("\n随机抽取 3 个样本进行详细分析:\n")
    
    samples = random.sample(with_inst, min(3, len(with_inst)))
    
    for idx, item in enumerate(samples, 1):
        print(f"--- 样本 {idx} ---")
        print(f"Query: {item['query'][:120]}...")
        print(f"Instruction: {item['instruction'][:80]}...")
        print(f"Dataset: {item['dataset']}, Base QID: {item['base_qid']}")
        
        stats = item.get('stats', {})
        print(f"Stats: {stats}")
        
        print(f"正样本 ({len(item['pos'])} 个):")
        for i, doc in enumerate(item['pos'][:1], 1):
            print(f"  [{i}] {doc[:150]}...")
        
        print(f"难负样本 ({len(item['neg'])} 个):")
        hard_neg_count = stats.get('num_hard_negatives', 0)
        easy_neg_count = stats.get('num_easy_negatives', 0)
        print(f"  (其中难负样本: {hard_neg_count}, 易负样本: {easy_neg_count})")
        for i, doc in enumerate(item['neg'][:2], 1):
            print(f"  [{i}] {doc[:150]}...")
        print()


def generate_quality_report(data: List[Dict]):
    print("\n" + "=" * 80)
    print("6. 质量报告总结")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # 检查1: 样本数量
    if len(data) < 50:
        issues.append(f"样本数量过少 ({len(data)} < 50)")
    
    # 检查2: 正负样本比例
    pos_counts = [len(item.get('pos', [])) for item in data]
    neg_counts = [len(item.get('neg', [])) for item in data]
    ratios = [n/p for p, n in zip(pos_counts, neg_counts) if p > 0]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    
    if avg_ratio < 3:
        issues.append(f"正负样本比例过低 (1:{avg_ratio:.1f})")
    elif avg_ratio > 15:
        warnings.append(f"正负样本比例过高 (1:{avg_ratio:.1f})")
    
    # 检查3: 空值检查
    empty_queries = sum(1 for item in data if not item.get('query', '').strip())
    empty_pos = sum(1 for item in data if not item.get('pos'))
    empty_neg = sum(1 for item in data if len(item.get('neg', [])) < 3)
    
    if empty_queries > 0:
        issues.append(f"有 {empty_queries} 个空查询")
    if empty_pos > 0:
        issues.append(f"有 {empty_pos} 个样本没有正样本")
    if empty_neg > 0:
        issues.append(f"有 {empty_neg} 个样本负样本不足3个")
    
    # 检查4: 指令样本检查
    with_inst = [item for item in data if item.get('type') == 'with_instruction']
    empty_inst = sum(1 for item in with_inst if not item.get('instruction', '').strip())
    
    if empty_inst > 0:
        issues.append(f"有 {empty_inst} 个有指令样本但指令为空")
    
    # 检查5: 难负样本检查
    low_hard_neg = sum(1 for item in with_inst 
                       if item.get('stats', {}).get('num_hard_negatives', 0) < 3)
    if low_hard_neg > 0:
        warnings.append(f"有 {low_hard_neg} 个有指令样本的难负样本少于3个")
    
    # 输出报告
    print(f"总样本数: {len(data)}")
    print(f"有指令样本: {len(with_inst)}")
    print(f"无指令样本: {len(data) - len(with_inst)}")
    print(f"平均正负比例: 1:{avg_ratio:.1f}")
    
    if issues:
        print(f"\n❌ 严重问题 ({len(issues)} 个):")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print(f"\n⚠️ 警告 ({len(warnings)} 个):")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("\n✅ 数据质量检查通过，未发现明显问题")
    elif not issues:
        print("\n✅ 没有严重问题，数据集可以使用")
    
    print("\n建议:")
    if issues:
        print("  - 请先修复上述严重问题后再使用数据集")
    elif warnings:
        print("  - 数据集可以使用，但建议关注上述警告")
    else:
        print("  - 数据集质量良好，可以直接使用")


def main():
    file_path = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions_v3.jsonl"
    
    print("=" * 80)
    print("过拟合训练集全面质量检查 (v3)")
    print("=" * 80)
    print(f"检查文件: {file_path}")
    
    try:
        data = load_data(file_path)
        print(f"成功加载 {len(data)} 条数据\n")
        
        check_basic_stats(data)
        check_document_quality(data)
        check_data_balance(data)
        check_duplicate_documents(data)
        check_hard_negatives_quality(data)
        generate_quality_report(data)
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
    except Exception as e:
        print(f"❌ 检查过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
