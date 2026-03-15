#!/usr/bin/env python3
"""
全面质量检查过拟合训练集
从多个维度验证数据质量
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Any

# 设置随机种子保证可重复性
random.seed(42)


def load_data(file_path: str) -> List[Dict]:
    """加载训练数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def check_basic_stats(data: List[Dict]):
    """检查基本统计信息"""
    print("=" * 80)
    print("1. 基本统计信息")
    print("=" * 80)
    
    total = len(data)
    with_inst = [item for item in data if item.get('type') == 'with_instruction']
    without_inst = [item for item in data if item.get('type') == 'without_instruction']
    
    print(f"总样本数: {total}")
    print(f"有指令样本: {len(with_inst)} ({len(with_inst)/total*100:.1f}%)")
    print(f"无指令样本: {len(without_inst)} ({len(without_inst)/total*100:.1f}%)")
    
    # 按数据集统计
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
    """检查文档质量"""
    print("\n" + "=" * 80)
    print("2. 文档质量检查")
    print("=" * 80)
    
    issues = []
    
    for i, item in enumerate(data):
        # 检查正样本
        pos_docs = item.get('pos', [])
        if not pos_docs:
            issues.append(f"样本 {i}: 没有正样本")
        elif len(pos_docs) == 0:
            issues.append(f"样本 {i}: 正样本为空列表")
        
        for j, doc in enumerate(pos_docs):
            if not doc or not doc.strip():
                issues.append(f"样本 {i}, 正样本 {j}: 文档为空")
            elif len(doc) < 10:
                issues.append(f"样本 {i}, 正样本 {j}: 文档过短 ({len(doc)} 字符)")
        
        # 检查负样本
        neg_docs = item.get('neg', [])
        if not neg_docs:
            issues.append(f"样本 {i}: 没有负样本")
        elif len(neg_docs) < 3:
            issues.append(f"样本 {i}: 负样本数量不足 ({len(neg_docs)} < 3)")
        
        for j, doc in enumerate(neg_docs):
            if not doc or not doc.strip():
                issues.append(f"样本 {i}, 负样本 {j}: 文档为空")
            elif len(doc) < 10:
                issues.append(f"样本 {i}, 负样本 {j}: 文档过短 ({len(doc)} 字符)")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:10]:  # 只显示前10个
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("✅ 所有文档质量检查通过")
    
    # 统计文档长度分布
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
        print(f"  最小: {min(pos_lengths)} 字符")
        print(f"  最大: {max(pos_lengths)} 字符")
    
    if neg_lengths:
        print(f"\n负样本长度统计:")
        print(f"  平均: {sum(neg_lengths)/len(neg_lengths):.0f} 字符")
        print(f"  中位数: {sorted(neg_lengths)[len(neg_lengths)//2]} 字符")
        print(f"  最小: {min(neg_lengths)} 字符")
        print(f"  最大: {max(neg_lengths)} 字符")


def check_query_quality(data: List[Dict]):
    """检查查询质量"""
    print("\n" + "=" * 80)
    print("3. 查询质量检查")
    print("=" * 80)
    
    issues = []
    query_lengths = []
    instruction_lengths = []
    
    for i, item in enumerate(data):
        query = item.get('query', '')
        instruction = item.get('instruction', '')
        
        if not query or not query.strip():
            issues.append(f"样本 {i}: 查询为空")
        else:
            query_lengths.append(len(query))
        
        if item.get('type') == 'with_instruction':
            if not instruction or not instruction.strip():
                issues.append(f"样本 {i}: 有指令样本但指令为空")
            else:
                instruction_lengths.append(len(instruction))
        
        # 检查查询和指令是否重复
        if instruction and query.strip() == instruction.strip():
            issues.append(f"样本 {i}: 查询和指令完全相同")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:")
        for issue in issues[:10]:
            print(f"  - {issue}")
        if len(issues) > 10:
            print(f"  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("✅ 所有查询质量检查通过")
    
    if query_lengths:
        print(f"\n查询长度统计:")
        print(f"  平均: {sum(query_lengths)/len(query_lengths):.0f} 字符")
        print(f"  中位数: {sorted(query_lengths)[len(query_lengths)//2]} 字符")
        print(f"  最小: {min(query_lengths)} 字符")
        print(f"  最大: {max(query_lengths)} 字符")
    
    if instruction_lengths:
        print(f"\n指令长度统计:")
        print(f"  平均: {sum(instruction_lengths)/len(instruction_lengths):.0f} 字符")
        print(f"  中位数: {sorted(instruction_lengths)[len(instruction_lengths)//2]} 字符")
        print(f"  最小: {min(instruction_lengths)} 字符")
        print(f"  最大: {max(instruction_lengths)} 字符")


def check_hard_negatives_quality(data: List[Dict]):
    """检查难负样本质量（语义相关性）"""
    print("\n" + "=" * 80)
    print("4. 难负样本质量检查 (语义层面)")
    print("=" * 80)
    
    with_inst = [item for item in data if item.get('type') == 'with_instruction']
    
    print(f"检查 {len(with_inst)} 个有指令样本的难负样本...")
    print("\n随机抽取 5 个样本进行详细分析:\n")
    
    samples = random.sample(with_inst, min(5, len(with_inst)))
    
    for idx, item in enumerate(samples, 1):
        print(f"--- 样本 {idx} ---")
        print(f"Query: {item['query'][:150]}...")
        print(f"Instruction: {item['instruction'][:100]}...")
        print(f"Dataset: {item['dataset']}, Base QID: {item['base_qid']}")
        
        stats = item.get('stats', {})
        print(f"Stats: {stats}")
        
        print(f"\n正样本 ({len(item['pos'])} 个):")
        for i, doc in enumerate(item['pos'][:2], 1):
            print(f"  [{i}] {doc[:200]}...")
        
        print(f"\n难负样本 ({len(item['neg'])} 个):")
        for i, doc in enumerate(item['neg'][:2], 1):
            print(f"  [{i}] {doc[:200]}...")
        
        # 简单语义检查：查看正样本和难负样本是否有明显主题差异
        print("\n语义分析:")
        query_keywords = set(item['query'].lower().split()[:10])
        pos_keywords = set(' '.join(item['pos'][:1]).lower().split()[:20])
        neg_keywords = set(' '.join(item['neg'][:1]).lower().split()[:20])
        
        pos_overlap = query_keywords & pos_keywords
        neg_overlap = query_keywords & neg_keywords
        
        print(f"  查询与正样本关键词重叠: {len(pos_overlap)} 个")
        print(f"  查询与难负样本关键词重叠: {len(neg_overlap)} 个")
        
        if len(pos_overlap) > len(neg_overlap):
            print(f"  ✅ 正样本与查询相关性更高")
        else:
            print(f"  ⚠️ 难负样本与查询也有较高相关性（符合难负样本定义）")
        
        print()


def check_data_balance(data: List[Dict]):
    """检查数据平衡性"""
    print("\n" + "=" * 80)
    print("5. 数据平衡性检查")
    print("=" * 80)
    
    # 检查正负样本比例
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
    
    # 检查正负比例
    ratios = [n/p if p > 0 else 0 for p, n in zip(pos_counts, neg_counts)]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    print(f"\n正负样本比例:")
    print(f"  平均比例: 1:{avg_ratio:.1f}")
    print(f"  推荐比例: 1:3 到 1:10")
    
    if 3 <= avg_ratio <= 10:
        print(f"  ✅ 正负样本比例合理")
    else:
        print(f"  ⚠️ 正负样本比例可能需要调整")


def check_duplicate_documents(data: List[Dict]):
    """检查重复文档"""
    print("\n" + "=" * 80)
    print("6. 重复文档检查")
    print("=" * 80)
    
    # 收集所有文档
    all_docs = set()
    duplicates = []
    
    for item in data:
        for doc in item.get('pos', []):
            doc_hash = hash(doc[:100])  # 使用前100字符作为指纹
            if doc_hash in all_docs:
                duplicates.append(doc[:100])
            all_docs.add(doc_hash)
        
        for doc in item.get('neg', []):
            doc_hash = hash(doc[:100])
            if doc_hash in all_docs:
                duplicates.append(doc[:100])
            all_docs.add(doc_hash)
    
    if duplicates:
        print(f"⚠️ 发现 {len(duplicates)} 个可能重复的文档")
        print("重复文档示例:")
        for dup in duplicates[:3]:
            print(f"  - {dup}...")
    else:
        print("✅ 未发现明显重复文档")


def generate_quality_report(data: List[Dict]):
    """生成质量报告"""
    print("\n" + "=" * 80)
    print("7. 质量报告总结")
    print("=" * 80)
    
    issues = []
    warnings = []
    
    # 检查1: 样本数量
    if len(data) < 100:
        warnings.append(f"样本数量较少 ({len(data)} < 100)")
    
    # 检查2: 正负样本比例
    pos_counts = [len(item.get('pos', [])) for item in data]
    neg_counts = [len(item.get('neg', [])) for item in data]
    avg_ratio = sum(n/p for p, n in zip(pos_counts, neg_counts) if p > 0) / len(data)
    
    if avg_ratio < 2:
        issues.append(f"正负样本比例过低 (1:{avg_ratio:.1f})")
    elif avg_ratio > 15:
        warnings.append(f"正负样本比例过高 (1:{avg_ratio:.1f})")
    
    # 检查3: 空值检查
    empty_queries = sum(1 for item in data if not item.get('query', '').strip())
    empty_pos = sum(1 for item in data if not item.get('pos'))
    empty_neg = sum(1 for item in data if not item.get('neg'))
    
    if empty_queries > 0:
        issues.append(f"有 {empty_queries} 个空查询")
    if empty_pos > 0:
        issues.append(f"有 {empty_pos} 个样本没有正样本")
    if empty_neg > 0:
        issues.append(f"有 {empty_neg} 个样本没有负样本")
    
    # 检查4: 指令样本检查
    with_inst = [item for item in data if item.get('type') == 'with_instruction']
    empty_inst = sum(1 for item in with_inst if not item.get('instruction', '').strip())
    
    if empty_inst > 0:
        issues.append(f"有 {empty_inst} 个有指令样本但指令为空")
    
    # 输出报告
    print(f"总样本数: {len(data)}")
    print(f"有指令样本: {len(with_inst)}")
    print(f"无指令样本: {len(data) - len(with_inst)}")
    
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
    
    print("\n建议:")
    if issues:
        print("  - 请先修复上述严重问题后再使用数据集")
    elif warnings:
        print("  - 数据集可以使用，但建议关注上述警告")
    else:
        print("  - 数据集质量良好，可以直接使用")


def main():
    file_path = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions_v2.jsonl"
    
    print("=" * 80)
    print("过拟合训练集全面质量检查")
    print("=" * 80)
    print(f"检查文件: {file_path}")
    
    try:
        data = load_data(file_path)
        print(f"成功加载 {len(data)} 条数据\n")
        
        # 执行各项检查
        check_basic_stats(data)
        check_document_quality(data)
        check_query_quality(data)
        check_hard_negatives_quality(data)
        check_data_balance(data)
        check_duplicate_documents(data)
        generate_quality_report(data)
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {file_path}")
    except Exception as e:
        print(f"❌ 检查过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
