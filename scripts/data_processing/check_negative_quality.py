"""
检查无指令数据集中负向文档的质量
分析负向文档与查询的相关性
"""

import json
import random
from collections import Counter


def check_negative_quality(dataset_path: str, sample_size: int = 20):
    """
    检查负向文档质量
    
    Args:
        dataset_path: 数据集路径
        sample_size: 随机抽查的样本数量
    """
    print(f"📂 加载数据集: {dataset_path}")
    
    # 读取数据集
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"   共 {len(samples)} 个样本\n")
    
    # 收集所有文档用于分析
    all_pos_docs = []
    all_neg_docs = []
    query_topics = []
    
    for sample in samples:
        query = sample['query']
        pos_docs = sample.get('pos', [])
        neg_docs = sample.get('neg', [])
        
        all_pos_docs.extend(pos_docs)
        all_neg_docs.extend(neg_docs)
        
        # 简单提取查询主题（取前3个词）
        topic = ' '.join(query.split()[:3])
        query_topics.append(topic)
    
    print("=" * 70)
    print("📊 数据集整体统计")
    print("=" * 70)
    print(f"总样本数: {len(samples)}")
    print(f"正文档总数: {len(all_pos_docs)}")
    print(f"负文档总数: {len(all_neg_docs)}")
    print(f"\n查询主题分布（前10）:")
    topic_counts = Counter(query_topics)
    for topic, count in topic_counts.most_common(10):
        print(f"  - {topic}: {count}个")
    
    # 随机抽查样本
    print(f"\n" + "=" * 70)
    print(f"🔍 随机抽查 {sample_size} 个样本的负向文档质量")
    print("=" * 70)
    
    random_samples = random.sample(samples, min(sample_size, len(samples)))
    
    for i, sample in enumerate(random_samples, 1):
        query = sample['query']
        pos_docs = sample.get('pos', [])
        neg_docs = sample.get('neg', [])
        
        print(f"\n{'─' * 70}")
        print(f"样本 {i}/{sample_size}")
        print(f"{'─' * 70}")
        print(f"📌 查询: {query}")
        print(f"\n✅ 正文档 ({len(pos_docs)}个):")
        for j, doc in enumerate(pos_docs[:2], 1):  # 只显示前2个
            print(f"   {j}. {doc[:100]}...")
        if len(pos_docs) > 2:
            print(f"   ... 还有 {len(pos_docs)-2} 个")
        
        print(f"\n❌ 负文档 ({len(neg_docs)}个):")
        for j, doc in enumerate(neg_docs[:3], 1):  # 显示前3个
            # 简单检查是否可能相关
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = query_words & doc_words
            overlap_ratio = len(overlap) / len(query_words) if query_words else 0
            
            warning = "⚠️ " if overlap_ratio > 0.3 else "   "
            print(f"{warning}{j}. [{len(overlap)}个词重叠] {doc[:100]}...")
            if overlap:
                print(f"       重叠词: {', '.join(list(overlap)[:5])}")
        if len(neg_docs) > 3:
            print(f"   ... 还有 {len(neg_docs)-3} 个")
    
    # 分析负文档与查询的词重叠情况
    print(f"\n" + "=" * 70)
    print("📈 负文档与查询的词重叠分析")
    print("=" * 70)
    
    overlap_ratios = []
    high_overlap_count = 0
    
    for sample in samples:
        query = sample['query']
        neg_docs = sample.get('neg', [])
        query_words = set(query.lower().split())
        
        for doc in neg_docs:
            doc_words = set(doc.lower().split())
            if query_words:
                overlap = query_words & doc_words
                overlap_ratio = len(overlap) / len(query_words)
                overlap_ratios.append(overlap_ratio)
                
                if overlap_ratio > 0.5:  # 重叠超过50%认为可能有问题
                    high_overlap_count += 1
    
    if overlap_ratios:
        avg_overlap = sum(overlap_ratios) / len(overlap_ratios)
        print(f"平均词重叠比例: {avg_overlap:.2%}")
        print(f"高重叠文档数(>50%): {high_overlap_count} / {len(overlap_ratios)} ({high_overlap_count/len(overlap_ratios):.2%})")
        
        # 分布
        print(f"\n词重叠比例分布:")
        bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]
        for low, high in bins:
            count = sum(1 for r in overlap_ratios if low <= r < high)
            print(f"  [{low*100:.0f}%-{high*100:.0f}%): {count}个 ({count/len(overlap_ratios):.1%})")
    
    # 检查负文档是否出现在当前查询的正文档中（这是真正的问题）
    print(f"\n" + "=" * 70)
    print("🔍 检查负文档是否出现在当前查询的正文档中（严格检查）")
    print("=" * 70)
    
    # 区分有指令和无指令样本
    instr_samples = [s for s in samples if 'instruction' in s]
    no_instr_samples = [s for s in samples if 'instruction' not in s]
    
    print(f"   有指令样本: {len(instr_samples)}个")
    print(f"   无指令样本: {len(no_instr_samples)}个")
    
    # 对于无指令样本：检查neg是否在其pos中（这是问题）
    # 对于有指令样本：neg本来就是负样本，不需要检查
    bad_negatives = []
    for sample in no_instr_samples:
        query = sample['query']
        pos_docs = set(sample.get('pos', []))
        for neg_doc in sample.get('neg', []):
            if neg_doc in pos_docs:
                bad_negatives.append((query, neg_doc))
    
    if bad_negatives:
        print(f"\n⚠️ 发现 {len(bad_negatives)} 个无指令样本的负文档同时是其查询的正文档！")
        print("部分示例:")
        for query, doc in bad_negatives[:3]:
            print(f"  查询: {query}")
            print(f"  文档: {doc[:80]}...")
            print()
    else:
        print("\n✅ 无指令样本中没有发现负文档出现在其正文档中")
    
    # 额外统计：负文档作为其他查询正文档的情况（这是正常的）
    print(f"\n📊 统计信息:")
    
    # 重新收集所有正文档集合
    all_pos_docs_set = set()
    for sample in samples:
        all_pos_docs_set.update(sample.get('pos', []))
    
    neg_as_other_pos = 0
    for sample in no_instr_samples:  # 只检查无指令样本
        query = sample['query']
        query_pos = set(sample.get('pos', []))
        for neg_doc in sample.get('neg', []):
            if neg_doc in all_pos_docs_set and neg_doc not in query_pos:
                neg_as_other_pos += 1
    
    print(f"   - 无指令样本的负文档同时是其他查询的正文档: {neg_as_other_pos}个（这是正常的）")
    print(f"   - 无指令样本的负文档同时是当前查询的正文档: {len(bad_negatives)}个（这是问题）")
    
    print("\n" + "=" * 70)
    print("💡 质量评估建议")
    print("=" * 70)
    
    if high_overlap_count / len(overlap_ratios) > 0.1:
        print("⚠️ 警告: 超过10%的负文档与查询有较高的词重叠(>50%)")
        print("   建议: 考虑使用更严格的负采样策略，如BM25负采样")
    else:
        print("✅ 负文档与查询的词重叠比例在合理范围内")
    
    if bad_negatives:
        print("⚠️ 警告: 部分无指令样本的负文档同时是当前查询的正文档")
        print("   建议: 这会导致训练时的严重混淆，必须清理")
    else:
        print("✅ 无指令样本的负文档没有与当前查询的正文档重叠（质量良好）")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查负向文档质量')
    parser.add_argument('--dataset', type=str,
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/no_instruction_train.jsonl',
                        help='数据集路径')
    parser.add_argument('--samples', type=int, default=20,
                        help='抽查样本数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    check_negative_quality(args.dataset, args.samples)


if __name__ == '__main__':
    main()
