#!/usr/bin/env python3
"""
p-MRR 手动验证脚本
用于验证 FollowIR 评估结果中 p-MRR 值的正确性
"""

import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def load_trec_run(run_path):
    """加载 TREC 格式的 run 文件"""
    print(f"\n📂 加载 TREC 文件: {run_path}")
    results = {}
    with open(run_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid = parts[0]
                doc_id = parts[2]
                score = float(parts[4])
                
                if qid not in results:
                    results[qid] = {}
                results[qid][doc_id] = score
    
    print(f"   ✅ 加载了 {len(results)} 个查询")
    return results


def load_qrel_diff(path):
    """加载 qrel_diff 数据"""
    print(f"\n📂 加载 qrel_diff: {path}")
    from datasets import load_dataset
    ds_diff = load_dataset(path, 'qrel_diff', split='qrel_diff')
    changed_qrels = {}
    for item in ds_diff:
        qid = item['query-id']
        corpus_ids = item['corpus-ids']
        if corpus_ids:
            changed_qrels[qid] = corpus_ids
    
    print(f"   ✅ 加载了 {len(changed_qrels)} 个变化的查询")
    return changed_qrels


def get_rank_from_dict(rank_dict, doc_id):
    """
    从排名字典中获取文档的排名
    按分数降序排列，分数越高排名越靠前（排名从1开始）
    """
    if doc_id not in rank_dict:
        return -1, None
    
    sorted_docs = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (did, score) in enumerate(sorted_docs, start=1):
        if did == doc_id:
            return rank, score
    return -1, None


def rank_score(og_rank, new_rank):
    """
    计算单个文档的 rank score
    公式来自 FollowIR 论文：
    - 如果 og_rank >= new_rank: ((1/og_rank) / (1/new_rank)) - 1 = new_rank/og_rank - 1
    - 如果 og_rank < new_rank: 1 - ((1/new_rank) / (1/og_rank)) = 1 - new_rank/og_rank
    """
    if og_rank <= 0 or new_rank <= 0:
        return 0.0
    
    if og_rank >= new_rank:
        result = (1 / og_rank) / (1 / new_rank) - 1
    else:
        result = 1 - ((1 / new_rank) / (1 / og_rank))
    
    return result


def calculate_pmrr_detailed(original_run, new_run, changed_qrels, verbose=True):
    """
    详细计算 p-MRR
    
    original_run: og 查询的检索结果 {qid: {doc_id: score}}
    new_run: changed 查询的检索结果 {qid: {doc_id: score}}
    changed_qrels: 变化的文档 {qid: [doc_id1, doc_id2, ...]}
    """
    print("\n" + "="*60)
    print("🔍 开始详细计算 p-MRR")
    print("="*60)
    
    changes = []
    skipped_missing = 0
    skipped_not_found = 0
    
    for qid in tqdm(changed_qrels.keys(), desc="处理查询"):
        og_key = qid + '-og'
        changed_key = qid + '-changed'
        
        if og_key not in original_run:
            if verbose:
                print(f"   ⚠️ 跳过 {qid}: 原始查询结果不存在")
            skipped_missing += 1
            continue
            
        if changed_key not in new_run:
            if verbose:
                print(f"   ⚠️ 跳过 {qid}: 变化查询结果不存在")
            skipped_missing += 1
            continue
        
        original_qid_run = original_run[og_key]
        new_qid_run = new_run[changed_key]
        
        for idx, changed_doc in enumerate(changed_qrels[qid]):
            original_rank, original_score = get_rank_from_dict(original_qid_run, changed_doc)
            new_rank, new_score = get_rank_from_dict(new_qid_run, changed_doc)
            
            if original_rank < 0 or new_rank < 0:
                skipped_not_found += 1
                continue
            
            score = rank_score(original_rank, new_rank)
            
            change_info = {
                'qid': qid,
                'doc_id': changed_doc,
                'original_rank': original_rank,
                'new_rank': new_rank,
                'original_score': original_score,
                'new_score': new_score,
                'rank_score': score,
                'rank_change': original_rank - new_rank
            }
            changes.append(change_info)
    
    print(f"\n📊 统计信息:")
    print(f"   - 成功处理的变化文档: {len(changes)}")
    print(f"   - 跳过 (查询结果不存在): {skipped_missing}")
    print(f"   - 跳过 (文档不在结果中): {skipped_not_found}")
    
    if len(changes) == 0:
        print("\n❌ 没有找到任何有效的变化文档!")
        return 0.0, []
    
    print(f"\n📋 前10个变化文档详情:")
    print("-" * 100)
    print(f"{'Query ID':<10} {'Doc ID':<20} {'OG Rank':<10} {'New Rank':<10} {'Rank Change':<15} {'Score':<10}")
    print("-" * 100)
    for c in changes[:10]:
        print(f"{c['qid']:<10} {c['doc_id']:<20} {c['original_rank']:<10} {c['new_rank']:<10} {c['rank_change']:<15} {c['rank_score']:<10.6f}")
    print("-" * 100)
    
    # 按查询分组计算平均分
    qid_scores = defaultdict(list)
    for c in changes:
        qid_scores[c['qid']].append(c['rank_score'])
    
    qid_avg = {}
    for qid, scores in qid_scores.items():
        qid_avg[qid] = sum(scores) / len(scores)
    
    print(f"\n📈 各查询的平均 p-MRR:")
    for qid in sorted(qid_avg.keys())[:5]:
        print(f"   {qid}: {qid_avg[qid]:.6f}")
    if len(qid_avg) > 5:
        print(f"   ... 还有 {len(qid_avg) - 5} 个查询")
    
    # 计算最终 p-MRR (按查询平均后再平均，与 MTEB 官方一致)
    if len(qid_avg) > 0:
        pmrr = sum(qid_avg.values()) / len(qid_avg)
    else:
        pmrr = 0.0
    
    print(f"\n🎯 最终 p-MRR = {pmrr:.10f}")
    print(f"   (基于 {len(changes)} 个变化文档，{len(qid_avg)} 个查询)")
    
    return pmrr, changes


def main():
    parser = argparse.ArgumentParser(description='手动验证 p-MRR 计算')
    parser.add_argument('--run_og', type=str, required=True, help='原始指令 (og) 的 TREC 结果文件')
    parser.add_argument('--run_changed', type=str, required=True, help='变化指令 (changed) 的 TREC 结果文件')
    parser.add_argument('--qrel_diff', type=str, default='jhu-clsp/core17-instructions-mteb', help='qrel_diff 数据集路径')
    parser.add_argument('--task', type=str, default='Core17InstructionRetrieval', help='任务名称')
    parser.add_argument('--summary_file', type=str, help='官方结果摘要文件 (JSON)')
    parser.add_argument('--verbose', action='store_true', default=True, help='显示详细日志')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 p-MRR 手动验证脚本")
    print("="*60)
    
    # 1. 加载 TREC 文件
    original_run = load_trec_run(args.run_og)
    new_run = load_trec_run(args.run_changed)
    
    # 合并结果用于分离 og 和 changed
    all_results = {**original_run, **new_run}
    og_results = {k: v for k, v in all_results.items() if k.endswith('-og')}
    changed_results = {k: v for k, v in all_results.items() if not k.endswith('-og')}
    
    print(f"\n📊 结果统计:")
    print(f"   - og 查询数: {len(og_results)}")
    print(f"   - changed 查询数: {len(changed_results)}")
    
    # 2. 加载 qrel_diff
    changed_qrels = load_qrel_diff(args.qrel_diff)
    
    # 3. 手动计算 p-MRR
    pmrr, changes = calculate_pmrr_detailed(
        og_results, 
        changed_results, 
        changed_qrels,
        verbose=args.verbose
    )
    
    # 4. 与官方结果对比
    if args.summary_file and os.path.exists(args.summary_file):
        print("\n" + "="*60)
        print("📊 与官方结果对比")
        print("="*60)
        
        with open(args.summary_file, 'r') as f:
            summary = json.load(f)
        
        # 尝试多种可能的 key 格式
        task_name = args.task
        if task_name in summary:
            official_pmrr = summary[task_name].get('p-MRR', summary[task_name].get('p_mrr', 'N/A'))
        else:
            official_pmrr = summary.get('p-MRR', summary.get('p_mrr', 'N/A'))
        
        print(f"\n   官方 p-MRR: {official_pmrr}")
        print(f"   手动计算:   {pmrr:.10f}")
        
        if isinstance(official_pmrr, (int, float)):
            diff = abs(pmrr - official_pmrr)
            print(f"   差异:      {diff:.10f}")
            
            if diff < 0.0001:
                print("\n   ✅ 验证通过! 差异在可接受范围内 (< 0.0001)")
            else:
                print(f"\n   ⚠️ 差异较大! 差异 = {diff:.6f}")
                # 计算百分比差异
                if official_pmrr != 0:
                    pct_diff = (diff / abs(official_pmrr)) * 100
                    print(f"   百分比差异: {pct_diff:.2f}%")
    else:
        print("\n⚠️ 未提供官方结果文件，无法对比")
    
    print("\n" + "="*60)
    print("✅ 验证完成")
    print("="*60)
    
    return pmrr


if __name__ == '__main__':
    main()
