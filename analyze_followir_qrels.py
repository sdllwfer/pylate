#!/usr/bin/env python3
"""
分析 FollowIR 测试集的 qrels 结构，理解 score 分布
"""

import os
import sys

conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env != 'pylate':
    import subprocess
    conda_path = subprocess.run(['which', 'conda'], capture_output=True, text=True).stdout.strip()
    if conda_path:
        env_python = f"{os.path.dirname(conda_path)}/envs/pylate/bin/python"
        if os.path.exists(env_python):
            os.execv(env_python, [sys.executable] + sys.argv)

import json
import datasets
import mteb
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_dataset_path(task_name: str) -> str:
    task = mteb.get_task(task_name)
    return task.metadata.dataset.get("path")


def analyze_qrels(task_name: str):
    print(f"\n{'='*80}")
    print(f"分析数据集: {task_name}")
    print(f"{'='*80}")
    
    dataset_path = get_dataset_path(task_name)
    print(f"数据集路径: {dataset_path}")
    
    # 加载 qrels (test split)
    ds_qrels = datasets.load_dataset(dataset_path, 'default', split='test')
    print(f"\nqrels 样本数: {len(ds_qrels)}")
    
    # 查看字段
    sample = ds_qrels[0]
    print(f"\n字段: {list(sample.keys())}")
    print(f"示例: {sample}")
    
    # 统计 score 分布
    score_counts = defaultdict(int)
    og_scores = defaultdict(int)
    changed_scores = defaultdict(int)
    
    # 按 query 分组统计
    query_scores = defaultdict(lambda: {'og': {}, 'changed': {}})
    
    for item in ds_qrels:
        qid = str(item.get('query-id', ''))
        doc_id = str(item.get('corpus-id', ''))
        score = item.get('score', 0)
        
        score_counts[score] += 1
        
        if '-og' in qid:
            base_qid = qid.replace('-og', '')
            og_scores[score] += 1
            query_scores[base_qid]['og'][doc_id] = score
        elif '-changed' in qid:
            base_qid = qid.replace('-changed', '')
            changed_scores[score] += 1
            query_scores[base_qid]['changed'][doc_id] = score
    
    print(f"\n📊 Score 分布 (总体):")
    for score in sorted(score_counts.keys()):
        print(f"   score={score}: {score_counts[score]} 个")
    
    print(f"\n📊 Score 分布 (og queries):")
    for score in sorted(og_scores.keys()):
        print(f"   score={score}: {og_scores[score]} 个")
    
    print(f"\n📊 Score 分布 (changed queries):")
    for score in sorted(changed_scores.keys()):
        print(f"   score={score}: {changed_scores[score]} 个")
    
    # 寻找难负样本：og 中 score>=1 但 changed 中 score=0
    print(f"\n🔍 寻找难负样本 (og score>=1, changed score=0)...")
    hard_negative_stats = []
    
    for base_qid, scores in query_scores.items():
        og_docs = scores['og']
        changed_docs = scores['changed']
        
        hard_negs = []
        for doc_id, og_score in og_docs.items():
            if og_score >= 1:
                changed_score = changed_docs.get(doc_id, 0)
                if changed_score == 0:
                    hard_negs.append({
                        'doc_id': doc_id,
                        'og_score': og_score,
                        'changed_score': changed_score
                    })
        
        if hard_negs:
            hard_negative_stats.append({
                'base_qid': base_qid,
                'count': len(hard_negs),
                'examples': hard_negs[:3]  # 只保存前3个示例
            })
    
    print(f"   找到 {len(hard_negative_stats)} 个 query 有难负样本")
    total_hard_negs = sum(s['count'] for s in hard_negative_stats)
    print(f"   总难负样本数: {total_hard_negs}")
    
    if hard_negative_stats:
        print(f"\n   示例 (前3个query):")
        for stat in hard_negative_stats[:3]:
            print(f"   Query {stat['base_qid']}: {stat['count']} 个难负样本")
            for ex in stat['examples']:
                print(f"      - doc_id: {ex['doc_id']}, og_score: {ex['og_score']}, changed_score: {ex['changed_score']}")
    
    return query_scores


def main():
    DATASETS = [
        "Core17InstructionRetrieval",
        "Robust04InstructionRetrieval", 
        "News21InstructionRetrieval",
    ]
    
    all_query_scores = {}
    
    for task_name in DATASETS:
        query_scores = analyze_qrels(task_name)
        all_query_scores[task_name] = query_scores
    
    # 保存分析结果
    output_file = "/home/luwa/Documents/pylate/followir_qrels_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # 转换为可序列化的格式
        serializable = {}
        for task, queries in all_query_scores.items():
            serializable[task] = {
                qid: {
                    'og': scores['og'],
                    'changed': scores['changed']
                }
                for qid, scores in queries.items()
            }
        json.dump(serializable, f, indent=2)
    
    print(f"\n✅ 分析结果已保存: {output_file}")


if __name__ == "__main__":
    main()
