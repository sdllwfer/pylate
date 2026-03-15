#!/usr/bin/env python3
"""
从 FollowIR 测试集中正确提取 hard negatives

Hard negatives 定义：
- 在 og query 下 score >= 1（相关）
- 但在 changed query 下 score = 0（不相关）

这意味着文档在原始查询下排名靠前，但在添加 instruction 后排名靠后
"""

import datasets
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_followir_data(task_name: str):
    """加载 FollowIR 数据集的所有组件"""
    import mteb
    
    task = mteb.get_task(task_name)
    path = task.metadata.dataset.get("path")
    
    print(f"\n加载数据集: {task_name}")
    print(f"路径: {path}")
    
    # 加载 corpus
    ds_corpus = datasets.load_dataset(path, 'corpus')
    corpus = {}
    for d in ds_corpus['corpus']:
        corpus[d['_id']] = {
            'title': d.get('title', ''),
            'text': d.get('text', '')
        }
    
    # 加载 queries
    ds_queries = datasets.load_dataset(path, 'queries')
    queries = {}
    for q in ds_queries['queries']:
        queries[q['_id']] = q['text']
    
    # 加载 instructions
    ds_inst = datasets.load_dataset(path, 'instruction')
    instructions = {}
    for inst in ds_inst['instruction']:
        qid = inst['query-id']
        instructions[qid] = inst['instruction']
    
    # 加载 test split（包含相关性分数）
    ds_test = datasets.load_dataset(path)
    
    # 构建 query-corpus-score 映射
    og_scores = defaultdict(dict)  # og_scores[query_id][doc_id] = score
    changed_scores = defaultdict(dict)
    
    for item in ds_test['test']:
        qid = item['query-id']
        doc_id = item['corpus-id']
        score = item['score']
        
        if qid.endswith('-og'):
            base_qid = qid[:-3]  # 去掉 '-og'
            og_scores[base_qid][doc_id] = score
        elif qid.endswith('-changed'):
            base_qid = qid[:-8]  # 去掉 '-changed'
            changed_scores[base_qid][doc_id] = score
    
    return corpus, queries, instructions, og_scores, changed_scores


def identify_hard_negatives(
    base_qid: str,
    og_scores: Dict[str, float],
    changed_scores: Dict[str, float],
    corpus: Dict,
    queries: Dict,
    instructions: Dict
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    识别 hard negatives
    
    返回: (positives, hard_negatives, easy_negatives)
    """
    og_qid = f"{base_qid}-og"
    changed_qid = f"{base_qid}-changed"
    
    # 获取该 query 的所有文档分数
    og_doc_scores = og_scores.get(base_qid, {})
    changed_doc_scores = changed_scores.get(base_qid, {})
    
    positives = []
    hard_negatives = []
    easy_negatives = []
    
    # 遍历所有在 og query 中有分数的文档
    for doc_id, og_score in og_doc_scores.items():
        changed_score = changed_doc_scores.get(doc_id, 0)
        doc_text = corpus.get(doc_id, {}).get('text', '')
        
        if og_score >= 1 and changed_score >= 1:
            # 在两种查询下都相关 - 正样本
            positives.append({
                'doc_id': doc_id,
                'text': doc_text,
                'og_score': og_score,
                'changed_score': changed_score
            })
        elif og_score >= 1 and changed_score == 0:
            # 在 og 下相关，但在 changed 下不相关 - HARD NEGATIVE!
            hard_negatives.append({
                'doc_id': doc_id,
                'text': doc_text,
                'og_score': og_score,
                'changed_score': changed_score
            })
        elif og_score == 0 and changed_score == 0:
            # 在两种查询下都不相关 - Easy negative
            easy_negatives.append({
                'doc_id': doc_id,
                'text': doc_text,
                'og_score': og_score,
                'changed_score': changed_score
            })
    
    return positives, hard_negatives, easy_negatives


def analyze_task(task_name: str):
    """分析单个任务的数据集质量"""
    corpus, queries, instructions, og_scores, changed_scores = load_followir_data(task_name)
    
    # 获取所有 base query IDs
    all_base_qids = set(og_scores.keys()) | set(changed_scores.keys())
    
    results = []
    stats = {
        'total_queries': 0,
        'has_hard_negatives': 0,
        'lacks_hard_negatives': 0,
        'total_positives': 0,
        'total_hard_negatives': 0,
        'total_easy_negatives': 0
    }
    
    for base_qid in sorted(all_base_qids):
        og_qid = f"{base_qid}-og"
        changed_qid = f"{base_qid}-changed"
        
        query_text = queries.get(og_qid, '')
        instruction_text = instructions.get(og_qid, '')
        
        positives, hard_negatives, easy_negatives = identify_hard_negatives(
            base_qid, og_scores, changed_scores, corpus, queries, instructions
        )
        
        stats['total_queries'] += 1
        stats['total_positives'] += len(positives)
        stats['total_hard_negatives'] += len(hard_negatives)
        stats['total_easy_negatives'] += len(easy_negatives)
        
        has_hard = len(hard_negatives) > 0
        if has_hard:
            stats['has_hard_negatives'] += 1
        else:
            stats['lacks_hard_negatives'] += 1
        
        result = {
            'base_qid': base_qid,
            'query': query_text,
            'instruction': instruction_text,
            'has_hard_negatives': has_hard,
            'num_positives': len(positives),
            'num_hard_negatives': len(hard_negatives),
            'num_easy_negatives': len(easy_negatives),
            'positives': positives[:5],  # 只保存前5个作为示例
            'hard_negatives': hard_negatives[:5],
            'easy_negatives': easy_negatives[:5]
        }
        results.append(result)
    
    return results, stats


def main():
    tasks = ["Core17InstructionRetrieval", "News21InstructionRetrieval", "Robust04InstructionRetrieval"]
    
    all_results = []
    all_stats = {
        'total_queries': 0,
        'has_hard_negatives': 0,
        'lacks_hard_negatives': 0,
        'total_positives': 0,
        'total_hard_negatives': 0,
        'total_easy_negatives': 0
    }
    
    for task in tasks:
        results, stats = analyze_task(task)
        all_results.extend(results)
        
        for key in all_stats:
            all_stats[key] += stats[key]
        
        print(f"\n{'='*80}")
        print(f"{task} 统计:")
        print(f"{'='*80}")
        print(f"总查询数: {stats['total_queries']}")
        print(f"有 hard negatives: {stats['has_hard_negatives']} ({stats['has_hard_negatives']/stats['total_queries']*100:.1f}%)")
        print(f"缺乏 hard negatives: {stats['lacks_hard_negatives']} ({stats['lacks_hard_negatives']/stats['total_queries']*100:.1f}%)")
        print(f"总正样本数: {stats['total_positives']}")
        print(f"总 hard negatives: {stats['total_hard_negatives']}")
        print(f"总 easy negatives: {stats['total_easy_negatives']}")
    
    # 保存结果
    output_file = '/home/luwa/Documents/pylate/followir_hard_negatives_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': all_stats,
            'details': all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("总体统计:")
    print(f"{'='*80}")
    print(f"总查询数: {all_stats['total_queries']}")
    print(f"有 hard negatives: {all_stats['has_hard_negatives']} ({all_stats['has_hard_negatives']/all_stats['total_queries']*100:.1f}%)")
    print(f"缺乏 hard negatives: {all_stats['lacks_hard_negatives']} ({all_stats['lacks_hard_negatives']/all_stats['total_queries']*100:.1f}%)")
    print(f"总正样本数: {all_stats['total_positives']}")
    print(f"总 hard negatives: {all_stats['total_hard_negatives']}")
    print(f"总 easy negatives: {all_stats['total_easy_negatives']}")
    print(f"\n详细结果已保存到: {output_file}")
    
    # 显示几个有 hard negatives 的示例
    print(f"\n{'='*80}")
    print("有 hard negatives 的查询示例:")
    print(f"{'='*80}")
    examples_with_hard = [r for r in all_results if r['has_hard_negatives']][:3]
    for i, ex in enumerate(examples_with_hard):
        print(f"\n示例 {i+1} (Query ID: {ex['base_qid']}):")
        print(f"Query: {ex['query'][:100]}...")
        print(f"Instruction: {ex['instruction'][:100]}...")
        print(f"Hard negatives 数量: {ex['num_hard_negatives']}")
        if ex['hard_negatives']:
            hn = ex['hard_negatives'][0]
            print(f"第一个 Hard negative (og_score={hn['og_score']}, changed_score={hn['changed_score']}):")
            print(f"  {hn['text'][:150]}...")


if __name__ == '__main__':
    main()
