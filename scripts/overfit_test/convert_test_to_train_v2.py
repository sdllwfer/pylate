#!/usr/bin/env python3
"""
从 FollowIR 测试集采样并转换为 ColBERT 训练格式 (改进版)
基于相似度分数正确识别难负样本：
- 难负样本：og query 中 score >= 1，但在 changed query 中 score = 0 的文档

混合版本：
1. 有指令样本 (50%): instruction 不为空
   - 正例：changed query 中 score >= 1 的文档
   - 负例：难负样本（og 中 score >= 1，changed 中 score = 0）
2. 无指令样本 (50%): instruction 为空
   - 正例：og query 中 score >= 1 的文档
   - 负例：og 查询中 score = 0 的样本
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
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATASETS = [
    {"name": "core17", "task": "Core17InstructionRetrieval"},
    {"name": "news21", "task": "News21InstructionRetrieval"},
    {"name": "robust04", "task": "Robust04InstructionRetrieval"},
]

OUTPUT_DIR = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data"
SAMPLES_PER_DATASET = 100  # 每个数据集采样的 query 数量
MIN_HARD_NEGATIVES = 3  # 每个 query 最少需要的难负样本数
MIN_POSITIVES = 1  # 每个 query 最少需要的正样本数

random.seed(42)
np.random.seed(42)


def get_dataset_path(task_name: str) -> str:
    import mteb
    task = mteb.get_task(task_name)
    return task.metadata.dataset.get("path")


def load_followir_data(task_name: str) -> Dict[str, Any]:
    import datasets
    
    dataset_path = get_dataset_path(task_name)
    print(f"📂 加载数据集: {dataset_path}")
    
    ds_corpus = datasets.load_dataset(dataset_path, 'corpus')
    ds_queries = datasets.load_dataset(dataset_path, 'queries')
    ds_instructions = datasets.load_dataset(dataset_path, 'instruction')
    ds_qrels = datasets.load_dataset(dataset_path, 'default', split='test')
    
    print(f"   加载完成: {len(ds_corpus['corpus'])} docs, {len(ds_queries['queries'])} queries")
    
    # 构建 corpus 字典
    corpus_dict = {}
    for item in ds_corpus['corpus']:
        doc_id = str(item.get('_id', item.get('id', '')))
        text = item.get('text', '')
        title = item.get('title', '')
        corpus_dict[doc_id] = {
            'id': doc_id,
            'text': f"{title} {text}".strip() if title else text
        }
    
    # 构建 query 字典
    query_dict = {}
    for item in ds_queries['queries']:
        qid = str(item.get('_id', item.get('id', '')))
        query_dict[qid] = item.get('text', '')
    
    # 构建 instruction 字典
    instruction_dict = {}
    for item in ds_instructions['instruction']:
        qid = str(item.get('query-id', ''))
        instruction_dict[qid] = item.get('instruction', '')
    
    # 按 base_qid 组织 qrels
    # 结构: {base_qid: {'og': {doc_id: score}, 'changed': {doc_id: score}}}
    qrels_by_query = defaultdict(lambda: {'og': {}, 'changed': {}})
    
    for item in ds_qrels:
        qid = str(item.get('query-id', ''))
        doc_id = str(item.get('corpus-id', ''))
        score = item.get('score', 0)
        
        if '-og' in qid:
            base_qid = qid.replace('-og', '')
            qrels_by_query[base_qid]['og'][doc_id] = score
        elif '-changed' in qid:
            base_qid = qid.replace('-changed', '')
            qrels_by_query[base_qid]['changed'][doc_id] = score
    
    print(f"   qrels: {len(qrels_by_query)} unique queries")
    
    task = task_name.replace("InstructionRetrieval", "")
    
    return {
        'corpus': corpus_dict,
        'queries': query_dict,
        'instructions': instruction_dict,
        'qrels': qrels_by_query,
        'name': task
    }


def identify_hard_negatives(qrels: Dict[str, Dict[str, float]]) -> Tuple[List[str], List[str], List[str]]:
    """
    识别难负样本
    
    返回: (positive_docs, hard_negative_docs, easy_negative_docs)
    - 正样本：changed query 中 score >= 1
    - 难负样本：og 中 score >= 1，changed 中 score = 0
    - 易负样本：og 和 changed 中 score = 0
    """
    og_scores = qrels['og']
    changed_scores = qrels['changed']
    
    positive_docs = []
    hard_negative_docs = []
    easy_negative_docs = []
    
    # 获取所有文档
    all_docs = set(og_scores.keys()) | set(changed_scores.keys())
    
    for doc_id in all_docs:
        og_score = og_scores.get(doc_id, 0)
        changed_score = changed_scores.get(doc_id, 0)
        
        if changed_score >= 1:
            # 在 changed query 下相关 - 正样本
            positive_docs.append(doc_id)
        elif og_score >= 1 and changed_score == 0:
            # 在 og 下相关，但在 changed 下不相关 - HARD NEGATIVE!
            hard_negative_docs.append(doc_id)
        elif og_score == 0 and changed_score == 0:
            # 在两种查询下都不相关 - Easy negative
            easy_negative_docs.append(doc_id)
    
    return positive_docs, hard_negative_docs, easy_negative_docs


def build_training_data_with_instruction(
    data: Dict[str, Any], 
    n_samples: int = 50
) -> List[Dict]:
    """
    有指令版本：instruction 不为空
    - 正例：changed query 中 score >= 1 的文档
    - 负例：难负样本（og 中 score >= 1，changed 中 score = 0）
    """
    training_data = []
    corpus = data['corpus']
    queries = data['queries']
    instructions = data['instructions']
    qrels = data['qrels']
    
    # 筛选有足够难负样本的 query
    valid_queries = []
    for base_qid, qrel_data in qrels.items():
        pos_docs, hard_neg_docs, easy_neg_docs = identify_hard_negatives(qrel_data)
        
        if len(hard_neg_docs) >= MIN_HARD_NEGATIVES and len(pos_docs) >= MIN_POSITIVES:
            valid_queries.append({
                'base_qid': base_qid,
                'positive_docs': pos_docs,
                'hard_negative_docs': hard_neg_docs,
                'easy_negative_docs': easy_neg_docs
            })
    
    print(f"   找到 {len(valid_queries)} 个有足够难负样本的 query")
    
    if len(valid_queries) < n_samples:
        print(f"   ⚠️ 警告: 只有 {len(valid_queries)} 个有效 query，少于请求的 {n_samples}")
        n_samples = len(valid_queries)
    
    # 随机采样
    selected = random.sample(valid_queries, n_samples)
    
    for item in selected:
        base_qid = item['base_qid']
        og_qid = f"{base_qid}-og"
        changed_qid = f"{base_qid}-changed"
        
        q_text = queries.get(og_qid, '')
        inst = instructions.get(changed_qid, '')
        
        if not q_text or not inst:
            continue
        
        # 获取正样本文档
        pos_doc_ids = item['positive_docs']
        pos_texts = []
        for doc_id in pos_doc_ids[:5]:  # 最多取5个正样本
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                pos_texts.append(doc['text'])
        
        if not pos_texts:
            continue
        
        # 获取难负样本
        hard_neg_doc_ids = item['hard_negative_docs']
        hard_neg_texts = []
        for doc_id in hard_neg_doc_ids[:10]:  # 最多取10个难负样本
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                hard_neg_texts.append(doc['text'])
        
        if len(hard_neg_texts) < MIN_HARD_NEGATIVES:
            continue
        
        full_query = f"{q_text} {inst}"
        
        training_data.append({
            'query': full_query,
            'instruction': inst,
            'pos': pos_texts,
            'neg': hard_neg_texts,
            'dataset': data['name'],
            'type': 'with_instruction',
            'base_qid': base_qid,
            'stats': {
                'num_positives': len(pos_texts),
                'num_hard_negatives': len(hard_neg_texts),
                'num_easy_negatives': len(item['easy_negative_docs'])
            }
        })
    
    return training_data


def build_training_data_without_instruction(
    data: Dict[str, Any], 
    n_samples: int = 50
) -> List[Dict]:
    """
    无指令版本：instruction 为空
    - 正例：og query 中 score >= 1 的文档
    - 负例：og 查询中 score = 0 的文档
    """
    training_data = []
    corpus = data['corpus']
    queries = data['queries']
    qrels = data['qrels']
    
    # 筛选有足够正负样本的 og query
    valid_queries = []
    for base_qid, qrel_data in qrels.items():
        og_scores = qrel_data['og']
        
        pos_docs = [doc_id for doc_id, score in og_scores.items() if score >= 1]
        neg_docs = [doc_id for doc_id, score in og_scores.items() if score == 0]
        
        if len(pos_docs) >= MIN_POSITIVES and len(neg_docs) >= MIN_HARD_NEGATIVES:
            valid_queries.append({
                'base_qid': base_qid,
                'og_qid': f"{base_qid}-og",
                'positive_docs': pos_docs,
                'negative_docs': neg_docs
            })
    
    print(f"   找到 {len(valid_queries)} 个有足够正负样本的 og query")
    
    if len(valid_queries) < n_samples:
        print(f"   ⚠️ 警告: 只有 {len(valid_queries)} 个有效 query，少于请求的 {n_samples}")
        n_samples = len(valid_queries)
    
    # 随机采样
    selected = random.sample(valid_queries, n_samples)
    
    for item in selected:
        og_qid = item['og_qid']
        q_text = queries.get(og_qid, '')
        
        if not q_text:
            continue
        
        # 获取正样本
        pos_doc_ids = item['positive_docs']
        pos_texts = []
        for doc_id in pos_doc_ids[:5]:
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                pos_texts.append(doc['text'])
        
        if not pos_texts:
            continue
        
        # 获取负样本
        neg_doc_ids = item['negative_docs']
        random.shuffle(neg_doc_ids)
        neg_texts = []
        for doc_id in neg_doc_ids[:10]:
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                neg_texts.append(doc['text'])
        
        if len(neg_texts) < MIN_HARD_NEGATIVES:
            continue
        
        training_data.append({
            'query': q_text,
            'instruction': '',
            'pos': pos_texts,
            'neg': neg_texts,
            'dataset': data['name'],
            'type': 'without_instruction',
            'base_qid': item['base_qid'],
            'stats': {
                'num_positives': len(pos_texts),
                'num_negatives': len(neg_texts)
            }
        })
    
    return training_data


def analyze_hard_negatives(data: Dict[str, Any]) -> Dict[str, Any]:
    """分析难负样本的统计信息"""
    qrels = data['qrels']
    
    stats = {
        'total_queries': len(qrels),
        'queries_with_hard_negs': 0,
        'total_hard_negs': 0,
        'avg_hard_negs_per_query': 0,
        'queries_with_positives': 0,
        'valid_queries': 0  # 既有正样本又有足够难负样本的 query
    }
    
    for base_qid, qrel_data in qrels.items():
        pos_docs, hard_neg_docs, easy_neg_docs = identify_hard_negatives(qrel_data)
        
        if hard_neg_docs:
            stats['queries_with_hard_negs'] += 1
            stats['total_hard_negs'] += len(hard_neg_docs)
        
        if pos_docs:
            stats['queries_with_positives'] += 1
        
        if len(hard_neg_docs) >= MIN_HARD_NEGATIVES and len(pos_docs) >= MIN_POSITIVES:
            stats['valid_queries'] += 1
    
    if stats['queries_with_hard_negs'] > 0:
        stats['avg_hard_negs_per_query'] = stats['total_hard_negs'] / stats['queries_with_hard_negs']
    
    return stats


def main():
    print("=" * 70)
    print("从测试集采样转换为 ColBERT 训练格式 (改进版 v2)")
    print("难负样本定义: og score >= 1, changed score = 0")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_training_data = []
    all_stats = {}
    
    for ds in DATASETS:
        task_name = ds['task']
        name = ds['name']
        print(f"\n{'='*60}")
        print(f"处理数据集: {name}")
        print("=" * 60)
        
        data = load_followir_data(task_name)
        
        # 分析难负样本
        stats = analyze_hard_negatives(data)
        all_stats[name] = stats
        
        print(f"\n📊 难负样本统计:")
        print(f"   总 query 数: {stats['total_queries']}")
        print(f"   有难负样本的 query: {stats['queries_with_hard_negs']}")
        print(f"   总难负样本数: {stats['total_hard_negs']}")
        print(f"   平均每 query 难负样本: {stats['avg_hard_negs_per_query']:.1f}")
        print(f"   有效 query (可用于训练): {stats['valid_queries']}")
        
        # 构建训练数据
        with_inst = build_training_data_with_instruction(data, n_samples=SAMPLES_PER_DATASET // 2)
        without_inst = build_training_data_without_instruction(data, n_samples=SAMPLES_PER_DATASET // 2)
        
        print(f"\n✅ 构建完成:")
        print(f"   有指令样本: {len(with_inst)}")
        print(f"   无指令样本: {len(without_inst)}")
        
        dataset_data = with_inst + without_inst
        random.shuffle(dataset_data)
        
        all_training_data.extend(dataset_data)
        
        # 保存单个数据集
        output_file = os.path.join(OUTPUT_DIR, f"train_{name}_mixed_v2.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"   ✅ 已保存: {output_file}")
    
    # 保存混合数据
    mixed_output = os.path.join(OUTPUT_DIR, "train_overfit_mixed_instructions_v2.jsonl")
    random.shuffle(all_training_data)
    with open(mixed_output, 'w', encoding='utf-8') as f:
        for item in all_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存统计信息
    stats_output = os.path.join(OUTPUT_DIR, "hard_negative_stats_v2.json")
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"总计: {len(all_training_data)} 训练样本")
    print(f"✅ 混合数据已保存: {mixed_output}")
    print(f"✅ 统计信息已保存: {stats_output}")
    print("=" * 60)
    
    with_count = sum(1 for item in all_training_data if item['type'] == 'with_instruction')
    without_count = sum(1 for item in all_training_data if item['type'] == 'without_instruction')
    print(f"  - 有指令: {with_count}")
    print(f"  - 无指令: {without_count}")
    
    # 打印每个数据集的统计
    print(f"\n📊 各数据集统计:")
    for name, stats in all_stats.items():
        print(f"  {name}:")
        print(f"    - 总 query: {stats['total_queries']}")
        print(f"    - 有难负样本: {stats['queries_with_hard_negs']}")
        print(f"    - 有效 query: {stats['valid_queries']}")


if __name__ == "__main__":
    main()
