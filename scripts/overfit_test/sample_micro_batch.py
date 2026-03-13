#!/usr/bin/env python3
"""
Upper-Bound Overfitting Test (Oracle Test) - Data Sampling Script
从 FollowIR 测试集采样高难度微批次数据用于过拟合测试

使用方法:
    conda activate pylate
    python scripts/overfit_test/sample_micro_batch.py
"""

import os
import sys

# 激活 conda 环境
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env != 'pylate':
    import subprocess
    conda_path = subprocess.run(['which', 'conda'], capture_output=True, text=True).stdout.strip()
    if conda_path:
        env_python = f"{os.path.dirname(conda_path)}/envs/pylate/bin/python"
        if os.path.exists(env_python):
            print(f"切换到 pylate 环境: {env_python}")
            os.execv(env_python, [sys.executable] + sys.argv)

import json
import random
import numpy as np
from typing import List, Dict, Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATASET_CONFIGS = [
    {"name": "core17", "config": "core17"},
    {"name": "news21", "config": "news21"},
    {"name": "robust04", "config": "robust04"},
]

NEGATIVE_KEYWORDS = [
    "exclude", "not", "without", "except", "avoid", "don't", "do not",
    "never", "neither", "nor", "instead", "rather than", "unlike",
]

OUTPUT_PATH = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_micro_batch.json"

random.seed(42)
np.random.seed(42)


def load_followir_data(config_name: str) -> Dict[str, Any]:
    """从 Hugging Face 加载 FollowIR 数据集"""
    import datasets
    
    print(f"📂 加载 FollowIR {config_name} 数据集...")
    
    try:
        ds_corpus = datasets.load_dataset("TREC-DL/FollowIR-2024", config_name, split="corpus", trust_remote_code=True)
        ds_queries = datasets.load_dataset("TREC-DL/FollowIR-2024", config_name, split="queries", trust_remote_code=True)
        ds_instructions = datasets.load_dataset("TREC-DL/FollowIR-2024", config_name, split="instruction", trust_remote_code=True)
        ds_top = datasets.load_dataset("TREC-DL/FollowIR-2024", config_name, split="test", trust_remote_code=True)
        
        print(f"   加载完成: {len(ds_corpus)} docs, {len(ds_queries)} queries")
        
    except Exception as e:
        print(f"   ⚠️ 加载出错: {e}")
        raise
    
    corpus_dict = {}
    for item in ds_corpus:
        doc_id = str(item.get('_id', item.get('id', '')))
        text = item.get('text', '')
        title = item.get('title', '')
        corpus_dict[doc_id] = {
            'id': doc_id,
            'text': f"{title} {text}".strip() if title else text
        }
    
    query_dict = {}
    instruction_dict = {}
    
    for item in ds_queries:
        qid = str(item.get('_id', item.get('id', '')))
        query_text = item.get('text', '')
        query_dict[qid] = query_text
    
    for item in ds_instructions:
        qid = str(item.get('query-id', ''))
        inst_text = item.get('instruction', '')
        instruction_dict[qid] = inst_text
    
    candidates = {}
    for item in ds_top:
        qid = str(item.get('query-id', item.get('qid', '')))
        corpus_ids = item.get('corpus-ids', [])
        if corpus_ids:
            candidates[qid] = [str(cid) for cid in corpus_ids]
    
    print(f"   候选文档数: {len(candidates)} queries")
    
    return {
        'corpus': corpus_dict,
        'queries': query_dict,
        'instructions': instruction_dict,
        'candidates': candidates,
        'config': config_name
    }


def check_negative_constraint(query_text: str, instruction_text: str) -> bool:
    """检查查询是否包含负面约束关键词"""
    combined = (query_text + " " + instruction_text).lower()
    return any(kw in combined for kw in NEGATIVE_KEYWORDS)


def build_triplets(data: Dict[str, Any], queries_with_neg: List[str], queries_other: List[str], n_neg: int = 25, n_other: int = 25) -> List[Dict]:
    """构建三元组: (query+instruction, positive_doc, hard_negative_doc)"""
    
    triplets = []
    corpus = data['corpus']
    queries = data['queries']
    instructions = data['instructions']
    candidates = data['candidates']
    
    def get_full_query(qid):
        base_qid = qid.replace('-og', '').replace('-changed', '')
        q_text = queries.get(base_qid, '')
        inst = instructions.get(base_qid, '')
        return f"{q_text} {inst}".strip() if inst else q_text
    
    for qid in queries_with_neg[:n_neg]:
        full_query = get_full_query(qid)
        if not full_query:
            continue
        
        cand_ids = candidates.get(qid, [])
        if len(cand_ids) < 2:
            continue
        
        pos_doc_id = cand_ids[0]
        pos_doc = corpus.get(pos_doc_id, {})
        
        neg_doc_id = cand_ids[1] if len(cand_ids) > 1 else cand_ids[-1]
        neg_doc = corpus.get(neg_doc_id, {})
        
        if pos_doc.get('text') and neg_doc.get('text'):
            triplets.append({
                'query': full_query,
                'positive': pos_doc['text'],
                'negative': neg_doc['text'],
                'positive_id': pos_doc_id,
                'negative_id': neg_doc_id,
                'has_constraint': True,
                'dataset': data['config']
            })
    
    for qid in queries_other[:n_other]:
        full_query = get_full_query(qid)
        if not full_query:
            continue
        
        cand_ids = candidates.get(qid, [])
        if len(cand_ids) < 2:
            continue
        
        pos_doc_id = cand_ids[0]
        pos_doc = corpus.get(pos_doc_id, {})
        
        neg_doc_id = cand_ids[1] if len(cand_ids) > 1 else cand_ids[-1]
        neg_doc = corpus.get(neg_doc_id, {})
        
        if pos_doc.get('text') and neg_doc.get('text'):
            triplets.append({
                'query': full_query,
                'positive': pos_doc['text'],
                'negative': neg_doc['text'],
                'positive_id': pos_doc_id,
                'negative_id': neg_doc_id,
                'has_constraint': False,
                'dataset': data['config']
            })
    
    return triplets


def main():
    print("=" * 70)
    print("Upper-Bound Overfitting Test - Data Sampling")
    print("=" * 70)
    
    all_triplets = []
    
    for dataset_cfg in DATASET_CONFIGS:
        config_name = dataset_cfg['config']
        print(f"\n{'='*60}")
        print(f"处理数据集: {config_name}")
        print("=" * 60)
        
        data = load_followir_data(config_name)
        
        queries_list = list(data['queries'].keys())
        
        queries_with_neg = []
        queries_other = []
        
        for qid in queries_list:
            base_qid = qid.replace('-og', '').replace('-changed', '')
            q_text = data['queries'].get(qid, '')
            inst = data['instructions'].get(base_qid, '')
            
            if check_negative_constraint(q_text, inst):
                queries_with_neg.append(base_qid)
            else:
                queries_other.append(base_qid)
        
        print(f"   包含负面约束的查询: {len(queries_with_neg)}")
        print(f"   其他查询: {len(queries_other)}")
        
        random.shuffle(queries_with_neg)
        random.shuffle(queries_other)
        
        n_neg = min(25, len(queries_with_neg))
        n_other = min(25, len(queries_other))
        
        print(f"   采样: {n_neg} 负面约束 + {n_other} 其他 = {n_neg + n_other}")
        
        triplets = build_triplets(data, queries_with_neg, queries_other, n_neg, n_other)
        print(f"   成功构建三元组: {len(triplets)}")
        
        all_triplets.extend(triplets)
    
    print(f"\n{'='*60}")
    print(f"总计: {len(all_triplets)} 三元组")
    print("=" * 60)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_triplets, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 数据已保存至: {OUTPUT_PATH}")
    
    stats = {
        'total': len(all_triplets),
        'has_constraint': sum(1 for t in all_triplets if t.get('has_constraint')),
        'datasets': {}
    }
    
    for t in all_triplets:
        ds = t.get('dataset', 'unknown')
        if ds not in stats['datasets']:
            stats['datasets'][ds] = 0
        stats['datasets'][ds] += 1
    
    print(f"\n统计信息:")
    print(f"  总数: {stats['total']}")
    print(f"  包含约束: {stats['has_constraint']}")
    print(f"  按数据集: {stats['datasets']}")
    
    print("\n示例数据:")
    if all_triplets:
        sample = all_triplets[0]
        print(f"  Query: {sample['query'][:100]}...")
        print(f"  Positive: {sample['positive'][:100]}...")
        print(f"  Negative: {sample['negative'][:100]}...")


if __name__ == "__main__":
    main()
