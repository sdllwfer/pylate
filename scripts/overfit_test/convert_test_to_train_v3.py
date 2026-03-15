#!/usr/bin/env python3
"""
从 FollowIR 测试集采样并转换为 ColBERT 训练格式 (v3 - 优化版)
改进：
1. 增加负样本数量，保持正负比例在 1:5 到 1:10
2. 避免跨样本重复文档
3. 更严格的难负样本筛选
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
SAMPLES_PER_DATASET = 100
MIN_HARD_NEGATIVES = 5  # 最少难负样本数
MIN_POSITIVES = 1
TARGET_POS_RATIO = 5  # 目标正负比例 1:5
MAX_POS_RATIO = 10    # 最大正负比例 1:10

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
    
    corpus_dict = {}
    for item in ds_corpus['corpus']:
        doc_id = str(item.get('_id', item.get('id', '')))
        text = item.get('text', '')
        title = item.get('title', '')
        corpus_dict[doc_id] = {
            'id': doc_id,
            'text': f"{title} {text}".strip() if title else text
        }
    
    query_dict = {}
    for item in ds_queries['queries']:
        qid = str(item.get('_id', item.get('id', '')))
        query_dict[qid] = item.get('text', '')
    
    instruction_dict = {}
    for item in ds_instructions['instruction']:
        qid = str(item.get('query-id', ''))
        instruction_dict[qid] = item.get('instruction', '')
    
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
    """识别难负样本"""
    og_scores = qrels['og']
    changed_scores = qrels['changed']
    
    positive_docs = []
    hard_negative_docs = []
    easy_negative_docs = []
    
    all_docs = set(og_scores.keys()) | set(changed_scores.keys())
    
    for doc_id in all_docs:
        og_score = og_scores.get(doc_id, 0)
        changed_score = changed_scores.get(doc_id, 0)
        
        if changed_score >= 1:
            positive_docs.append(doc_id)
        elif og_score >= 1 and changed_score == 0:
            hard_negative_docs.append(doc_id)
        elif og_score == 0 and changed_score == 0:
            easy_negative_docs.append(doc_id)
    
    return positive_docs, hard_negative_docs, easy_negative_docs


def build_training_data_with_instruction(
    data: Dict[str, Any], 
    n_samples: int = 50,
    used_doc_ids: set = None
) -> Tuple[List[Dict], set]:
    """有指令版本"""
    if used_doc_ids is None:
        used_doc_ids = set()
    
    training_data = []
    corpus = data['corpus']
    queries = data['queries']
    instructions = data['instructions']
    qrels = data['qrels']
    
    valid_queries = []
    for base_qid, qrel_data in qrels.items():
        pos_docs, hard_neg_docs, easy_neg_docs = identify_hard_negatives(qrel_data)
        
        # 过滤掉已使用的文档
        available_pos = [d for d in pos_docs if d not in used_doc_ids]
        available_hard_neg = [d for d in hard_neg_docs if d not in used_doc_ids]
        available_easy_neg = [d for d in easy_neg_docs if d not in used_doc_ids]
        
        total_available_neg = len(available_hard_neg) + len(available_easy_neg)
        
        if len(available_hard_neg) >= MIN_HARD_NEGATIVES and len(available_pos) >= MIN_POSITIVES:
            valid_queries.append({
                'base_qid': base_qid,
                'positive_docs': available_pos,
                'hard_negative_docs': available_hard_neg,
                'easy_negative_docs': available_easy_neg,
                'total_negatives': total_available_neg
            })
    
    print(f"   找到 {len(valid_queries)} 个有效 query")
    
    if len(valid_queries) < n_samples:
        n_samples = len(valid_queries)
    
    # 按难负样本数量排序，优先选择难负样本多的
    valid_queries.sort(key=lambda x: len(x['hard_negative_docs']), reverse=True)
    selected = valid_queries[:n_samples]
    random.shuffle(selected)
    
    for item in selected:
        base_qid = item['base_qid']
        og_qid = f"{base_qid}-og"
        changed_qid = f"{base_qid}-changed"
        
        q_text = queries.get(og_qid, '')
        inst = instructions.get(changed_qid, '')
        
        if not q_text or not inst:
            continue
        
        # 获取正样本（最多5个）
        pos_doc_ids = item['positive_docs'][:5]
        pos_texts = []
        for doc_id in pos_doc_ids:
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                pos_texts.append(doc['text'])
                used_doc_ids.add(doc_id)
        
        if not pos_texts:
            continue
        
        # 计算需要的负样本数量
        n_pos = len(pos_texts)
        n_neg_needed = min(max(n_pos * TARGET_POS_RATIO, MIN_HARD_NEGATIVES * 2), n_pos * MAX_POS_RATIO)
        n_neg_needed = int(n_neg_needed)
        
        # 优先使用难负样本
        neg_texts = []
        neg_doc_ids = item['hard_negative_docs'][:n_neg_needed]
        for doc_id in neg_doc_ids:
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                neg_texts.append(doc['text'])
                used_doc_ids.add(doc_id)
        
        # 如果难负样本不够，补充易负样本
        if len(neg_texts) < n_neg_needed:
            remaining = n_neg_needed - len(neg_texts)
            easy_neg_ids = item['easy_negative_docs'][:remaining]
            for doc_id in easy_neg_ids:
                doc = corpus.get(doc_id, {})
                if doc.get('text'):
                    neg_texts.append(doc['text'])
                    used_doc_ids.add(doc_id)
        
        if len(neg_texts) < MIN_HARD_NEGATIVES:
            continue
        
        full_query = f"{q_text} {inst}"
        
        training_data.append({
            'query': full_query,
            'instruction': inst,
            'pos': pos_texts,
            'neg': neg_texts,
            'dataset': data['name'],
            'type': 'with_instruction',
            'base_qid': base_qid,
            'stats': {
                'num_positives': len(pos_texts),
                'num_hard_negatives': len([d for d in neg_doc_ids if d in item['hard_negative_docs']]),
                'num_easy_negatives': len([d for d in neg_doc_ids if d in item['easy_negative_docs']]),
                'target_neg_ratio': n_neg_needed / n_pos if n_pos > 0 else 0
            }
        })
    
    return training_data, used_doc_ids


def build_training_data_without_instruction(
    data: Dict[str, Any], 
    n_samples: int = 50,
    used_doc_ids: set = None
) -> Tuple[List[Dict], set]:
    """无指令版本"""
    if used_doc_ids is None:
        used_doc_ids = set()
    
    training_data = []
    corpus = data['corpus']
    queries = data['queries']
    qrels = data['qrels']
    
    valid_queries = []
    for base_qid, qrel_data in qrels.items():
        og_scores = qrel_data['og']
        
        pos_docs = [doc_id for doc_id, score in og_scores.items() if score >= 1 and doc_id not in used_doc_ids]
        neg_docs = [doc_id for doc_id, score in og_scores.items() if score == 0 and doc_id not in used_doc_ids]
        
        if len(pos_docs) >= MIN_POSITIVES and len(neg_docs) >= MIN_HARD_NEGATIVES * 2:
            valid_queries.append({
                'base_qid': base_qid,
                'og_qid': f"{base_qid}-og",
                'positive_docs': pos_docs,
                'negative_docs': neg_docs
            })
    
    print(f"   找到 {len(valid_queries)} 个有效 og query")
    
    if len(valid_queries) < n_samples:
        n_samples = len(valid_queries)
    
    selected = random.sample(valid_queries, n_samples)
    
    for item in selected:
        og_qid = item['og_qid']
        q_text = queries.get(og_qid, '')
        
        if not q_text:
            continue
        
        # 获取正样本
        pos_doc_ids = item['positive_docs'][:5]
        pos_texts = []
        for doc_id in pos_doc_ids:
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                pos_texts.append(doc['text'])
                used_doc_ids.add(doc_id)
        
        if not pos_texts:
            continue
        
        # 计算需要的负样本数量
        n_pos = len(pos_texts)
        n_neg_needed = min(max(n_pos * TARGET_POS_RATIO, MIN_HARD_NEGATIVES * 2), n_pos * MAX_POS_RATIO)
        n_neg_needed = int(n_neg_needed)
        
        # 随机选择负样本
        neg_doc_ids = item['negative_docs'][:n_neg_needed]
        random.shuffle(neg_doc_ids)
        neg_doc_ids = neg_doc_ids[:n_neg_needed]
        
        neg_texts = []
        for doc_id in neg_doc_ids:
            doc = corpus.get(doc_id, {})
            if doc.get('text'):
                neg_texts.append(doc['text'])
                used_doc_ids.add(doc_id)
        
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
                'num_negatives': len(neg_texts),
                'target_neg_ratio': n_neg_needed / n_pos if n_pos > 0 else 0
            }
        })
    
    return training_data, used_doc_ids


def main():
    print("=" * 70)
    print("从测试集采样转换为 ColBERT 训练格式 (优化版 v3)")
    print("改进:")
    print("  - 增加负样本数量 (目标比例 1:5 到 1:10)")
    print("  - 避免跨样本重复文档")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_training_data = []
    all_stats = {}
    global_used_doc_ids = set()  # 全局已使用文档集合
    
    for ds in DATASETS:
        task_name = ds['task']
        name = ds['name']
        print(f"\n{'='*60}")
        print(f"处理数据集: {name}")
        print("=" * 60)
        
        data = load_followir_data(task_name)
        
        # 构建训练数据（传递已使用文档集合）
        with_inst, used_docs_1 = build_training_data_with_instruction(
            data, 
            n_samples=SAMPLES_PER_DATASET // 2,
            used_doc_ids=global_used_doc_ids.copy()
        )
        
        # 更新全局已使用文档集合
        global_used_doc_ids.update(used_docs_1)
        
        without_inst, used_docs_2 = build_training_data_without_instruction(
            data, 
            n_samples=SAMPLES_PER_DATASET // 2,
            used_doc_ids=global_used_doc_ids.copy()
        )
        
        global_used_doc_ids.update(used_docs_2)
        
        print(f"\n✅ 构建完成:")
        print(f"   有指令样本: {len(with_inst)}")
        print(f"   无指令样本: {len(without_inst)}")
        
        dataset_data = with_inst + without_inst
        random.shuffle(dataset_data)
        
        all_training_data.extend(dataset_data)
        
        # 保存单个数据集
        output_file = os.path.join(OUTPUT_DIR, f"train_{name}_mixed_v3.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"   ✅ 已保存: {output_file}")
    
    # 保存混合数据
    mixed_output = os.path.join(OUTPUT_DIR, "train_overfit_mixed_instructions_v3.jsonl")
    random.shuffle(all_training_data)
    with open(mixed_output, 'w', encoding='utf-8') as f:
        for item in all_training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"总计: {len(all_training_data)} 训练样本")
    print(f"✅ 混合数据已保存: {mixed_output}")
    print("=" * 60)
    
    with_count = sum(1 for item in all_training_data if item['type'] == 'with_instruction')
    without_count = sum(1 for item in all_training_data if item['type'] == 'without_instruction')
    print(f"  - 有指令: {with_count}")
    print(f"  - 无指令: {without_count}")
    print(f"  - 唯一文档数: {len(global_used_doc_ids)}")


if __name__ == "__main__":
    main()
