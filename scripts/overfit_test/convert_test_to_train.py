#!/usr/bin/env python3
"""
从 FollowIR 测试集采样并转换为 ColBERT 训练格式
混合版本：
1. 有指令样本 (50%): instruction 不为空
   - 正例：score = 2
   - 负例：难负样本（og 相关，changed 不相关）
2. 无指令样本 (50%): instruction 为空
   - 正例：score = 2
   - 负例：og 查询中 score = 2 的样本
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
from typing import List, Dict, Any

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DATASETS = [
    {"name": "core17", "task": "Core17InstructionRetrieval"},
    {"name": "news21", "task": "News21InstructionRetrieval"},
    {"name": "robust04", "task": "Robust04InstructionRetrieval"},
]

NEGATIVE_KEYWORDS = [
    "exclude", "not", "without", "except", "avoid", "don't", "do not",
    "never", "neither", "nor", "instead", "rather than", "unlike",
]

OUTPUT_DIR = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data"
SAMPLES_PER_DATASET = 100

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
    ds_qrel_diff = datasets.load_dataset(dataset_path, 'qrel_diff')
    
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
    
    qrels = {}
    for item in ds_qrels:
        qid = str(item.get('query-id', ''))
        doc_id = str(item.get('corpus-id', ''))
        score = item.get('score', 0)
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = score
    
    qrel_diff = {}
    for item in ds_qrel_diff['qrel_diff']:
        qid = str(item.get('query-id', ''))
        doc_ids = item.get('corpus-ids', [])
        qrel_diff[qid] = doc_ids
    
    print(f"   qrels: {len(qrels)} queries")
    print(f"   qrel_diff: {len(qrel_diff)} queries with changed relevance")
    
    task = task_name.replace("InstructionRetrieval", "")
    
    return {
        'corpus': corpus_dict,
        'queries': query_dict,
        'instructions': instruction_dict,
        'qrels': qrels,
        'qrel_diff': qrel_diff,
        'name': task
    }


def check_negative_constraint(query_text: str, instruction_text: str) -> bool:
    combined = (query_text + " " + instruction_text).lower()
    return any(kw in combined for kw in NEGATIVE_KEYWORDS)


def build_training_data_with_instruction(data: Dict[str, Any], n_samples: int = 50) -> List[Dict]:
    """有指令版本：instruction 不为空，负例为难负样本"""
    training_data = []
    corpus = data['corpus']
    queries = data['queries']
    instructions = data['instructions']
    qrels = data['qrels']
    qrel_diff = data['qrel_diff']
    
    all_qids = list(qrel_diff.keys())
    
    queries_with_neg = []
    queries_other = []
    
    for qid in all_qids:
        changed_qid = f"{qid}-changed"
        og_qid = f"{qid}-og"
        
        q_text = queries.get(og_qid, queries.get(changed_qid, ''))
        inst = instructions.get(changed_qid, '')
        
        if check_negative_constraint(q_text, inst):
            queries_with_neg.append(qid)
        else:
            queries_other.append(qid)
    
    random.shuffle(queries_with_neg)
    random.shuffle(queries_other)
    
    n_with_neg = min(n_samples // 2, len(queries_with_neg))
    n_other = min(n_samples - n_with_neg, len(queries_other))
    
    selected_qids = queries_with_neg[:n_with_neg] + queries_other[:n_other]
    random.shuffle(selected_qids)
    
    for qid in selected_qids:
        changed_qid = f"{qid}-changed"
        og_qid = f"{qid}-og"
        
        q_text = queries.get(og_qid, '')
        inst = instructions.get(changed_qid, '')
        
        if not q_text or not inst:
            continue
        
        qrel = qrels.get(changed_qid, {})
        if not qrel:
            continue
        
        hard_neg_doc_ids = qrel_diff.get(qid, [])
        pos_docs = [doc_id for doc_id, score in qrel.items() if score == 2]
        
        if not hard_neg_doc_ids or not pos_docs:
            continue
        
        pos_doc_id = pos_docs[0]
        pos_doc = corpus.get(pos_doc_id, {})
        if not pos_doc.get('text'):
            continue
        
        neg_texts = []
        for neg_doc_id in hard_neg_doc_ids[:10]:
            neg_doc = corpus.get(neg_doc_id, {})
            if neg_doc.get('text'):
                neg_texts.append(neg_doc['text'])
        
        if len(neg_texts) < 3:
            continue
        
        full_query = f"{q_text} {inst}"
        
        training_data.append({
            'query': full_query,
            'instruction': inst,
            'pos': [pos_doc['text']],
            'neg': neg_texts,
            'dataset': data['name'],
            'type': 'with_instruction'
        })
    
    return training_data


def build_training_data_without_instruction(data: Dict[str, Any], n_samples: int = 50) -> List[Dict]:
    """无指令版本：instruction 为空，负例为 og 中 score=2 的样本"""
    training_data = []
    corpus = data['corpus']
    queries = data['queries']
    qrels = data['qrels']
    
    # qrels 的 key 已经是完整的 "xxx-og" 或 "xxx-changed" 格式
    # 过滤出 og 查询
    og_qids = [qid for qid in qrels.keys() if '-og' in qid]
    
    selected_qids = random.sample(og_qids, min(n_samples, len(og_qids)))
    
    for og_qid in selected_qids:
        q_text = queries.get(og_qid, '')
        if not q_text:
            continue
        
        qrel = qrels.get(og_qid, {})
        if not qrel:
            continue
        
        pos_docs = [doc_id for doc_id, score in qrel.items() if score == 2]
        
        if not pos_docs:
            continue
        
        pos_doc_id = pos_docs[0]
        pos_doc = corpus.get(pos_doc_id, {})
        if not pos_doc.get('text'):
            continue
        
        neg_docs = [doc_id for doc_id, score in qrel.items() if score == 0]
        
        if len(neg_docs) < 3:
            continue
        
        random.shuffle(neg_docs)
        neg_doc_ids = neg_docs[:10]
        
        neg_texts = []
        for neg_doc_id in neg_doc_ids:
            neg_doc = corpus.get(neg_doc_id, {})
            if neg_doc.get('text'):
                neg_texts.append(neg_doc['text'])
        
        if len(neg_texts) < 3:
            continue
        
        training_data.append({
            'query': q_text,
            'instruction': '',
            'pos': [pos_doc['text']],
            'neg': neg_texts,
            'dataset': data['name'],
            'type': 'without_instruction'
        })
    
    return training_data


def main():
    print("=" * 70)
    print("从测试集采样转换为 ColBERT 训练格式")
    print("混合版本：")
    print("  1. 有指令样本：instruction 不为空，负例为难负样本")
    print("  2. 无指令样本：instruction 为空，负例为 og 中 score=2 的样本")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_training_data = []
    
    for ds in DATASETS:
        task_name = ds['task']
        name = ds['name']
        print(f"\n{'='*60}")
        print(f"处理数据集: {name}")
        print("=" * 60)
        
        data = load_followir_data(task_name)
        
        with_inst = build_training_data_with_instruction(data, n_samples=SAMPLES_PER_DATASET)
        without_inst = build_training_data_without_instruction(data, n_samples=SAMPLES_PER_DATASET)
        
        print(f"   有指令样本: {len(with_inst)}")
        print(f"   无指令样本: {len(without_inst)}")
        
        dataset_data = with_inst + without_inst
        random.shuffle(dataset_data)
        
        all_training_data.extend(dataset_data)
        
        output_file = os.path.join(OUTPUT_DIR, f"train_{name}_mixed.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"   ✅ 已保存: {output_file}")
    
    mixed_output = os.path.join(OUTPUT_DIR, "train_overfit_mixed_instructions.jsonl")
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


if __name__ == "__main__":
    main()
