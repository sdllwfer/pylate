#!/usr/bin/env python3
"""
查看 FollowIR 测试集数据结构，找出包含相似度分数的数据
"""

import datasets
import json
import os

# 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def examine_dataset(task_name="Core17InstructionRetrieval"):
    """检查数据集结构"""
    print(f"\n{'='*80}")
    print(f"检查数据集: {task_name}")
    print(f"{'='*80}\n")
    
    # 获取任务
    import mteb
    task = mteb.get_task(task_name)
    path = task.metadata.dataset.get("path")
    print(f"数据集路径: {path}")
    
    # 加载各个 split
    print("\n1. 加载 corpus...")
    ds_corpus = datasets.load_dataset(path, 'corpus')
    print(f"   Corpus splits: {list(ds_corpus.keys())}")
    print(f"   样本数: {len(ds_corpus[list(ds_corpus.keys())[0]])}")
    print(f"   字段: {list(ds_corpus[list(ds_corpus.keys())[0]][0].keys())}")
    
    print("\n2. 加载 queries...")
    ds_queries = datasets.load_dataset(path, 'queries')
    print(f"   Queries splits: {list(ds_queries.keys())}")
    print(f"   样本数: {len(ds_queries[list(ds_queries.keys())[0]])}")
    print(f"   字段: {list(ds_queries[list(ds_queries.keys())[0]][0].keys())}")
    
    # 查看 query 示例
    queries_split = list(ds_queries.keys())[0]
    sample_queries = ds_queries[queries_split].select(range(min(3, len(ds_queries[queries_split]))))
    print(f"\n   Query 示例:")
    for i, q in enumerate(sample_queries):
        print(f"   [{i}] ID: {q.get('_id', q.get('id'))}")
        print(f"       Text: {q.get('text', '')[:100]}...")
    
    print("\n3. 加载 instruction...")
    ds_inst = datasets.load_dataset(path, 'instruction')
    print(f"   Instruction splits: {list(ds_inst.keys())}")
    print(f"   样本数: {len(ds_inst[list(ds_inst.keys())[0]])}")
    print(f"   字段: {list(ds_inst[list(ds_inst.keys())[0]][0].keys())}")
    
    # 查看 instruction 示例
    inst_split = list(ds_inst.keys())[0]
    sample_insts = ds_inst[inst_split].select(range(min(3, len(ds_inst[inst_split]))))
    print(f"\n   Instruction 示例:")
    for i, inst in enumerate(sample_insts):
        print(f"   [{i}] Query ID: {inst.get('query-id', '')}")
        print(f"       Instruction: {inst.get('instruction', '')[:100]}...")
    
    print("\n4. 加载 qrels...")
    try:
        ds_qrels = datasets.load_dataset(path, 'qrels')
        print(f"   Qrels splits: {list(ds_qrels.keys())}")
        print(f"   样本数: {len(ds_qrels[list(ds_qrels.keys())[0]])}")
        print(f"   字段: {list(ds_qrels[list(ds_qrels.keys())[0]][0].keys())}")
        
        # 查看 qrels 示例
        qrels_split = list(ds_qrels.keys())[0]
        sample_qrels = ds_qrels[qrels_split].select(range(min(3, len(ds_qrels[qrels_split]))))
        print(f"\n   Qrels 示例:")
        for i, qrel in enumerate(sample_qrels):
            print(f"   [{i}] {qrel}")
    except Exception as e:
        print(f"   无法加载 qrels: {e}")
    
    print("\n5. 加载 top_ranked...")
    try:
        ds_top = datasets.load_dataset(path, 'top_ranked')
        print(f"   Top ranked splits: {list(ds_top.keys())}")
        print(f"   样本数: {len(ds_top[list(ds_top.keys())[0]])}")
        print(f"   字段: {list(ds_top[list(ds_top.keys())[0]][0].keys())}")
        
        # 查看 top_ranked 示例
        top_split = list(ds_top.keys())[0]
        sample_top = ds_top[top_split][0]
        print(f"\n   Top ranked 示例:")
        print(f"   Query ID: {sample_top.get('query-id', sample_top.get('qid', ''))}")
        print(f"   字段: {list(sample_top.keys())}")
        
        # 检查是否有相似度分数
        if 'results' in sample_top:
            print(f"\n   Results 结构:")
            if len(sample_top['results']) > 0:
                print(f"   第一个结果: {sample_top['results'][0]}")
        if 'corpus-ids' in sample_top:
            print(f"\n   Corpus IDs (前5个): {sample_top['corpus-ids'][:5]}")
    except Exception as e:
        print(f"   无法加载 top_ranked: {e}")
    
    # 尝试加载其他可能的 splits
    print("\n6. 尝试加载其他 splits...")
    try:
        # 尝试加载默认 split
        ds_default = datasets.load_dataset(path)
        print(f"   可用 splits: {list(ds_default.keys())}")
        for split_name in ds_default.keys():
            print(f"\n   Split: {split_name}")
            print(f"   样本数: {len(ds_default[split_name])}")
            if len(ds_default[split_name]) > 0:
                print(f"   字段: {list(ds_default[split_name][0].keys())}")
                print(f"   示例: {ds_default[split_name][0]}")
    except Exception as e:
        print(f"   无法加载默认 split: {e}")

if __name__ == '__main__':
    # 检查 Core17
    examine_dataset("Core17InstructionRetrieval")
    
    # 检查 News21
    examine_dataset("News21InstructionRetrieval")
