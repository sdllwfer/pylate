"""
ColBERT x FollowIR 评测脚本
利用 pylate.rank.rerank 进行指令重排，并输出标准评测文件
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
            os.execv(env_python, [sys.executable] + sys.argv)

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import mteb
from pylate import models, rank


def load_model(model_path: str, device: str = "cuda"):
    print(f"📥 正在加载模型: {model_path} on {device}")
    print(f"   (如果是首次运行，可能需要下载模型，请耐心等待...)")
    import time
    start = time.time()
    model = models.ColBERT(model_name_or_path=model_path, device=device)
    print(f"✅ 模型加载完成，耗时: {time.time() - start:.1f}秒")
    return model


def load_data_pairwise(task):
    """加载 FollowIR 数据，正确处理 og/changed 查询"""
    import datasets
    import time
    
    corpus, q_og, q_changed, candidates = {}, {}, {}, {}
    path = task.metadata.dataset.get("path")
    
    print(f"📂 加载数据集: {path}")
    t0 = time.time()
    
    try:
        print("   加载 corpus...")
        ds_c = datasets.load_dataset(path, 'corpus', trust_remote_code=True)
        c_split = 'corpus' if 'corpus' in ds_c else 'train'
        for d in ds_c[c_split]:
            corpus[str(d.get('_id', d.get('id')))] = {'text': str(d.get('text', ''))}
        print(f"   ✅ 加载 {len(corpus)} 个文档 ({time.time()-t0:.1f}s)")
        
        print("   加载 queries...")
        ds_q = datasets.load_dataset(path, 'queries', trust_remote_code=True)
        q_split = 'queries' if 'queries' in ds_q else 'train'
        
        print("   加载 instruction...")
        ds_inst = datasets.load_dataset(path, 'instruction', trust_remote_code=True)
        i_split = 'instruction' if 'instruction' in ds_inst else 'train'
        
        instruction_dict = {}
        for inst_item in ds_inst[i_split]:
            qid = str(inst_item.get('query-id', ''))
            inst_text = str(inst_item.get('instruction', ''))
            instruction_dict[qid] = inst_text
        
        for q in ds_q[q_split]:
            full_qid = str(q.get('_id', q.get('id', '')))
            query_text = q.get('text', '')
            inst = instruction_dict.get(full_qid, "")
            
            if full_qid.endswith('-og'):
                q_og[full_qid] = f"{query_text} {inst}".strip()
            elif full_qid.endswith('-changed'):
                q_changed[full_qid] = f"{query_text} {inst}".strip()
        
        print(f"   ✅ 加载 {len(q_og)} 个 og 查询, {len(q_changed)} 个 changed 查询 ({time.time()-t0:.1f}s)")
        
        try:
            print("   加载 top_ranked...")
            ds_top = datasets.load_dataset(path, 'top_ranked', trust_remote_code=True)
            available_splits = list(ds_top.keys())
            t_split = available_splits[0] if available_splits else None
            if t_split:
                for item in ds_top[t_split]:
                    full_qid = str(item.get('query-id', item.get('query_id', item.get('qid', ''))))
                    base_qid = full_qid.replace('-og', '').replace('-changed', '')
                    results_list = item.get('corpus-ids', item.get('results', []))
                    
                    if base_qid not in candidates:
                        candidates[base_qid] = [str(did) for did in results_list]
                
                if candidates:
                    avg_cand = sum(len(v) for v in candidates.values()) / len(candidates)
                    print(f"   ✅ 加载 {len(candidates)} 个查询的候选文档, 平均 {avg_cand:.0f} 个/查询 ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  -> ⚠️ 无法加载 top_ranked: {e}")

    except Exception as e:
        print(f"加载数据出错: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"📂 数据加载完成，总耗时: {time.time()-t0:.1f}s")
        
    return corpus, q_og, q_changed, candidates


def batch_rerank(model, queries_dict, corpus, candidates, batch_size=64):
    """对批量查询进行重排 - 优化版本"""
    results = {}
    
    print("🔧 优化重排过程...")
    
    all_doc_ids_set = set()
    for qid, doc_ids in candidates.items():
        all_doc_ids_set.update(doc_ids)
    
    all_doc_ids = list(all_doc_ids_set)
    doc_id_to_idx = {did: idx for idx, did in enumerate(all_doc_ids)}
    
    print(f"📚 编码 {len(all_doc_ids)} 个文档 (batch_size={batch_size})...")
    doc_texts = [corpus[did].get('text', '') for did in all_doc_ids]
    all_doc_emb = model.encode(
        doc_texts,
        is_query=False,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    print(f"✅ 文档编码完成，共 {len(all_doc_emb)} 个tensor")
    
    queries_list = list(queries_dict.keys())
    print(f"📝 批量编码 {len(queries_list)} 个查询 (batch_size={batch_size})...")
    query_texts = [queries_dict[qid] for qid in queries_list]
    all_query_emb = model.encode(
        query_texts,
        is_query=True,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )
    print(f"✅ 查询编码完成，共 {len(all_query_emb)} 个tensor")
    
    for idx, qid in enumerate(tqdm(queries_list, desc="Reranking")):
        base_qid = qid.replace('-og', '').replace('-changed', '')
        if base_qid not in candidates or not candidates[base_qid]:
            print(f"⚠️ 警告: 查询 {qid} (base: {base_qid}) 没有候选文档")
            continue
        
        doc_ids = candidates[base_qid]
        
        doc_indices = [doc_id_to_idx[did] for did in doc_ids if did in doc_id_to_idx]
        if not doc_indices:
            continue
        
        doc_emb_list = [all_doc_emb[i] for i in doc_indices]
        q_emb = all_query_emb[idx]
        
        documents_ids = [[str(i) for i in range(len(doc_indices))]]
        
        doc_emb_stacked = torch.nn.utils.rnn.pad_sequence(
            [emb.cpu() for emb in doc_emb_list], 
            batch_first=True, 
            padding_value=0
        ).unsqueeze(0)
        
        reranked = rank.rerank(
            documents_ids=documents_ids,
            queries_embeddings=[q_emb.cpu()],
            documents_embeddings=doc_emb_stacked,
        )
        
        q_results = {}
        for doc_info in reranked[0]:
            doc_idx = int(doc_info["id"])
            doc_id = doc_ids[doc_idx]
            score = doc_info["score"]
            q_results[doc_id] = score
        
        results[qid] = q_results
    
    return results


def save_to_trec_format(results, output_path, run_name="pylate_colbert"):
    """保存为 TREC 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for q_id, doc_scores in results.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank_idx, (doc_id, score) in enumerate(sorted_docs, start=1):
                f.write(f"{q_id} Q0 {doc_id} {rank_idx} {score:.4f} {run_name}\n")
    print(f"💾 TREC Run 文件已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ColBERT FollowIR 评测脚本')
    parser.add_argument('--model_path', type=str, default='lightonai/GTE-ModernColBERT-v1')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--task', type=str, default='Core17InstructionRetrieval')
    parser.add_argument('--device', type=str, default='cuda', help='GPU device (e.g., cuda:0, cuda:1)')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小，用于编码文档和查询')
    parser.add_argument('--note', type=str, default='', help='备注信息，会记录到参数文件中')
    args = parser.parse_args()

    if args.output_dir is None:
        base_output_dir = "/home/luwa/Documents/pylate/evaluation_data/followir"
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_dir = os.path.join(base_output_dir, timestamp)
    else:
        output_dir = args.output_dir

    print(f"📁 输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(args.model_path, device=args.device)

    print(f"📚 加载 FollowIR 评测任务: {args.task}")
    task = mteb.get_task(args.task)
    
    print(f"\n{'='*60}")
    print(f"🚀 评测任务: {task.metadata.name}")
    print(f"{'='*60}")
    
    corpus, q_og, q_changed, candidates = load_data_pairwise(task)
    
    if not corpus or not q_og:
        print("❌ 数据加载失败")
        return
    
    avg_cand = sum(len(v) for v in candidates.values()) / len(candidates) if candidates else 0
    print(f"\n📊 共 {len(q_og)} og 查询 + {len(q_changed)} changed 查询, 平均 {avg_cand:.0f} 候选文档")
    
    trec_dir = os.path.join(output_dir, "trec")
    os.makedirs(trec_dir, exist_ok=True)
    
    task_dir = os.path.join(output_dir, task.metadata.name)
    os.makedirs(task_dir, exist_ok=True)
    
    print("\n--- 开始评测: Original Instructions (og) ---")
    results_og = batch_rerank(model, q_og, corpus, candidates, batch_size=args.batch_size)
    run_og_path = os.path.join(trec_dir, f"run_{task.metadata.name}_og.trec")
    save_to_trec_format(results_og, run_og_path)
    
    print("\n--- 开始评测: Altered Instructions (changed) ---")
    results_changed = batch_rerank(model, q_changed, corpus, candidates, batch_size=args.batch_size)
    run_changed_path = os.path.join(trec_dir, f"run_{task.metadata.name}_changed.trec")
    save_to_trec_format(results_changed, run_changed_path)

    params_file = os.path.join(output_dir, "eval_params.txt")
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FollowIR 评估参数\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型: {args.model_path}\n")
        f.write(f"数据集: {task.metadata.name}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"GPU设备: {args.device}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.note:
            f.write(f"备注: {args.note}\n")
        f.write("=" * 60 + "\n")
    print(f"📝 参数已保存至: {params_file}")

    print("\n" + "=" * 60)
    print("✅ 重排完成！")
    print(f"📁 结果已保存至: {output_dir}")
    print(f"  : {trec_dir}/ - TREC 文件")
    print(f"   - {task.metadata.name}: {task_dir}/")
    print("\n要计算 FollowIR p-MRR 指标，请使用 FollowIR 官方脚本:")
    print(f"python -m followir.evaluate --qrels path/to/qrels --run_org {run_og_path} --run_alt {run_changed_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
