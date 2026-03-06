"""
FollowIR 完整评测脚本
计算所有 FollowIR 指标：p-MRR + 原始/改变指令的检索指标
支持评估 Core17、Robust04、News21 三个数据集
支持生成每个查询的 p-MRR 诊断报告
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
from collections import defaultdict
from datetime import datetime

import mteb
from datasets import load_dataset
from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change

FOLLOWIR_TASKS = {
    "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
    "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
    "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
}


def get_rank_from_dict(rank_dict, doc_id):
    """从排名字典中获取文档的排名"""
    if doc_id not in rank_dict:
        return -1, None
    
    sorted_docs = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (did, score) in enumerate(sorted_docs, start=1):
        if did == doc_id:
            return rank, score
    return -1, None


def rank_score(og_rank, new_rank):
    """计算单个文档的 rank score"""
    if og_rank <= 0 or new_rank <= 0:
        return 0.0
    
    if og_rank >= new_rank:
        result = (1 / og_rank) / (1 / new_rank) - 1
    else:
        result = 1 - ((1 / new_rank) / (1 / og_rank))
    
    return result


def calculate_query_pmrr(results_og, results_changed, changed_qrels, queries_dict=None):
    """计算每个查询的 p-MRR 并生成诊断报告"""
    qid_pmrr = {}
    qid_details = {}
    
    for qid in changed_qrels.keys():
        og_key = qid + '-og'
        changed_key = qid + '-changed'
        
        if og_key not in results_og or changed_key not in results_changed:
            continue
        
        original_qid_run = results_og[og_key]
        new_qid_run = results_changed[changed_key]
        
        query_scores = []
        for changed_doc in changed_qrels[qid]:
            original_rank, original_score = get_rank_from_dict(original_qid_run, changed_doc)
            new_rank, new_score = get_rank_from_dict(new_qid_run, changed_doc)
            
            if original_rank < 0 or new_rank < 0:
                continue
            
            score = rank_score(original_rank, new_rank)
            query_scores.append({
                'doc_id': changed_doc,
                'original_rank': original_rank,
                'new_rank': new_rank,
                'score': score
            })
        
        if query_scores:
            qid_pmrr[qid] = sum(s['score'] for s in query_scores) / len(query_scores)
            qid_details[qid] = query_scores
    
    return qid_pmrr, qid_details


def generate_diagnostic_report(qid_pmrr, output_path, queries_dict=None):
    """生成诊断报告，按 p-MRR 从低到高排序"""
    sorted_qids = sorted(qid_pmrr.items(), key=lambda x: x[1])
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("📋 FollowIR 诊断报告 - 按 p-MRR 排序 (低到高)")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"{'排名':<6} {'查询ID':<12} {'p-MRR':<12} {'查询内容 (前100字符)':<80}")
    report_lines.append("-" * 110)
    
    for rank, (qid, pmrr) in enumerate(sorted_qids, start=1):
        query_text = ""
        if queries_dict and qid in queries_dict:
            query_info = queries_dict[qid]
            query_text = query_info.get('full_text', query_info.get('query', ''))[:100]
        
        report_lines.append(f"{rank:<6} {qid:<12} {pmrr:<12.6f} {query_text:<80}")
    
    report_lines.append("")
    report_lines.append("-" * 110)
    report_lines.append(f"{'总计':<6} {len(qid_pmrr):<12} 平均: {sum(qid_pmrr.values())/len(qid_pmrr):.6f}")
    report_lines.append("=" * 100)
    
    report_text = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\n📋 诊断报告已保存至: {output_path}")
    print("\n" + report_text)
    
    if queries_dict:
        json_path = output_path.replace('.txt', '_details.json')
        json_data = {
            qid: {
                'pmrr': pmrr,
                'query': queries_dict.get(qid, {}).get('query', ''),
                'instruction': queries_dict.get(qid, {}).get('instruction', ''),
            }
            for qid, pmrr in sorted_qids
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"📋 详细JSON已保存至: {json_path}")
    
    return report_text


def load_qrels(task_path):
    """加载 qrels"""
    ds_qrels = load_dataset(task_path, 'default', trust_remote_code=True)
    q_split = 'test' if 'test' in ds_qrels else list(ds_qrels.keys())[0]
    qrels = {}
    for item in ds_qrels[q_split]:
        qid = item.get('query-id', item.get('query_id', ''))
        doc_id = str(item.get('corpus-id', item.get('doc_id', '')))
        relevance = int(item.get('score', item.get('relevance', 1)))
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = relevance
    
    print(f"✅ 加载 qrels: {len(qrels)} 个查询")
    return qrels


def load_qrel_diff(task_path):
    """加载 qrel_diff - 记录哪些文档的相关性发生了变化"""
    ds_diff = load_dataset(task_path, 'qrel_diff', trust_remote_code=True)
    diff_splits = [k for k in ds_diff.keys() if 'qrel' in k or 'diff' in k]
    d_split = diff_splits[0] if diff_splits else list(ds_diff.keys())[0]
    changed_qrels = {}
    for item in ds_diff[d_split]:
        qid = item.get('query-id', item.get('query_id', ''))
        corpus_ids = item.get('corpus-ids', item.get('results', []))
        if corpus_ids:
            changed_qrels[qid] = corpus_ids
    
    print(f"✅ 加载 qrel_diff: {len(changed_qrels)} 个查询有变化的文档")
    return changed_qrels


def load_queries_with_instructions(task_path):
    """加载查询文本和指令"""
    ds_q = load_dataset(task_path, 'queries', trust_remote_code=True)
    ds_inst = load_dataset(task_path, 'instruction', trust_remote_code=True)
    
    q_split = 'queries' if 'queries' in ds_q else list(ds_q.keys())[0]
    i_split = 'instruction' if 'instruction' in ds_inst else list(ds_inst.keys())[0]
    
    instruction_dict = {}
    for item in ds_inst[i_split]:
        qid = item.get('query-id', '')
        instruction_dict[qid] = item.get('instruction', '')
    
    queries_dict = {}
    for item in ds_q[q_split]:
        qid = item.get('_id', item.get('query-id', ''))
        query_text = item.get('text', '')
        instruction = instruction_dict.get(qid, '')
        
        queries_dict[qid] = {
            'query': query_text,
            'instruction': instruction,
            'full_text': f"{query_text} {instruction}".strip()
        }
    
    print(f"✅ 加载查询: {len(queries_dict)} 个")
    return queries_dict


def load_trec_run(run_path):
    """加载 TREC 格式的 run 文件"""
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
    
    print(f"✅ 加载 run 文件: {len(results)} 个查询")
    return results


def evaluate_single_task(task_name, task_path, run_og, run_changed, output_path=None, all_results=None, base_output_dir=None):
    """评估单个任务"""
    print(f"\n{'='*60}")
    print(f"📊 评测任务: {task_name}")
    print(f"{'='*60}")
    
    print("\n📥 加载数据...")
    qrels = load_qrels(task_path)
    changed_qrels = load_qrel_diff(task_path)
    
    print(f"\n📥 加载 run 文件...")
    results_og = load_trec_run(run_og)
    results_changed = load_trec_run(run_changed)
    
    results = {**results_og, **results_changed}
    
    print("\n🔢 计算 FollowIR 指标...")
    k_values = [1, 3, 5, 10, 100, 1000]
    
    scores = evaluate_p_mrr_change(
        qrels=qrels,
        results=results,
        changed_qrels=changed_qrels,
        k_values=k_values,
    )
    
    pmrr = scores.get('p-MRR', 0)
    print(f"\n🎯 p-MRR: {pmrr:.4f}")
    
    og_scores = scores.get('og', {})
    changed_scores = scores.get('changed', {})
    
    print(f"\n📊 原始指令 - nDCG@5: {og_scores.get('ndcg_at_5', 0):.4f}, MAP@1000: {og_scores.get('map_at_1000', 0):.4f}")
    print(f"📊 改变指令 - nDCG@5: {changed_scores.get('ndcg_at_5', 0):.4f}, MAP@1000: {changed_scores.get('map_at_1000', 0):.4f}")
    
    result = {
        "task": task_name,
        "p-MRR": pmrr,
        "original": og_scores,
        "changed": changed_scores,
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 结果已保存至: {output_path}")
    
    if base_output_dir:
        metrics_dir = os.path.join(base_output_dir, task_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "results.json")
        with open(metrics_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 指标已保存至: {metrics_path}")
        
        params_file = os.path.join(base_output_dir, "eval_params.txt")
        if not os.path.exists(params_file):
            import datetime
            with open(params_file, 'w', encoding='utf-8') as pf:
                pf.write("=" * 60 + "\n")
                pf.write("FollowIR 评估参数\n")
                pf.write("=" * 60 + "\n\n")
                if args.model_path:
                    pf.write(f"模型: {args.model_path}\n")
                pf.write(f"输出目录: {base_output_dir}\n")
                pf.write(f"评估时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if args.note:
                    pf.write(f"备注: {args.note}\n")
                pf.write("=" * 60 + "\n")
            print(f"📝 参数已保存至: {params_file}")
        
        diagnostic_dir = os.path.join(base_output_dir, "diagnostic")
        os.makedirs(diagnostic_dir, exist_ok=True)
        diagnostic_path = os.path.join(diagnostic_dir, f"diagnostic_{task_name}.txt")
        queries_dict = load_queries_with_instructions(task_path)
        qid_pmrr, qid_details = calculate_query_pmrr(results_og, results_changed, changed_qrels, queries_dict)
        generate_diagnostic_report(qid_pmrr, diagnostic_path, queries_dict)
    elif output_path:
        diagnostic_path = output_path.replace(f"results_{task_name}.json", f"diagnostic_{task_name}.txt")
        queries_dict = load_queries_with_instructions(task_path)
        qid_pmrr, qid_details = calculate_query_pmrr(results_og, results_changed, changed_qrels, queries_dict)
        generate_diagnostic_report(qid_pmrr, diagnostic_path, queries_dict)
    
    if all_results is not None:
        all_results[task_name] = result
        if output_path:
            summary_path = output_path.replace(f"results_{task_name}.json", "results_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"📝 汇总已更新: {summary_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='FollowIR 完整评测脚本')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='要评测的数据集: Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval')
    parser.add_argument('--run_og', type=str, default=None, help='原始指令的 TREC run 文件 (或目录)')
    parser.add_argument('--run_changed', type=str, default=None, help='改变指令的 TREC run 文件 (或目录)')
    parser.add_argument('--run_dir', type=str, default=None, help='包含 run 文件的目录 (会自动匹配)')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--note', type=str, default='', help='备注信息，会记录到参数文件中')
    parser.add_argument('--model_path', type=str, default='', help='模型路径，会记录到参数文件中')
    args = parser.parse_args()

    tasks_to_run = args.tasks if args.tasks else list(FOLLOWIR_TASKS.keys())
    
    print("=" * 60)
    print("📊 FollowIR 完整评测")
    print("=" * 60)
    print(f"\n将评测以下数据集: {', '.join(tasks_to_run)}")

    all_results = {}
    
    for task_name in tasks_to_run:
        if task_name not in FOLLOWIR_TASKS:
            print(f"\n⚠️ 跳过未知任务: {task_name}")
            continue
        
        task_path = FOLLOWIR_TASKS[task_name]
        
        if args.run_dir:
            run_og = os.path.join(args.run_dir, "trec", f"run_{task_name}_og.trec")
            run_changed = os.path.join(args.run_dir, "trec", f"run_{task_name}_changed.trec")
        elif args.run_og and args.run_changed:
            run_og = args.run_og
            run_changed = args.run_changed
        else:
            print(f"\n⚠️ 请提供 --run_dir 或 --run_og/--run_changed")
            break
        
        if not os.path.exists(run_og):
            print(f"\n⚠️ 文件不存在: {run_og}")
            continue
        if not os.path.exists(run_changed):
            print(f"\n⚠️ 文件不存在: {run_changed}")
            continue
        
        if args.output_dir:
            output_path = os.path.join(args.output_dir, f"results_{task_name}.json")
        else:
            output_path = None
        
        result = evaluate_single_task(task_name, task_path, run_og, run_changed, output_path, all_results, args.output_dir)
    
    if len(all_results) > 1 and args.output_dir:
        summary_path = os.path.join(args.output_dir, "results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n{'='*60}")
        print(f"💾 汇总结果已保存至: {summary_path}")
        
        print(f"\n📊 汇总 p-MRR:")
        for task_name, result in all_results.items():
            print(f"  {task_name}: {result['p-MRR']:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
