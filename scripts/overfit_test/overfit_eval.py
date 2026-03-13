#!/usr/bin/env python3
"""
Upper-Bound Overfitting Test - Evaluation Script
在过拟合数据集上评估模型性能
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_overfit_data(data_path):
    """加载过拟合测试数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


class OverfitEvalDataset(Dataset):
    """过拟合评估数据集"""
    
    def __init__(self, data_path, tokenizer, max_query_length=32, max_doc_length=256):
        self.data = load_overfit_data(data_path)
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            'query': item['query'],
            'positive': item['positive'],
            'negative': item['negative'],
            'positive_id': item.get('positive_id', ''),
            'negative_id': item.get('negative_id', ''),
            'dataset': item.get('dataset', ''),
        }


def collate_fn_eval(batch):
    return batch


def evaluate_model(model_path, data_path, output_dir, device='cuda', batch_size=16):
    """评估模型"""
    from pylate import models, rank
    from pylate.models.igp.instruction_probe import InstructionProbe
    from pylate.models.igp.igp_adapter import IGPAdapter
    from pylate.models.igp.ratio_gate_v3 import RatioGateV3
    import json
    
    print("=" * 70)
    print("Upper-Bound Overfitting Test - Evaluation")
    print("=" * 70)
    
    # 加载模型
    print(f"\n📥 加载模型: {model_path}")
    base_model = models.ColBERT(model_name_or_path=model_path, device=device)
    
    # 加载 IGP 模块
    igp_info_path = os.path.join(model_path, "igp_info.json")
    
    if os.path.exists(igp_info_path):
        with open(igp_info_path, 'r') as f:
            igp_info = json.load(f)
        
        underlying_hidden_size = base_model[0].get_word_embedding_dimension()
        
        # 加载 Probe
        probe_path = os.path.join(model_path, "igp_probe.pt")
        if os.path.exists(probe_path):
            probe = InstructionProbe(hidden_size=underlying_hidden_size, num_heads=8, num_layers=3)
            probe.load_state_dict(torch.load(probe_path, map_location=device))
            print(f"   ✅ Probe 已加载")
        
        # 加载 Adapter
        adapter_path = os.path.join(model_path, "igp_adapter.pt")
        if os.path.exists(adapter_path):
            adapter = IGPAdapter(hidden_size=underlying_hidden_size, bottleneck_dim=128)
            adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            print(f"   ✅ Adapter 已加载")
        
        # 加载 Gate
        gate_path = os.path.join(model_path, "igp_gate.pt")
        if os.path.exists(gate_path):
            gate = RatioGateV3(hidden_size=underlying_hidden_size, max_ratio=1.0)
            gate.load_state_dict(torch.load(gate_path, map_location=device))
            print(f"   ✅ Gate 已加载")
        
        igp_modules = (probe, adapter, gate)
    else:
        print("   ⚠️ 未找到 IGP 模块配置，使用基础模型")
        igp_modules = None
    
    # 加载评估数据
    print(f"\n📂 加载评估数据: {data_path}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = OverfitEvalDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_eval)
    
    print(f"   评估样本数: {len(dataset)}")
    
    # 评估
    print("\n🔍 开始评估...")
    
    results = {
        'total': 0,
        'correct_positive': 0,
        'correct_negative': 0,
        'by_dataset': {}
    }
    
    for batch in tqdm(dataloader, desc="评估"):
        for item in batch:
            query = item['query']
            positive = item['positive']
            negative = item['negative']
            dataset_name = item['dataset']
            
            # 编码
            with torch.no_grad():
                query_emb = base_model.encode_queries([query], batch_size=1)
                pos_emb = base_model.encode_corpus([positive], batch_size=1)
                neg_emb = base_model.encode_corpus([negative], batch_size=1)
            
            # 计算相似度
            pos_score = torch.sum(query_emb * pos_emb, dim=-1).item()
            neg_score = torch.sum(query_emb * neg_emb, dim=-1).item()
            
            # 判断正确性
            is_correct_pos = pos_score > neg_score
            is_correct_neg = neg_score > pos_score
            
            results['total'] += 1
            if is_correct_pos:
                results['correct_positive'] += 1
            if is_correct_neg:
                results['correct_negative'] += 1
            
            # 按数据集统计
            if dataset_name not in results['by_dataset']:
                results['by_dataset'][dataset_name] = {'total': 0, 'correct': 0}
            results['by_dataset'][dataset_name]['total'] += 1
            if is_correct_pos:
                results['by_dataset'][dataset_name]['correct'] += 1
    
    # 计算指标
    accuracy = results['correct_positive'] / results['total'] if results['total'] > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 评估结果")
    print("=" * 70)
    print(f"总体准确率: {accuracy:.4f} ({results['correct_positive']}/{results['total']})")
    
    print("\n按数据集:")
    for ds_name, ds_stats in results['by_dataset'].items():
        ds_acc = ds_stats['correct'] / ds_stats['total'] if ds_stats['total'] > 0 else 0
        print(f"  {ds_name}: {ds_acc:.4f} ({ds_stats['correct']}/{ds_stats['total']})")
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "overfit_eval_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'correct': results['correct_positive'],
            'total': results['total'],
            'by_dataset': results['by_dataset']
        }, f, indent=2)
    
    print(f"\n✅ 结果已保存至: {output_path}")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="过拟合测试评估")
    parser.add_argument('--model_path', type=str, 
                        default="/home/luwa/Documents/pylate/output/colbert_igp_train/overfit_test")
    parser.add_argument('--data_path', type=str, 
                        default="/home/luwa/Documents/pylate/dataset/colbert_data/overfit_micro_batch.json")
    parser.add_argument('--output_dir', type=str, 
                        default="/home/luwa/Documents/pylate/evaluation_data/colbert_igp/overfit_test")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
