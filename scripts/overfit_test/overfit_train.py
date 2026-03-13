#!/usr/bin/env python3
"""
Upper-Bound Overfitting Test (Oracle Test) - Training Script
过拟合训练脚本，用于验证 Probe 和 Gate 模块的理论容量

使用方法:
    conda activate pylate
    python scripts/overfit_test/overfit_train.py
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

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

# ================= 配置参数 =================
MODEL_PATH = "/home/luwa/Documents/pylate/output/colbert_igp_train/col_two_stage_short_then_long_v2/stage2_long_data/final_model_20260308_144406"
DATA_PATH = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_micro_batch.json"
OUTPUT_DIR = "/home/luwa/Documents/pylate/output/colbert_igp_train/overfit_test"

# 过拟合训练参数
OVERFIT_EPOCHS = 100
OVERFIT_BATCH_SIZE = 8
OVERFIT_LR = 5e-4  # 基础学习率，会对 IGP 参数乘以 4
OVERFIT_LR_MULTIPLIER = 4.0  # IGP 参数的学习率倍数
WEIGHT_DECAY = 0.0  # 禁用权重衰减
DROPOUT = 0.0  # 禁用 dropout
MAX_RATIO = 1.0  # 放松门控上限到 1.0
GRADIENT_LOG_STEPS = 10  # 每多少步打印一次梯度

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ================= 数据加载 =================

class OverfitDataset(Dataset):
    """过拟合测试数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_query_length=32, max_doc_length=256):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        
        print(f"📊 加载过拟合数据: {len(self.data)} 样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        query = item['query']
        positive = item['positive']
        negative = item['negative']
        
        # Tokenize query
        query_enc = self.tokenizer(
            query,
            max_length=self.max_query_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize positive document
        pos_enc = self.tokenizer(
            positive,
            max_length=self.max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize negative document
        neg_enc = self.tokenizer(
            negative,
            max_length=self.max_doc_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_enc['input_ids'].squeeze(0),
            'query_attention_mask': query_enc['attention_mask'].squeeze(0),
            'positive_input_ids': pos_enc['input_ids'].squeeze(0),
            'positive_attention_mask': pos_enc['attention_mask'].squeeze(0),
            'negative_input_ids': neg_enc['input_ids'].squeeze(0),
            'negative_attention_mask': neg_enc['attention_mask'].squeeze(0),
        }


def collate_fn(batch):
    """自定义批处理函数"""
    return {
        'query_input_ids': torch.stack([x['query_input_ids'] for x in batch]),
        'query_attention_mask': torch.stack([x['query_attention_mask'] for x in batch]),
        'positive_input_ids': torch.stack([x['positive_input_ids'] for x in batch]),
        'positive_attention_mask': torch.stack([x['positive_attention_mask'] for x in batch]),
        'negative_input_ids': torch.stack([x['negative_input_ids'] for x in batch]),
        'negative_attention_mask': torch.stack([x['negative_attention_mask'] for x in batch]),
    }


# ================= 模型加载 =================

def load_igp_model(model_path: str, device: str = "cuda", max_ratio: float = 1.0):
    """加载支持 IGP 的 ColBERT 模型"""
    from pylate import models
    from pylate.models.igp.instruction_probe import InstructionProbe
    from pylate.models.igp.igp_adapter import IGPAdapter
    from pylate.models.igp.ratio_gate_v3 import RatioGateV3
    
    print(f"📥 加载基础模型: {model_path}")
    
    base_model = models.ColBERT(model_name_or_path=model_path, device=device)
    
    underlying_hidden_size = base_model[0].get_word_embedding_dimension()
    
    print(f"   底层 hidden_size: {underlying_hidden_size}")
    
    # 初始化 IGP 模块
    probe = InstructionProbe(
        hidden_size=underlying_hidden_size,
        num_heads=8,
        num_layers=3,
        dropout=DROPOUT
    ).to(device)
    
    adapter = IGPAdapter(
        hidden_size=underlying_hidden_size,
        bottleneck_dim=128,
        dropout=DROPOUT
    ).to(device)
    
    gate = RatioGateV3(
        hidden_size=underlying_hidden_size,
        max_ratio=max_ratio  # 放松到 1.0
    ).to(device)
    
    print(f"   ✅ IGP 模块已初始化 (max_ratio={max_ratio}, dropout={DROPOUT})")
    
    return base_model, probe, adapter, gate


# ================= 训练循环 =================

class OverfitTrainer:
    """过拟合训练器"""
    
    def __init__(
        self,
        base_model,
        probe,
        adapter,
        gate,
        device,
        lr=5e-4,
        lr_multiplier=4.0,
        weight_decay=0.0
    ):
        self.base_model = base_model
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        self.device = device
        
        # 冻结基础模型参数
        for param in base_model.parameters():
            param.requires_grad = False
        
        # IGP 参数
        self.igp_params = list(probe.parameters()) + list(adapter.parameters()) + list(gate.parameters())
        
        # 优化器：IGP 参数使用更高学习率
        self.base_lr = lr
        self.lr_multiplier = lr_multiplier
        
        param_groups = [
            {'params': self.igp_params, 'lr': lr * lr_multiplier, 'weight_decay': weight_decay}
        ]
        
        self.optimizer = torch.optim.AdamW(param_groups)
        
        print(f"   优化器配置:")
        print(f"     - IGP 参数学习率: {lr * lr_multiplier}")
        print(f"     - 权重衰减: {weight_decay}")
        
        # 梯度记录
        self.grad_history = []
        self.gate_ratio_history = []
    
    def compute_loss(self, query_emb, pos_doc_emb, neg_doc_emb, instruction_mask=None):
        """计算对比损失"""
        
        # 计算相似度
        pos_scores = torch.sum(query_emb * pos_doc_emb, dim=-1)
        neg_scores = torch.sum(query_emb * neg_doc_emb, dim=-1)
        
        # InfoNCE 损失
        loss = F.cross_entropy(
            torch.stack([pos_scores, neg_scores], dim=1),
            torch.zeros(len(pos_scores), dtype=torch.long, device=self.device)
        )
        
        return loss
    
    def train_step(self, batch, step):
        """单步训练"""
        self.optimizer.zero_grad()
        
        query_ids = batch['query_input_ids'].to(self.device)
        query_mask = batch['query_attention_mask'].to(self.device)
        pos_ids = batch['positive_input_ids'].to(self.device)
        pos_mask = batch['positive_attention_mask'].to(self.device)
        neg_ids = batch['negative_input_ids'].to(self.device)
        neg_mask = batch['negative_attention_mask'].to(self.device)
        
        # 编码查询 - 使用基础模型
        with torch.no_grad():
            query_emb = self.base_model(
                query_ids,
                query_mask,
                is_query=True
            )
        
        # 编码文档 - 使用基础模型
        with torch.no_grad():
            pos_emb = self.base_model(pos_ids, pos_mask, is_query=False)
            neg_emb = self.base_model(neg_ids, neg_mask, is_query=False)
        
        # IGP 前向传播
        query_emb = query_emb.to(self.device)
        pos_emb = pos_emb.to(self.device)
        neg_emb = neg_emb.to(self.device)
        
        # 使用 Probe 处理查询
        processed_query = self.probe(query_emb, query_mask)
        
        # 使用 Adapter 和 Gate 处理查询
        adapted_query, gate_ratio = self.gate(processed_query, processed_query)
        
        # 计算损失
        loss = self.compute_loss(adapted_query, pos_emb, neg_emb)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.igp_params, max_norm=1.0)
        
        self.optimizer.step()
        
        # 记录梯度信息
        if step % GRADIENT_LOG_STEPS == 0:
            self.log_gradients(step)
            self.log_gate_ratio(step, gate_ratio)
        
        return loss.item(), gate_ratio.mean().item()
    
    def log_gradients(self, step):
        """记录梯度信息"""
        grad_norms = {}
        
        for name, param in self.probe.named_parameters():
            if param.grad is not None:
                grad_norms[f'probe_{name}'] = param.grad.norm().item()
        
        for name, param in self.adapter.named_parameters():
            if param.grad is not None:
                grad_norms[f'adapter_{name}'] = param.grad.norm().item()
        
        for name, param in self.gate.named_parameters():
            if param.grad is not None:
                grad_norms[f'gate_{name}'] = param.grad.norm().item()
        
        # 检查梯度消失
        for name, grad_norm in grad_norms.items():
            if grad_norm < 1e-6:
                print(f"⚠️ Step {step}: 梯度消失检测 - {name}: {grad_norm:.2e}")
        
        self.grad_history.append({
            'step': step,
            'norms': grad_norms
        })
    
    def log_gate_ratio(self, step, gate_ratio):
        """记录门控激活"""
        gate_stats = {
            'mean': gate_ratio.mean().item(),
            'max': gate_ratio.max().item(),
            'min': gate_ratio.min().item(),
            'std': gate_ratio.std().item()
        }
        
        self.gate_ratio_history.append({
            'step': step,
            'stats': gate_stats
        })
        
        if step % (GRADIENT_LOG_STEPS * 10) == 0:
            print(f"   Gate Ratio - Mean: {gate_stats['mean']:.4f}, Max: {gate_stats['max']:.4f}, Std: {gate_stats['std']:.4f}")
    
    def train(self, dataloader, epochs):
        """训练循环"""
        print(f"\n🚀 开始过拟合训练 ({epochs} epochs)")
        print("=" * 60)
        
        global_step = 0
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # 重复使用同一批次数据 (真正的过拟合)
            dataloader_iter = iter(dataloader)
            
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
            
            while True:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)
                
                loss, gate_ratio = self.train_step(batch, global_step)
                epoch_losses.append(loss)
                
                pbar.set_postfix({
                    'loss': f'{np.mean(epoch_losses[-10:]):.4f}',
                    'gate': f'{gate_ratio:.4f}'
                })
                pbar.update(1)
                
                global_step += 1
                
                # 每 epoch 打印一次
                if global_step % len(dataloader) == 0:
                    avg_loss = np.mean(epoch_losses)
                    print(f"\n📊 Epoch {epoch+1}/{epochs}: Avg Loss = {avg_loss:.6f}")
                    
                    # 检查损失是否收敛到接近零
                    if avg_loss < 0.01:
                        print(f"✅ 损失已收敛到 {avg_loss:.6f}，提前停止训练")
                        break
            
            pbar.close()
            
            # 每个 epoch 打印门控统计
            if self.gate_ratio_history:
                last_gate = self.gate_ratio_history[-1]['stats']
                print(f"   Gate 统计: mean={last_gate['mean']:.4f}, max={last_gate['max']:.4f}")
        
        print("\n" + "=" * 60)
        print("✅ 过拟合训练完成")
        
        return self.grad_history, self.gate_ratio_history
    
    def save_model(self, output_dir):
        """保存模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(self.probe.state_dict(), os.path.join(output_dir, "igp_probe.pt"))
        torch.save(self.adapter.state_dict(), os.path.join(output_dir, "igp_adapter.pt"))
        torch.save(self.gate.state_dict(), os.path.join(output_dir, "igp_gate.pt"))
        
        igp_info = {
            'phase': 'overfit_test',
            'modules': {
                'probe': 'igp_probe.pt',
                'adapter': 'igp_adapter.pt',
                'gate': 'igp_gate.pt'
            },
            'config': {
                'overfit_epochs': OVERFIT_EPOCHS,
                'lr': OVERFIT_LR,
                'lr_multiplier': OVERFIT_LR_MULTIPLIER,
                'max_ratio': MAX_RATIO,
                'dropout': DROPOUT
            }
        }
        
        with open(os.path.join(output_dir, "igp_info.json"), 'w') as f:
            json.dump(igp_info, f, indent=2)
        
        # 保存梯度历史
        with open(os.path.join(output_dir, "grad_history.json"), 'w') as f:
            json.dump(self.grad_history, f, indent=2)
        
        # 保存门控历史
        with open(os.path.join(output_dir, "gate_history.json"), 'w') as f:
            json.dump(self.gate_ratio_history, f, indent=2)
        
        print(f"✅ 模型已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="过拟合训练脚本")
    parser.add_argument('--model_path', type=str, default=MODEL_PATH)
    parser.add_argument('--data_path', type=str, default=DATA_PATH)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--epochs', type=int, default=OVERFIT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=OVERFIT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=OVERFIT_LR)
    parser.add_argument('--lr_multiplier', type=float, default=OVERFIT_LR_MULTIPLIER)
    parser.add_argument('--max_ratio', type=float, default=MAX_RATIO)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Upper-Bound Overfitting Test - Training")
    print("=" * 70)
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.data_path}")
    print(f"输出路径: {args.output_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr} (IGP: {args.lr * args.lr_multiplier})")
    print(f"门控上限: {args.max_ratio}")
    print("=" * 70)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"❌ 数据文件不存在: {args.data_path}")
        print("请先运行: python scripts/overfit_test/sample_micro_batch.py")
        return
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载 tokenizer
    print("\n📝 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 加载数据集
    print("\n📂 加载数据集...")
    dataset = OverfitDataset(args.data_path, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # 加载模型
    print("\n📥 加载模型...")
    base_model, probe, adapter, gate = load_igp_model(
        args.model_path,
        device=args.device,
        max_ratio=args.max_ratio
    )
    
    # 创建训练器
    trainer = OverfitTrainer(
        base_model=base_model,
        probe=probe,
        adapter=adapter,
        gate=gate,
        device=args.device,
        lr=args.lr,
        lr_multiplier=args.lr_multiplier,
        weight_decay=WEIGHT_DECAY
    )
    
    # 训练
    grad_history, gate_history = trainer.train(dataloader, args.epochs)
    
    # 保存模型
    trainer.save_model(args.output_dir)
    
    print("\n✅ 过拟合测试训练完成!")
    print(f"   查看梯度历史: {os.path.join(args.output_dir, 'grad_history.json')}")
    print(f"   查看门控历史: {os.path.join(args.output_dir, 'gate_history.json')}")


if __name__ == "__main__":
    main()
