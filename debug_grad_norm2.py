#!/usr/bin/env python3
"""
调试 grad_norm=0.0 的问题 - 使用 training_step
"""
import torch
import sys
sys.path.insert(0, '/home/luwa/Documents/pylate')

from datasets import Dataset
from pylate import models, losses
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate
from scripts.training.train_colbert_igp import IGPColBERTWrapper
from scripts.training.igp_losses import IGPLoss
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

print("=" * 70)
print("🔍 调试 grad_norm=0.0 的问题")
print("=" * 70)

# 创建测试数据
print("\n[1/3] 创建测试数据...")
train_data = {
    'anchor': ['find information about cats', 'search for dog breeds'] * 4,
    'positive': ['cats are furry animals', 'dogs are loyal pets'] * 4,
    'negative': ['the sky is blue', 'apples are fruits'] * 4,
}
train_dataset = Dataset.from_dict(train_data)

# 加载基础模型
print("\n[2/3] 加载基础模型...")
base_model = models.ColBERT(model_name_or_path="answerdotai/ModernBERT-base")
device = next(base_model.parameters()).device

# 初始化 IGP 模块
hidden_size = base_model[0].get_word_embedding_dimension()
probe = InstructionProbe(hidden_size=hidden_size, num_heads=8, dropout=0.1).to(device)
adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=64, dropout=0.1, input_dim=hidden_size * 2).to(device)
gate = RatioGate(hidden_size=hidden_size, max_ratio=0.2, use_dynamic=False).to(device)

# 创建 IGPColBERTWrapper
print("\n[3/3] 创建 IGPColBERTWrapper...")
igp_model = IGPColBERTWrapper(
    base_model=base_model,
    probe=probe,
    adapter=adapter,
    gate=gate,
)
igp_model.set_phase2_mode()

# 检查梯度状态
print("\n📍 检查梯度状态:")
probe_has_grad = any(p.requires_grad for p in probe.parameters())
adapter_has_grad = any(p.requires_grad for p in adapter.parameters())
gate_has_grad = any(p.requires_grad for p in gate.parameters())
print(f"   Probe requires_grad: {probe_has_grad}")
print(f"   Adapter requires_grad: {adapter_has_grad}")
print(f"   Gate requires_grad: {gate_has_grad}")

# 创建 IGPLoss
base_loss = losses.Contrastive(model=base_model)
igp_loss = IGPLoss(
    base_loss=base_loss,
    base_model=igp_model,
    probe=probe,
    adapter=adapter,
    gate=gate,
    aux_loss_weight=0.0,
)

# 创建优化器
param_groups = [
    {'params': [p for p in probe.parameters() if p.requires_grad], 'lr': 1e-5},
    {'params': [p for p in adapter.parameters() if p.requires_grad], 'lr': 1e-5},
    {'params': [p for p in gate.parameters() if p.requires_grad], 'lr': 1e-2},
]
optimizer = torch.optim.AdamW(param_groups)

# 创建 trainer
print("\n   创建 trainer...")
training_args = SentenceTransformerTrainingArguments(
    output_dir="/tmp/test_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
)

trainer = SentenceTransformerTrainer(
    model=igp_model,
    args=training_args,
    train_dataset=train_dataset,
    loss=igp_loss,
    optimizers=(optimizer, None),
)

# 检查 trainer 中的优化器
print("\n📍 检查 trainer 中的优化器:")
print(f"   trainer.optimizer: {trainer.optimizer}")
print(f"   trainer.optimizer type: {type(trainer.optimizer)}")
if hasattr(trainer.optimizer, 'param_groups'):
    print(f"   参数组数: {len(trainer.optimizer.param_groups)}")
    for i, pg in enumerate(trainer.optimizer.param_groups):
        print(f"   组 {i}: {len(pg['params'])} 个参数, lr={pg.get('lr', 'N/A')}")

# 获取一个 batch 进行测试
print("\n📍 获取一个 batch 进行测试...")
dataloader = trainer.get_train_dataloader()
batch = next(iter(dataloader))

# 准备输入
inputs = trainer._prepare_inputs(batch)

# 使用 training_step
print("\n📍 调用 training_step...")
loss = trainer.training_step(igp_model, inputs, num_items_in_batch=None)

print(f"   loss: {loss}")
print(f"   loss.requires_grad: {loss.requires_grad if torch.is_tensor(loss) else 'N/A'}")

# 检查梯度
print("\n📍 检查梯度:")
probe_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in probe.parameters())
adapter_down_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.down_project.parameters())
adapter_up_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.up_project.parameters())
adapter_ln_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.layer_norm.parameters())
gate_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters())

print(f"   Probe 有梯度: {probe_has_grad}")
print(f"   Adapter down_project 有梯度: {adapter_down_has_grad}")
print(f"   Adapter up_project 有梯度: {adapter_up_has_grad}")
print(f"   Adapter layer_norm 有梯度: {adapter_ln_has_grad}")
print(f"   Gate 有梯度: {gate_has_grad}")

# 计算 grad_norm
total_norm = 0.0
for p in probe.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
for p in adapter.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
for p in gate.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5

print(f"\n   总 grad_norm: {total_norm:.6f}")

print("\n" + "=" * 70)
print("✅ 调试完成")
print("=" * 70)
