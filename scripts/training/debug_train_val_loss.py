"""
调试脚本：检查训练和验证损失的具体值
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from train_colbert_igp import IGPColBERTWrapper
from igp_losses import IGPLoss
from pylate import models, losses
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate

def debug_loss():
    """调试损失计算"""
    print("=" * 60)
    print("调试训练和验证损失")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    
    # 加载模型
    model_name = "answerdotai/ModernBERT-base"
    print(f"加载模型: {model_name}")
    base_model = models.ColBERT(model_name_or_path=model_name, device=device)
    
    hidden_size = base_model[0].get_word_embedding_dimension()
    print(f"隐藏层维度: {hidden_size}")
    
    # 创建 IGP 模块
    probe = InstructionProbe(hidden_size=hidden_size, num_heads=4, dropout=0.1).to(device)
    adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=64, input_dim=hidden_size*2).to(device)
    gate = RatioGate(hidden_size=hidden_size, max_ratio=0.2).to(device)
    
    # 创建包装器
    igp_model = IGPColBERTWrapper(
        base_model=base_model,
        probe=probe,
        adapter=adapter,
        gate=gate,
    )
    
    # 创建损失函数
    base_loss = losses.Contrastive(model=base_model)
    igp_loss = IGPLoss(
        base_loss=base_loss,
        base_model=igp_model,
        probe=probe,
        adapter=adapter,
        gate=gate,
        aux_loss_weight=0.0,
    )
    
    # 创建相同的测试数据
    batch_size = 4
    seq_len = 32
    
    # 固定随机种子以确保可重复
    torch.manual_seed(42)
    
    query_features = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
    }
    pos_features = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
    }
    neg_features = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'attention_mask': torch.ones(batch_size, seq_len).to(device),
    }
    
    sentence_features = [query_features, pos_features, neg_features]
    
    # 测试1: 训练模式
    print("\n" + "=" * 60)
    print("测试1: 训练模式 (model.train())")
    print("=" * 60)
    igp_model.train()
    probe.train()
    adapter.train()
    gate.train()
    
    # 运行多次取平均
    train_losses = []
    for i in range(5):
        with torch.set_grad_enabled(True):
            loss = igp_loss(sentence_features)
            train_losses.append(loss.item())
            last_losses = igp_loss.get_last_losses()
    
    avg_train_loss = sum(train_losses) / len(train_losses)
    print(f"训练损失 (平均): {avg_train_loss:.4f}")
    print(f"训练损失 (各次): {[f'{l:.4f}' for l in train_losses]}")
    print(f"详细损失: {last_losses}")
    
    # 测试2: 评估模式
    print("\n" + "=" * 60)
    print("测试2: 评估模式 (model.eval())")
    print("=" * 60)
    igp_model.eval()
    probe.eval()
    adapter.eval()
    gate.eval()
    
    eval_losses = []
    for i in range(5):
        with torch.no_grad():
            loss = igp_loss(sentence_features)
            eval_losses.append(loss.item())
            last_losses_eval = igp_loss.get_last_losses()
    
    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    print(f"评估损失 (平均): {avg_eval_loss:.4f}")
    print(f"评估损失 (各次): {[f'{l:.4f}' for l in eval_losses]}")
    print(f"详细损失: {last_losses_eval}")
    
    # 比较
    print("\n" + "=" * 60)
    print("比较结果")
    print("=" * 60)
    diff = abs(avg_train_loss - avg_eval_loss)
    print(f"平均损失差异: {diff:.4f}")
    print(f"训练损失标准差: {torch.tensor(train_losses).std().item():.4f}")
    print(f"评估损失标准差: {torch.tensor(eval_losses).std().item():.4f}")
    
    if diff < 0.1:
        print("\n✅ 训练和评估损失基本一致！")
    else:
        print("\n⚠️ 训练和评估损失存在显著差异")
        print(f"   训练损失: {avg_train_loss:.4f}")
        print(f"   评估损失: {avg_eval_loss:.4f}")
        print(f"   差异: {diff:.4f}")
    
    # 分析可能原因
    print("\n" + "=" * 60)
    print("分析")
    print("=" * 60)
    
    if diff > 0.1:
        print("\n可能原因:")
        print("1. Dropout 在训练时启用，评估时关闭")
        print("   - 训练时 dropout=0.1，评估时 dropout=0")
        print("   - 这会导致训练损失略高，但差距不应超过 0.1-0.2")
        print("\n2. 如果差距很大，可能是:")
        print("   - IGP 模块在评估模式下行为不同")
        print("   - 某些层在 eval() 模式下被禁用")
        
        # 检查是否有 BatchNorm 或 Dropout
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in igp_model.modules())
        has_batchnorm = any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)) for m in igp_model.modules())
        
        print(f"\n模型结构检查:")
        print(f"   包含 Dropout: {has_dropout}")
        print(f"   包含 BatchNorm/LayerNorm: {has_batchnorm}")
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    debug_loss()
