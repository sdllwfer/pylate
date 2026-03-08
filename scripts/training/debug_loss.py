"""
调试脚本：检查训练和验证损失的差异
"""
import os
import sys
import torch

# 设置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_colbert_igp import IGPColBERTTrainer, IGPColBERTWrapper
from igp_losses import IGPLoss
from data_utils import DataLoader
from pylate import models, losses
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate

def debug_loss_computation():
    """调试损失计算"""
    print("=" * 60)
    print("调试训练/验证损失差异")
    print("=" * 60)
    
    # 加载模型
    model_name = "answerdotai/ModernBERT-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n设备: {device}")
    
    print(f"\n加载基础模型: {model_name}")
    base_model = models.ColBERT(model_name_or_path=model_name, device=device)
    
    # 获取维度 - 使用 underlying_hidden_size (768) 而不是投影后的 128
    underlying_hidden_size = base_model[0].get_word_embedding_dimension()
    print(f"底层隐藏层维度: {underlying_hidden_size}")
    print(f"投影后维度: {base_model.get_sentence_embedding_dimension()}")
    
    # 创建 IGP 模块
    probe = InstructionProbe(
        hidden_size=underlying_hidden_size,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    adapter = IGPAdapter(
        hidden_size=underlying_hidden_size,
        bottleneck_dim=64
    ).to(device)
    
    gate = RatioGate(
        hidden_size=underlying_hidden_size,
        max_ratio=0.2
    ).to(device)
    
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
        aux_loss_weight=0.0,  # 跳过 aux loss
    )
    
    # 创建测试数据
    print("\n创建测试数据...")
    batch_size = 4
    seq_len = 32
    
    # 模拟输入
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
    
    # 测试1: 训练模式下的损失
    print("\n" + "=" * 60)
    print("测试1: 训练模式")
    print("=" * 60)
    igp_model.train()
    probe.train()
    adapter.train()
    gate.train()
    
    sentence_features = [query_features, pos_features, neg_features]
    
    # 前向传播
    with torch.set_grad_enabled(True):
        loss_train = igp_loss(sentence_features)
    
    print(f"训练损失: {loss_train.item():.4f}")
    print(f"损失 requires_grad: {loss_train.requires_grad}")
    
    # 获取详细损失信息
    last_losses = igp_loss.get_last_losses()
    print(f"详细损失: {last_losses}")
    
    # 测试2: 评估模式下的损失
    print("\n" + "=" * 60)
    print("测试2: 评估模式")
    print("=" * 60)
    igp_model.eval()
    probe.eval()
    adapter.eval()
    gate.eval()
    
    with torch.no_grad():
        loss_eval = igp_loss(sentence_features)
    
    print(f"评估损失: {loss_eval.item():.4f}")
    
    last_losses_eval = igp_loss.get_last_losses()
    print(f"详细损失: {last_losses_eval}")
    
    # 比较
    print("\n" + "=" * 60)
    print("比较结果")
    print("=" * 60)
    diff = abs(loss_train.item() - loss_eval.item())
    print(f"损失差异: {diff:.4f}")
    
    if diff < 0.01:
        print("✅ 训练和评估损失基本一致")
    else:
        print("⚠️ 训练和评估损失存在显著差异")
        print("\n可能原因:")
        print("1. Dropout 在训练时启用，评估时关闭")
        print("2. BatchNorm 行为不同")
        print("3. 某些层在训练和评估模式下行为不同")
    
    # 测试3: 检查 IGP 模块是否被使用
    print("\n" + "=" * 60)
    print("测试3: 检查 IGP 模块使用情况")
    print("=" * 60)
    
    # 重新运行训练模式，检查 gate ratio
    igp_model.train()
    probe.train()
    adapter.train()
    gate.train()
    
    with torch.set_grad_enabled(True):
        _ = igp_loss(sentence_features)
        last_losses = igp_loss.get_last_losses()
        print(f"Gate Ratio: {last_losses.get('gate_ratio', 'N/A')}")
        print(f"Rank Loss: {last_losses.get('rank_loss', 'N/A')}")
        print(f"Aux Loss: {last_losses.get('aux_loss', 'N/A')}")
        print(f"Reg Loss: {last_losses.get('reg_loss', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    debug_loss_computation()
