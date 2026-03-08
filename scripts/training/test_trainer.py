"""
测试 IGPColBERTTrainer 是否正确使用 IGP 损失
"""
import os
import sys
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_colbert_igp import IGPColBERTTrainer, IGPColBERTWrapper
from igp_losses import IGPLoss
from pylate import models, losses
from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate
from sentence_transformers import SentenceTransformerTrainingArguments
from datasets import Dataset

def test_trainer_loss():
    """测试 Trainer 是否正确使用 IGP 损失"""
    print("=" * 60)
    print("测试 IGPColBERTTrainer 损失计算")
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
    adapter = IGPAdapter(hidden_size=hidden_size, bottleneck_dim=64).to(device)
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
    
    # 创建测试数据
    print("\n创建测试数据...")
    data = {
        'anchor': ['query 1', 'query 2', 'query 3', 'query 4'],
        'positive': ['pos 1', 'pos 2', 'pos 3', 'pos 4'],
        'negative': ['neg 1', 'neg 2', 'neg 3', 'neg 4'],
    }
    dataset = Dataset.from_dict(data)
    
    # 配置训练参数
    training_args = SentenceTransformerTrainingArguments(
        output_dir="/tmp/test_output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        eval_strategy="epoch",
        save_strategy="no",
    )
    
    # 创建 Trainer
    print("\n创建 IGPColBERTTrainer...")
    trainer = IGPColBERTTrainer(
        model=igp_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        loss=igp_loss,
    )
    
    # 测试1: 检查 self.loss 是否正确设置
    print("\n" + "=" * 60)
    print("测试1: 检查 self.loss 设置")
    print("=" * 60)
    print(f"trainer.loss 类型: {type(trainer.loss)}")
    print(f"trainer.loss 是 IGPLoss: {isinstance(trainer.loss, IGPLoss)}")
    
    # 测试2: 测试 compute_loss
    print("\n" + "=" * 60)
    print("测试2: 测试 compute_loss")
    print("=" * 60)
    
    # 模拟输入
    batch_size = 2
    seq_len = 32
    inputs = {
        'sentence_0_input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'sentence_0_attention_mask': torch.ones(batch_size, seq_len).to(device),
        'sentence_1_input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'sentence_1_attention_mask': torch.ones(batch_size, seq_len).to(device),
        'sentence_2_input_ids': torch.randint(0, 1000, (batch_size, seq_len)).to(device),
        'sentence_2_attention_mask': torch.ones(batch_size, seq_len).to(device),
    }
    
    # 训练模式
    igp_model.train()
    loss_train = trainer.compute_loss(igp_model, inputs.copy(), return_outputs=False)
    print(f"训练损失: {loss_train.item():.4f}")
    
    # 获取详细损失
    last_losses = igp_loss.get_last_losses()
    print(f"详细损失: {last_losses}")
    
    # 测试3: 测试 prediction_step
    print("\n" + "=" * 60)
    print("测试3: 测试 prediction_step")
    print("=" * 60)
    
    igp_model.eval()
    loss_eval, _, _ = trainer.prediction_step(igp_model, inputs.copy(), prediction_loss_only=True)
    print(f"评估损失: {loss_eval.item():.4f}")
    
    # 获取详细损失
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
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_trainer_loss()
