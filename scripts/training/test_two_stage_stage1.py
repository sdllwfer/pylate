"""
测试两阶段训练的第一阶段（短数据集）
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_colbert_igp import IGPTrainer
import json

def test_stage1():
    """测试阶段1：短数据集训练"""
    print("=" * 60)
    print("测试阶段1：短数据集训练")
    print("=" * 60)
    
    # 短数据集路径
    short_train_data = "/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl"
    
    print(f"\n训练数据: {short_train_data}")
    
    # 检查文件是否存在
    if not os.path.exists(short_train_data):
        print(f"❌ 文件不存在: {short_train_data}")
        return False
    
    # 统计样本数
    with open(short_train_data, 'r') as f:
        num_samples = sum(1 for _ in f)
    print(f"样本数: {num_samples}")
    
    # 配置训练参数
    output_dir = "/tmp/test_stage1_output"
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    trainer = IGPTrainer(
        model_name="answerdotai/ModernBERT-base",
        train_data_path=short_train_data,
        output_dir=output_dir,
        device="cuda:0",
        phase1_epochs=0,
        phase2_epochs=5,  # 只训练5个epoch进行测试
        batch_size=32,
        eval_ratio=0.05,
        base_lr=2e-5,
        gate_lr=5e-2,
        enable_phase1=False,
        enable_phase2=True,
        enable_probe=True,
        enable_adapter=True,
        enable_gate=True,
        max_ratio=0.2,
        bottleneck_dim=128,
        aux_loss_weight=0.0,
        phase2_patience=10,
        phase2_early_stop_threshold=0.0001,
        log_interval=10,
    )
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    try:
        trainer.train()
        
        # 检查损失历史
        loss_history_path = os.path.join(output_dir, "phase2", "loss_history.json")
        if os.path.exists(loss_history_path):
            with open(loss_history_path, 'r') as f:
                history = json.load(f)
            
            print("\n" + "=" * 60)
            print("训练结果")
            print("=" * 60)
            
            eval_losses = history.get('eval_losses', [])
            epochs = history.get('epochs', [])
            
            print(f"\n验证损失记录:")
            print("-" * 60)
            for epoch, eval_loss in zip(epochs, eval_losses):
                if eval_loss is not None:
                    print(f"  Epoch {epoch}: {eval_loss:.4f}")
            
            if len(eval_losses) >= 2 and eval_losses[-1] is not None and eval_losses[1] is not None:
                initial = eval_losses[1]  # 第一个非null值
                final = eval_losses[-1]
                improvement = initial - final
                print(f"\n初始损失: {initial:.4f}")
                print(f"最终损失: {final:.4f}")
                print(f"改进: {improvement:.4f}")
                
                if improvement > 0.1:
                    print("\n✅ 模型正在学习！")
                else:
                    print("\n⚠️ 学习效果不明显")
        
        print("\n" + "=" * 60)
        print("✅ 阶段1测试完成")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stage1()
    sys.exit(0 if success else 1)
