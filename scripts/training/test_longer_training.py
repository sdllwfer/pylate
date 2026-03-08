"""
测试脚本：运行更长时间的训练，观察损失差距的变化
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

def test_longer_training():
    """运行更长时间的训练"""
    print("=" * 60)
    print("测试：运行 10 个 epoch，观察损失差距的变化")
    print("=" * 60)
    
    # 创建测试数据
    test_data_path = "/tmp/test_longer_data.jsonl"
    
    # 生成更多测试数据
    test_data = []
    for i in range(500):
        test_data.append({
            "query": f"This is a test query about topic {i % 20} with some additional context to make it more realistic",
            "pos": [f"This is a positive document for query {i}. It contains relevant information about the topic."],
            "neg": [f"This is a negative document for query {i}. It contains irrelevant information about a different topic."]
        })
    
    with open(test_data_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n测试数据已生成: {test_data_path}")
    print(f"样本数: {len(test_data)}")
    
    # 配置训练参数
    output_dir = "/tmp/test_longer_output"
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    trainer = IGPTrainer(
        model_name="answerdotai/ModernBERT-base",
        train_data_path=test_data_path,
        output_dir=output_dir,
        device="cuda:0",
        phase1_epochs=0,
        phase2_epochs=10,
        batch_size=32,
        eval_ratio=0.1,
        base_lr=2e-5,
        gate_lr=5e-2,
        enable_phase1=False,
        enable_phase2=True,
        enable_probe=True,
        enable_adapter=True,
        enable_gate=True,
        max_ratio=0.2,
        bottleneck_dim=64,
        aux_loss_weight=0.0,
        phase2_patience=20,
        phase2_early_stop_threshold=0.0001,
        log_interval=10,
    )
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    try:
        trainer.train()
        
        # 读取损失历史
        loss_history_path = os.path.join(output_dir, "phase2", "loss_history.json")
        if os.path.exists(loss_history_path):
            with open(loss_history_path, 'r') as f:
                history = json.load(f)
            
            print("\n" + "=" * 60)
            print("训练结果分析")
            print("=" * 60)
            
            eval_losses = history.get('eval_losses', [])
            epochs = history.get('epochs', [])
            
            print(f"\n验证损失记录数: {len(eval_losses)}")
            
            if eval_losses:
                print("\n各 Epoch 验证损失:")
                print("-" * 60)
                print(f"{'Epoch':<10} {'Eval Loss':<15}")
                print("-" * 60)
                
                for epoch, eval_loss in zip(epochs, eval_losses):
                    if eval_loss is not None:
                        print(f"{epoch:<10} {eval_loss:<15.4f}")
                
                print("-" * 60)
                
                # 检查损失是否收敛
                if len(eval_losses) >= 2 and eval_losses[-1] is not None and eval_losses[0] is not None:
                    initial_loss = eval_losses[0] if eval_losses[0] is not None else eval_losses[1]
                    final_loss = eval_losses[-1]
                    improvement = initial_loss - final_loss
                    
                    print(f"\n初始验证损失: {initial_loss:.4f}")
                    print(f"最终验证损失: {final_loss:.4f}")
                    print(f"改进: {improvement:.4f}")
                    
                    if improvement > 0.1:
                        print("\n✅ 模型正在学习，损失在下降！")
                    else:
                        print("\n⚠️ 模型学习效果不明显")
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_longer_training()
    sys.exit(0 if success else 1)
