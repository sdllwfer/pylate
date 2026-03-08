"""
快速测试训练脚本 - 使用小数据集和少 epoch 验证修复
"""
import os
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_colbert_igp import IGPTrainer

def quick_test():
    """快速测试训练"""
    print("=" * 60)
    print("快速测试 - 验证训练/验证损失一致性")
    print("=" * 60)
    
    # 创建测试用的最小数据集
    test_data_path = "/tmp/test_train_data.jsonl"
    
    # 生成测试数据
    import json
    test_data = []
    for i in range(100):
        test_data.append({
            "query": f"This is a test query about topic {i % 10}",
            "pos": [f"This is a positive document for query {i}"],
            "neg": [f"This is a negative document for query {i}"]
        })
    
    with open(test_data_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n测试数据已生成: {test_data_path}")
    print(f"样本数: {len(test_data)}")
    
    # 配置训练参数 - 使用非常小的参数快速测试
    output_dir = "/tmp/test_igp_output"
    
    # 清理之前的输出
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    trainer = IGPTrainer(
        model_name="answerdotai/ModernBERT-base",
        train_data_path=test_data_path,
        output_dir=output_dir,
        device="cuda:0",
        phase1_epochs=0,  # 跳过 phase1
        phase2_epochs=3,  # 只训练 3 个 epoch
        batch_size=16,
        eval_ratio=0.1,  # 10% 验证集
        base_lr=1e-5,
        gate_lr=1e-2,
        enable_phase1=False,
        enable_phase2=True,
        enable_probe=True,
        enable_adapter=True,
        enable_gate=True,
        max_ratio=0.2,
        bottleneck_dim=64,
        aux_loss_weight=0.0,
        phase2_patience=10,
        phase2_early_stop_threshold=0.001,
        log_interval=5,
    )
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    try:
        trainer.train()
        
        # 读取损失历史
        import json
        loss_history_path = os.path.join(output_dir, "phase2", "loss_history.json")
        if os.path.exists(loss_history_path):
            with open(loss_history_path, 'r') as f:
                history = json.load(f)
            
            print("\n" + "=" * 60)
            print("训练结果分析")
            print("=" * 60)
            
            train_losses = history.get('train_losses', [])
            eval_losses = history.get('eval_losses', [])
            
            print(f"\n训练损失记录数: {len(train_losses)}")
            print(f"验证损失记录数: {len(eval_losses)}")
            
            if train_losses and eval_losses:
                # 计算每个 epoch 的平均训练损失
                epochs = history.get('epochs', [])
                
                print("\n各 Epoch 损失对比:")
                print("-" * 60)
                print(f"{'Epoch':<10} {'Train Loss':<15} {'Eval Loss':<15} {'Diff':<15}")
                print("-" * 60)
                
                for i, (epoch, train_loss, eval_loss) in enumerate(zip(epochs, train_losses, eval_losses)):
                    if train_loss is not None and eval_loss is not None:
                        diff = abs(train_loss - eval_loss)
                        print(f"{epoch:<10} {train_loss:<15.4f} {eval_loss:<15.4f} {diff:<15.4f}")
                
                print("-" * 60)
                
                # 检查损失差距
                valid_diffs = [abs(t - e) for t, e in zip(train_losses, eval_losses) 
                              if t is not None and e is not None]
                if valid_diffs:
                    avg_diff = sum(valid_diffs) / len(valid_diffs)
                    max_diff = max(valid_diffs)
                    
                    print(f"\n平均损失差距: {avg_diff:.4f}")
                    print(f"最大损失差距: {max_diff:.4f}")
                    
                    if avg_diff < 0.5:
                        print("\n✅ 训练和验证损失基本一致！")
                    else:
                        print("\n⚠️ 训练和验证损失差距较大，需要进一步调查")
        
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
    success = quick_test()
    sys.exit(0 if success else 1)
