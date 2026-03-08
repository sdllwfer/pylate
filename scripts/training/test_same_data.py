"""
测试脚本：使用相同的数据进行训练和验证
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

def test_same_data():
    """使用相同数据进行训练和验证"""
    print("=" * 60)
    print("测试：使用相同数据进行训练和验证")
    print("=" * 60)
    
    # 创建测试数据
    test_data_path = "/tmp/test_same_data.jsonl"
    
    # 生成测试数据
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
    
    # 配置训练参数
    output_dir = "/tmp/test_same_data_output"
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # 关键：设置 eval_ratio=0，这样训练和验证使用相同的数据
    trainer = IGPTrainer(
        model_name="answerdotai/ModernBERT-base",
        train_data_path=test_data_path,
        output_dir=output_dir,
        device="cuda:0",
        phase1_epochs=0,
        phase2_epochs=3,
        batch_size=16,
        eval_ratio=0.0,  # 使用相同数据进行训练和验证
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
    print("注意：eval_ratio=0，训练和验证使用相同的数据")
    print("=" * 60)
    
    try:
        trainer.train()
        
        # 读取训练日志
        import subprocess
        result = subprocess.run(
            ["grep", "train_loss\|eval_loss", f"{output_dir}/training_summary.txt"],
            capture_output=True,
            text=True
        )
        
        print("\n" + "=" * 60)
        print("训练日志中的损失值")
        print("=" * 60)
        print(result.stdout)
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        print("\n如果训练和验证使用相同的数据，但损失差距仍然很大，")
        print("则说明问题不在于数据分布，而在于模型在训练和评估模式下的行为差异。")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_same_data()
    sys.exit(0 if success else 1)
