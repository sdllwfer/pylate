#!/usr/bin/env python3
"""
验证改进版过拟合训练集中的难负样本质量
"""

import json
import random

def load_training_data(file_path: str):
    """加载训练数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def verify_samples(data: list, n_samples: int = 5):
    """验证随机样本"""
    print(f"\n{'='*80}")
    print(f"随机验证 {n_samples} 个样本")
    print(f"{'='*80}")
    
    # 分离有指令和无指令样本
    with_inst = [item for item in data if item['type'] == 'with_instruction']
    without_inst = [item for item in data if item['type'] == 'without_instruction']
    
    print(f"\n📊 数据集统计:")
    print(f"   总样本数: {len(data)}")
    print(f"   有指令样本: {len(with_inst)}")
    print(f"   无指令样本: {len(without_inst)}")
    
    # 验证有指令样本
    print(f"\n{'='*80}")
    print("有指令样本验证 (难负样本应该是 og 相关但 changed 不相关)")
    print(f"{'='*80}")
    
    samples = random.sample(with_inst, min(n_samples, len(with_inst)))
    for i, item in enumerate(samples, 1):
        print(f"\n--- 样本 {i} ---")
        print(f"Query: {item['query'][:200]}...")
        print(f"Instruction: {item['instruction'][:150]}...")
        print(f"Dataset: {item['dataset']}, Base QID: {item['base_qid']}")
        print(f"Stats: {item['stats']}")
        print(f"\n正样本数量: {len(item['pos'])}")
        print(f"难负样本数量: {len(item['neg'])}")
        
        print(f"\n正样本示例 (前1个):")
        if item['pos']:
            print(f"  {item['pos'][0][:300]}...")
        
        print(f"\n难负样本示例 (前1个):")
        if item['neg']:
            print(f"  {item['neg'][0][:300]}...")
    
    # 验证无指令样本
    print(f"\n{'='*80}")
    print("无指令样本验证 (普通负样本)")
    print(f"{'='*80}")
    
    samples = random.sample(without_inst, min(n_samples, len(without_inst)))
    for i, item in enumerate(samples, 1):
        print(f"\n--- 样本 {i} ---")
        print(f"Query: {item['query'][:200]}...")
        print(f"Dataset: {item['dataset']}, Base QID: {item['base_qid']}")
        print(f"Stats: {item['stats']}")
        print(f"\n正样本数量: {len(item['pos'])}")
        print(f"负样本数量: {len(item['neg'])}")
        
        print(f"\n正样本示例 (前1个):")
        if item['pos']:
            print(f"  {item['pos'][0][:300]}...")
        
        print(f"\n负样本示例 (前1个):")
        if item['neg']:
            print(f"  {item['neg'][0][:300]}...")

def compare_with_old_version():
    """对比新旧版本的数据"""
    print(f"\n{'='*80}")
    print("新旧版本对比")
    print(f"{'='*80}")
    
    old_file = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions.jsonl"
    new_file = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions_v2.jsonl"
    
    try:
        old_data = load_training_data(old_file)
        print(f"\n旧版本: {len(old_data)} 样本")
        old_with_inst = sum(1 for item in old_data if item.get('type') == 'with_instruction')
        old_without_inst = sum(1 for item in old_data if item.get('type') == 'without_instruction')
        print(f"  - 有指令: {old_with_inst}")
        print(f"  - 无指令: {old_without_inst}")
    except FileNotFoundError:
        print(f"\n旧版本文件不存在: {old_file}")
    
    new_data = load_training_data(new_file)
    print(f"\n新版本 (v2): {len(new_data)} 样本")
    new_with_inst = sum(1 for item in new_data if item.get('type') == 'with_instruction')
    new_without_inst = sum(1 for item in new_data if item.get('type') == 'without_instruction')
    print(f"  - 有指令: {new_with_inst}")
    print(f"  - 无指令: {new_without_inst}")
    
    # 检查新版本是否有 stats 字段
    has_stats = sum(1 for item in new_data if 'stats' in item)
    print(f"\n新版本包含统计信息的样本: {has_stats}/{len(new_data)}")

def main():
    new_file = "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions_v2.jsonl"
    
    print("="*80)
    print("验证改进版过拟合训练集 (v2)")
    print("="*80)
    
    data = load_training_data(new_file)
    verify_samples(data, n_samples=3)
    compare_with_old_version()
    
    print(f"\n{'='*80}")
    print("验证完成!")
    print(f"{'='*80}")
    print(f"\n改进版训练集路径:")
    print(f"  {new_file}")
    print(f"\n难负样本定义:")
    print(f"  - og query 中 score >= 1 (相关)")
    print(f"  - changed query 中 score = 0 (不相关)")

if __name__ == "__main__":
    main()
