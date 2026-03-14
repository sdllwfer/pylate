"""
将V2版本的无指令数据集与原始有指令数据集混合

混合策略：
1. 读取原始有指令数据集
2. 读取V2版本的无指令数据集（已去除instruction字段）
3. 按一定比例混合两个数据集
4. 保存为新的混合数据集
"""

import json
import random
import argparse
from pathlib import Path


def create_mixed_dataset_v2(
    instruction_path: str,
    no_instruction_path: str,
    output_path: str,
    no_instruction_ratio: float = 0.3,
    shuffle: bool = True
):
    """
    创建混合数据集V2
    
    Args:
        instruction_path: 原始有指令数据集路径
        no_instruction_path: V2版本无指令数据集路径
        output_path: 输出混合数据集路径
        no_instruction_ratio: 无指令样本比例（0-1之间）
        shuffle: 是否随机打乱顺序
    """
    print(f"📂 读取有指令数据集: {instruction_path}")
    instruction_samples = []
    with open(instruction_path, 'r', encoding='utf-8') as f:
        for line in f:
            instruction_samples.append(json.loads(line.strip()))
    print(f"   共读取 {len(instruction_samples)} 个有指令样本")
    
    print(f"\n📂 读取无指令数据集(V2): {no_instruction_path}")
    no_instruction_samples = []
    with open(no_instruction_path, 'r', encoding='utf-8') as f:
        for line in f:
            no_instruction_samples.append(json.loads(line.strip()))
    print(f"   共读取 {len(no_instruction_samples)} 个无指令样本")
    
    # 计算混合数量
    total_target = len(instruction_samples)
    no_instr_count = int(total_target * no_instruction_ratio)
    instr_count = total_target - no_instr_count
    
    print(f"\n📊 混合策略:")
    print(f"   - 目标总样本数: {total_target}")
    print(f"   - 有指令样本: {instr_count} ({instr_count/total_target*100:.1f}%)")
    print(f"   - 无指令样本: {no_instr_count} ({no_instr_count/total_target*100:.1f}%)")
    
    # 采样
    if len(instruction_samples) >= instr_count:
        selected_instr = random.sample(instruction_samples, instr_count)
    else:
        selected_instr = instruction_samples
        print(f"   ⚠️ 警告: 有指令样本不足，使用全部 {len(instruction_samples)} 个")
    
    if len(no_instruction_samples) >= no_instr_count:
        selected_no_instr = random.sample(no_instruction_samples, no_instr_count)
    else:
        selected_no_instr = no_instruction_samples
        print(f"   ⚠️ 警告: 无指令样本不足，使用全部 {len(no_instruction_samples)} 个")
    
    # 合并
    mixed_samples = selected_instr + selected_no_instr
    
    # 打乱顺序
    if shuffle:
        random.shuffle(mixed_samples)
    
    # 保存
    print(f"\n💾 保存混合数据集到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in mixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"   共生成 {len(mixed_samples)} 个混合样本")
    
    # 统计
    instr_in_mix = sum(1 for s in mixed_samples if 'instruction' in s)
    no_instr_in_mix = len(mixed_samples) - instr_in_mix
    
    print(f"\n📈 最终统计:")
    print(f"   - 总样本数: {len(mixed_samples)}")
    print(f"   - 有指令样本: {instr_in_mix} ({instr_in_mix/len(mixed_samples)*100:.1f}%)")
    print(f"   - 无指令样本: {no_instr_in_mix} ({no_instr_in_mix/len(mixed_samples)*100:.1f}%)")
    
    # 检查样本示例
    print(f"\n🔍 样本示例:")
    has_instr = [s for s in mixed_samples if 'instruction' in s][0]
    no_instr = [s for s in mixed_samples if 'instruction' not in s][0]
    
    print(f"\n有指令样本:")
    print(f"  查询: {has_instr['query'][:60]}...")
    print(f"  指令: {has_instr.get('instruction', 'N/A')[:60]}...")
    print(f"  正样本数: {len(has_instr.get('pos', []))}")
    print(f"  负样本数: {len(has_instr.get('neg', []))}")
    
    print(f"\n无指令样本:")
    print(f"  查询: {no_instr['query'][:60]}...")
    print(f"  正样本数: {len(no_instr.get('pos', []))}")
    print(f"  负样本数: {len(no_instr.get('neg', []))}")


def main():
    parser = argparse.ArgumentParser(description='创建混合数据集V2')
    parser.add_argument('--instruction', type=str,
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl',
                        help='有指令数据集路径')
    parser.add_argument('--no-instruction', type=str,
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/no_instruction_train_v2.jsonl',
                        help='无指令数据集V2路径')
    parser.add_argument('--output', type=str,
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/mixed_train_v2.jsonl',
                        help='输出混合数据集路径')
    parser.add_argument('--ratio', type=float, default=0.3,
                        help='无指令样本比例 (0-1)')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='不打乱顺序')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    create_mixed_dataset_v2(
        args.instruction,
        args.no_instruction,
        args.output,
        args.ratio,
        not args.no_shuffle
    )
    
    print("\n✅ 混合数据集V2生成完成！")


if __name__ == '__main__':
    main()
