#!/usr/bin/env python3
"""
准备两阶段训练数据集

Phase 1: 短指令 + 部分长指令（混合）
Phase 2: 剩余长指令（单独微调）

使用方法:
    python prepare_two_stage_data.py \
        --short_data /path/to/short_data.jsonl \
        --long_data /path/to/long_data.jsonl \
        --output_dir /path/to/output \
        --long_for_phase1_ratio 0.35  # 35%的长指令用于第一阶段
"""

import json
import argparse
import os
import random
from pathlib import Path


def load_jsonl(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """保存为 JSONL 文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_short_data(data):
    """
    处理短指令数据
    保持原有格式，确保有 instruction 字段
    """
    processed = []
    for item in data:
        processed.append({
            'query': item.get('query', ''),
            'instruction': item.get('instruction', ''),
            'pos': item.get('pos', []),
            'neg': item.get('neg', [])
        })
    return processed


def process_long_data(data):
    """
    处理长指令数据
    长指令数据的 instruction 隐藏在 query 中
    直接保留原格式，instruction 字段设为空（表示指令已包含在 query 中）
    """
    processed = []
    for item in data:
        processed.append({
            'query': item.get('query', ''),
            'instruction': '',  # 长指令的指令已包含在 query 中
            'pos': item.get('pos', []),
            'neg': item.get('neg', [])
        })
    return processed


def split_long_data(long_data, ratio_for_phase1):
    """
    将长指令数据划分为两部分

    Args:
        long_data: 长指令数据列表
        ratio_for_phase1: 用于第一阶段的比例

    Returns:
        (phase1_long, phase2_long)
    """
    total = len(long_data)
    phase1_size = int(total * ratio_for_phase1)

    # 随机打乱后分割
    shuffled = long_data.copy()
    random.shuffle(shuffled)

    phase1_long = shuffled[:phase1_size]
    phase2_long = shuffled[phase1_size:]

    return phase1_long, phase2_long


def main():
    parser = argparse.ArgumentParser(description='准备两阶段训练数据集')
    parser.add_argument('--short_data', type=str, required=True,
                        help='短指令数据集路径')
    parser.add_argument('--long_data', type=str, required=True,
                        help='长指令数据集路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--long_for_phase1_ratio', type=float, default=0.35,
                        help='用于第一阶段的长指令比例 (默认: 0.35)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)

    print(f"{'='*70}")
    print("准备两阶段训练数据集")
    print(f"{'='*70}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据
    print(f"\n加载短指令数据: {args.short_data}")
    short_data = load_jsonl(args.short_data)
    print(f"  短指令样本数: {len(short_data)}")

    print(f"\n加载长指令数据: {args.long_data}")
    long_data = load_jsonl(args.long_data)
    print(f"  长指令样本数: {len(long_data)}")

    # 处理短指令数据
    print(f"\n处理短指令数据...")
    short_processed = process_short_data(short_data)

    # 处理长指令数据
    print(f"处理长指令数据...")
    long_processed = process_long_data(long_data)

    # 分割长指令数据
    print(f"\n分割长指令数据 (Phase 1 比例: {args.long_for_phase1_ratio:.0%})...")
    phase1_long, phase2_long = split_long_data(long_processed, args.long_for_phase1_ratio)

    print(f"  用于 Phase 1 的长指令: {len(phase1_long)}")
    print(f"  用于 Phase 2 的长指令: {len(phase2_long)}")

    # 构造 Phase 1 数据集: 短指令 + 部分长指令
    phase1_data = short_processed + phase1_long
    print(f"\nPhase 1 数据集:")
    print(f"  短指令样本: {len(short_processed)}")
    print(f"  长指令样本: {len(phase1_long)}")
    print(f"  总计: {len(phase1_data)}")

    # 构造 Phase 2 数据集: 剩余长指令
    phase2_data = phase2_long
    print(f"\nPhase 2 数据集:")
    print(f"  长指令样本: {len(phase2_data)}")

    # 保存数据集
    phase1_path = os.path.join(args.output_dir, 'phase1_short_long_mixed.jsonl')
    phase2_path = os.path.join(args.output_dir, 'phase2_long_only.jsonl')

    print(f"\n保存数据集...")
    save_jsonl(phase1_data, phase1_path)
    print(f"  Phase 1: {phase1_path}")

    save_jsonl(phase2_data, phase2_path)
    print(f"  Phase 2: {phase2_path}")

    # 保存统计信息
    stats = {
        'short_data_total': len(short_data),
        'long_data_total': len(long_data),
        'long_for_phase1': len(phase1_long),
        'long_for_phase2': len(phase2_long),
        'phase1_total': len(phase1_data),
        'phase2_total': len(phase2_data),
        'phase1_short_ratio': len(short_processed) / len(phase1_data),
        'phase1_long_ratio': len(phase1_long) / len(phase1_data),
    }

    stats_path = os.path.join(args.output_dir, 'stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  统计信息: {stats_path}")

    print(f"\n{'='*70}")
    print("数据集准备完成!")
    print(f"{'='*70}")
    print(f"\n使用方式:")
    print(f"  Phase 1 训练: 使用 {phase1_path}")
    print(f"  Phase 2 训练: 使用 {phase2_path}")


if __name__ == '__main__':
    main()
