#!/bin/bash

# 只运行 Stage 2 的脚本（从 Stage 1 的模型继续）

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES="1"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# ========== 配置参数 ==========
# Stage 1 保存的最佳模型
STAGE1_MODEL="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令/stage1_short_data/phase2/best_model_phase2_20260314_134029"

# Stage 2: 长数据集
LONG_TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/train_data_igp.jsonl"
# 训练控制：可以设置 EPOCHS 或 MAX_STEPS（优先级：MAX_STEPS > EPOCHS）
# 如果设置了 PHASE2_MAX_STEPS，则忽略 PHASE2_EPOCHS
LONG_EPOCHS=30
PHASE2_MAX_STEPS=5000  # 增加训练步数从 2000 到 5000
LONG_BATCH_SIZE=64  # 减小 batch size 从 128 到 64 以避免显存不足

# 输出目录
OUTPUT_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令"
NOTE="v1-两阶段训练-Stage2从Stage1最佳模型继续-MAX_RATIO=0.2-动态门控V3-优化版"

# IGP 模块参数
ENABLE_PROBE=true
ENABLE_ADAPTER=true
ENABLE_GATE=true
MAX_RATIO=0.2
BOTTLENECK_DIM=128
PROBE_NUM_LAYERS=3
AUX_LOSS_WEIGHT=0.1  # 增加辅助损失权重从 0 到 0.1
LOG_INTERVAL=10

# Stage 2 学习率
LONG_BASE_LR=1e-5
LONG_GATE_LR=1e-3  # 降低 Gate 学习率从 2e-2 到 1e-3

# 早停配置
PHASE2_PATIENCE=20
PHASE2_EARLY_STOP_THRESHOLD=0.0001

# 验证集比例
EVAL_RATIO=0.05

STAGE2_OUTPUT="${OUTPUT_DIR}/stage2_long_data"
mkdir -p "${STAGE2_OUTPUT}"

echo "============================================================"
echo "🚀 Stage 2: 从 Stage 1 最佳模型继续训练"
echo "============================================================"
echo ""
echo "📂 Stage 1 模型: ${STAGE1_MODEL}"
echo "📊 Stage 2 数据: ${LONG_TRAIN_DATA}"
if [ -n "${PHASE2_MAX_STEPS}" ] && [ "${PHASE2_MAX_STEPS}" != "null" ]; then
    echo "📊 Max Steps: ${PHASE2_MAX_STEPS} (按步数训练)"
else
    echo "📊 Epochs: ${LONG_EPOCHS} (按轮数训练)"
fi
echo "📊 Batch Size: ${LONG_BATCH_SIZE}"
echo "============================================================"

cd /home/luwa/Documents/pylate

python scripts/training/train_colbert_igp.py \
    --model_name "${STAGE1_MODEL}" \
    --train_data "${LONG_TRAIN_DATA}" \
    --output_dir "${STAGE2_OUTPUT}" \
    --phase1_epochs 0 \
    --phase2_epochs ${LONG_EPOCHS} \
    --phase2_max_steps ${PHASE2_MAX_STEPS} \
    --batch_size ${LONG_BATCH_SIZE} \
    --eval_ratio ${EVAL_RATIO} \
    --base_lr ${LONG_BASE_LR} \
    --gate_lr ${LONG_GATE_LR} \
    --enable_phase1 false \
    --enable_phase2 true \
    --enable_probe ${ENABLE_PROBE} \
    --enable_adapter ${ENABLE_ADAPTER} \
    --enable_gate ${ENABLE_GATE} \
    --max_ratio ${MAX_RATIO} \
    --bottleneck_dim ${BOTTLENECK_DIM} \
    --probe_num_layers ${PROBE_NUM_LAYERS} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --phase2_patience ${PHASE2_PATIENCE} \
    --phase2_early_stop_threshold ${PHASE2_EARLY_STOP_THRESHOLD} \
    --log_interval ${LOG_INTERVAL} \
    --device "cuda:0" \
    --note "${NOTE}-Stage2-LongData" \
    2>&1 | tee "${STAGE2_OUTPUT}/training_stage2.log"

echo ""
echo "✅ Stage 2 训练完成"
echo "============================================================"
