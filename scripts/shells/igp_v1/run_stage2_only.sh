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
LONG_EPOCHS=30
LONG_BATCH_SIZE=128

# 输出目录
OUTPUT_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令"
NOTE="v1-两阶段训练-Stage2从Stage1最佳模型继续-MAX_RATIO=0.2-动态门控V3"

# IGP 模块参数
ENABLE_PROBE=true
ENABLE_ADAPTER=true
ENABLE_GATE=true
MAX_RATIO=0.2
BOTTLENECK_DIM=128
PROBE_NUM_LAYERS=3
AUX_LOSS_WEIGHT=0
LOG_INTERVAL=10

# Stage 2 学习率
LONG_BASE_LR=1e-5
LONG_GATE_LR=2e-2

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
echo "📊 Epochs: ${LONG_EPOCHS}"
echo "📊 Batch Size: ${LONG_BATCH_SIZE}"
echo "============================================================"

cd /home/luwa/Documents/pylate

python scripts/training/train_colbert_igp.py \
    --model_name "${STAGE1_MODEL}" \
    --train_data "${LONG_TRAIN_DATA}" \
    --output_dir "${STAGE2_OUTPUT}" \
    --phase1_epochs 0 \
    --phase2_epochs ${LONG_EPOCHS} \
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
