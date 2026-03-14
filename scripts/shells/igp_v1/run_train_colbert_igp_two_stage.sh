#!/bin/bash

# 设置 CUDA 设备（在 conda activate 之前）
export CUDA_VISIBLE_DEVICES="1"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 调试：打印 CUDA 环境变量
echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# ============================================================
# ColBERT-IGP 两阶段训练脚本
# 阶段1: 在短数据集上训练（学习基本指令识别能力）
# 阶段2: 在长数据集上微调（适应复杂真实场景）
# ============================================================

# ========== 配置参数 ==========
# 模型路径 (基座模型)
MODEL_NAME="lightonai/ColBERT-Zero"

# 阶段1: 短数据集（有明确 instruction 字段）
# SHORT_TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl"
# SHORT_TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/two_stage_mixed_v2/phase1_short_long_mixed.jsonl"
SHORT_TRAIN_DATA="dataset/colbert_data/mixed_short_long/mixed_short50%_long50%.jsonl"
SHORT_EPOCHS=8
SHORT_BATCH_SIZE=256

# 阶段2: 长数据集（端到端学习）
LONG_TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/train_data_igp.jsonl"
# LONG_TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions.jsonl"
# LONG_TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/two_stage_mixed_v2/phase2_long_only.jsonl"
LONG_EPOCHS=30
LONG_BATCH_SIZE=128

# 输出目录
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令"
NOTE="v1-两阶段训练-短指令增强数据集先训练，再单纯长数据集-MAX_RATIO=0.2-数据增强加上无指令版本-动态门控V3-修复显存差距过大的原因-probe增加为三层-根据查询是否包含指令（有监督信号）自动调整门控强度-发现之前的长指令超过配置32了-stage2设置为长指令重新训练"

# ============================
# IGP 模块参数
# ============================
ENABLE_PROBE=true
ENABLE_ADAPTER=true
ENABLE_GATE=true
MAX_RATIO=0.2
BOTTLENECK_DIM=128
PROBE_NUM_LAYERS=3
AUX_LOSS_WEIGHT=0
LOG_INTERVAL=10

# ============================
# 训练参数
# ============================
# 阶段1学习率（短数据集）
SHORT_BASE_LR=2e-5
SHORT_GATE_LR=5e-2

# 阶段2学习率（长数据集）- 使用更小学习率微调
LONG_BASE_LR=1e-5
LONG_GATE_LR=2e-2

# 早停配置
PHASE2_PATIENCE=20
PHASE2_EARLY_STOP_THRESHOLD=0.0001

# 验证集比例
EVAL_RATIO=0.05

# 自动生成时间戳
TIMESTAMP=$(date +%m%d%H%M%S)

# 确定最终输出路径
if [ -n "${CUSTOM_OUTPUT_PATH}" ]; then
    OUTPUT_DIR="${CUSTOM_OUTPUT_PATH}"
    echo "📂 使用自定义输出路径: ${OUTPUT_DIR}"
else
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}"
    echo "📂 使用时间戳输出路径: ${OUTPUT_DIR}"
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "🚀 ColBERT-IGP 两阶段训练脚本"
echo "============================================================"
echo ""
echo "📊 阶段1: 短数据集训练"
echo "   数据: ${SHORT_TRAIN_DATA}"
echo "   Epochs: ${SHORT_EPOCHS}"
echo "   Batch Size: ${SHORT_BATCH_SIZE}"
echo "   Base LR: ${SHORT_BASE_LR}"
echo "   Gate LR: ${SHORT_GATE_LR}"
echo ""
echo "📊 阶段2: 长数据集微调"
echo "   数据: ${LONG_TRAIN_DATA}"
echo "   Epochs: ${LONG_EPOCHS}"
echo "   Batch Size: ${LONG_BATCH_SIZE}"
echo "   Base LR: ${LONG_BASE_LR}"
echo "   Gate LR: ${LONG_GATE_LR}"
echo ""
echo "============================================================"

# ============================
# 阶段1: 在短数据集上训练
# ============================
echo ""
echo "============================================================"
echo "📚 阶段1: 在短数据集上训练"
echo "============================================================"

STAGE1_OUTPUT="${OUTPUT_DIR}/stage1_short_data"
mkdir -p "${STAGE1_OUTPUT}"

cd /home/luwa/Documents/pylate

python scripts/training/train_colbert_igp.py \
    --model_name "${MODEL_NAME}" \
    --train_data "${SHORT_TRAIN_DATA}" \
    --output_dir "${STAGE1_OUTPUT}" \
    --phase1_epochs 0 \
    --phase2_epochs ${SHORT_EPOCHS} \
    --batch_size ${SHORT_BATCH_SIZE} \
    --eval_ratio ${EVAL_RATIO} \
    --base_lr ${SHORT_BASE_LR} \
    --gate_lr ${SHORT_GATE_LR} \
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
    --note "${NOTE}-Stage1-ShortData" \
    2>&1 | tee "${STAGE1_OUTPUT}/training_stage1.log"

STAGE1_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "[DEBUG] ============================================"
echo "[DEBUG] 阶段1已结束"
echo "[DEBUG] 退出码: ${STAGE1_EXIT_CODE}"
echo "[DEBUG] ============================================"

if [ ${STAGE1_EXIT_CODE} -ne 0 ]; then
    echo "❌ 阶段1训练失败，退出码: ${STAGE1_EXIT_CODE}"
    exit ${STAGE1_EXIT_CODE}
fi

echo ""
echo "✅ 阶段1训练完成"
echo ""

# 验证阶段1输出
if [ ! -d "${STAGE1_OUTPUT}/phase2" ]; then
    echo "❌ 错误: 阶段1输出目录不存在: ${STAGE1_OUTPUT}/phase2"
    exit 1
fi

# 查找阶段1的模型
STAGE1_MODEL_COUNT=$(find "${STAGE1_OUTPUT}" -type d \( -name "best_model_*" -o -name "final_model_*" \) | wc -l)
if [ ${STAGE1_MODEL_COUNT} -eq 0 ]; then
    echo "❌ 错误: 阶段1没有保存任何模型"
    exit 1
fi

echo "📊 阶段1已保存 ${STAGE1_MODEL_COUNT} 个模型"
echo ""
echo "[DEBUG] 等待2秒后启动阶段2..."
sleep 2
echo ""

# ============================
# 阶段2: 在长数据集上微调
# ============================
echo "============================================================"
echo "📚 阶段2: 在长数据集上微调"
echo "============================================================"

STAGE2_OUTPUT="${OUTPUT_DIR}/stage2_long_data"
mkdir -p "${STAGE2_OUTPUT}"

# 从阶段1加载最佳模型
STAGE1_BEST_MODEL="${STAGE1_OUTPUT}/phase2/best_model_*"
if ls ${STAGE1_BEST_MODEL} 1> /dev/null 2>&1; then
    STAGE1_BEST_MODEL_PATH=$(ls -d ${STAGE1_BEST_MODEL} | head -1)
    echo "📂 从阶段1加载最佳模型: ${STAGE1_BEST_MODEL_PATH}"
    MODEL_NAME="${STAGE1_BEST_MODEL_PATH}"
else
    echo "⚠️ 未找到阶段1的最佳模型，使用原始模型"
fi

python scripts/training/train_colbert_igp.py \
    --model_name "${MODEL_NAME}" \
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

STAGE2_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "[DEBUG] ============================================"
echo "[DEBUG] 阶段2已结束"
echo "[DEBUG] 退出码: ${STAGE2_EXIT_CODE}"
echo "[DEBUG] ============================================"

if [ ${STAGE2_EXIT_CODE} -ne 0 ]; then
    echo "❌ 阶段2训练失败，退出码: ${STAGE2_EXIT_CODE}"
    exit ${STAGE2_EXIT_CODE}
fi

echo ""
echo "✅ 阶段2训练完成"
echo ""

# ============================
# 训练完成总结
# ============================
echo "============================================================"
echo "🎉 两阶段训练全部完成!"
echo "============================================================"
echo ""
echo "📊 阶段1输出: ${STAGE1_OUTPUT}"
echo "📊 阶段2输出: ${STAGE2_OUTPUT}"
echo ""
echo "============================================================"
