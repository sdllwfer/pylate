#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate
# 设置无缓冲输出
export PYTHONUNBUFFERED=1
# ============================================================
# ColBERT-IGP 两阶段训练脚本
# ============================================================
# 使用方法：
#   1. 确保 conda 环境已激活: conda activate pylate
#   2. 运行脚本: bash scripts/run_train_colbert_igp.sh
#
# 注意：由于已激活环境，脚本中直接使用 python 命令
# ============================================================
# 需配置以下参数即可运行
# ========== 配置参数 ==========
# 模型路径 (基座模型)
# MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
MODEL_NAME="lightonai/ColBERT-Zero"
# 训练数据路径
# TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/colbert_train_final.jsonl"
TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl"
# 输出目录 (基础目录，会在此目录下创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/output/colbert_igp_train/col_phase2_maxratio0.2-bs64"
NOTE="ColBERT-Zero-门控最大比例0.2-bs64"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="0"
# ============================
# IGP 模块参数
# ============================
# 是否启用 InstructionProbe (true/false)
ENABLE_PROBE=true
# 是否启用 IGPAdapter (true/false)  
ENABLE_ADAPTER=true
# 是否启用 RatioGate (true/false)
ENABLE_GATE=true
# 门控最大比率 (防止指令破坏原语义，建议 0.1-0.3)
MAX_RATIO=0.2
# Adapter 瓶颈维度 (建议 32-128)
BOTTLENECK_DIM=128
# 辅助损失权重
AUX_LOSS_WEIGHT=0
# 损失记录间隔 (每多少个step记录一次，用于生成更细致的损失曲线，设为0则记录每个step)
LOG_INTERVAL=10
# ============================
# 训练参数
# ============================
# 是否启用 Phase 1 (Probe Warm-up)
ENABLE_PHASE1=false
# 是否启用 Phase 2 (Joint Training)
ENABLE_PHASE2=true
# Phase 1 检查点路径 (为空则从头训练)
# PHASE1_CHECKPOINT="/home/luwa/Documents/pylate/output/colbert_igp_train/short_instruction/phase1/best_model_phase1_20260228_224521"
PHASE1_CHECKPOINT=""
# Phase 2 检查点路径 (为空则从头训练)
PHASE2_CHECKPOINT=""
# Phase 1 训练轮数 (Probe Warm-up)
PHASE1_EPOCHS=10
# Phase 2 训练轮数 (Joint Training)
PHASE2_EPOCHS=100
# 批次大小
BATCH_SIZE=64
# 验证集比例
EVAL_RATIO=0.01
# 基础学习率 (BERT/Probe/Adapter) - 增加学习率以加速收敛
BASE_LR=2e-5
# 门控学习率 (建议 1e-2 到 1e-1，确保门控激活)
GATE_LR=5e-2
# Phase 2 早停配置 - 降低阈值让训练更充分
PHASE2_PATIENCE=20
PHASE2_EARLY_STOP_THRESHOLD=0.0001
# 检查点保存配置 (设为空则自动使用 epoch 总数)
SAVE_TOTAL_LIMIT=""
# ==============================
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
echo "📊 ColBERT-IGP 两阶段训练"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "训练数据: ${TRAIN_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "GPU: cuda:${CUDA_VISIBLE_DEVICES}"
echo ""
echo "--- IGP 模块配置 ---"
echo "启用 Probe: ${ENABLE_PROBE}"
echo "启用 Adapter: ${ENABLE_ADAPTER}"
echo "启用 Gate: ${ENABLE_GATE}"
echo "门控最大比率: ${MAX_RATIO}"
echo "Adapter瓶颈维度: ${BOTTLENECK_DIM}"
echo "辅助损失权重: ${AUX_LOSS_WEIGHT}"
echo ""
echo "--- 训练参数 ---"
echo "启用 Phase 1: ${ENABLE_PHASE1}"
echo "启用 Phase 2: ${ENABLE_PHASE2}"
echo "Phase 1 epochs: ${PHASE1_EPOCHS}"
echo "Phase 2 epochs: ${PHASE2_EPOCHS}"
echo "批次大小: ${BATCH_SIZE}"
echo "基础学习率: ${BASE_LR}"
echo "门控学习率: ${GATE_LR}"
[ -n "${PHASE1_CHECKPOINT}" ] && echo "Phase 1 检查点: ${PHASE1_CHECKPOINT}"
[ -n "${PHASE2_CHECKPOINT}" ] && echo "Phase 2 检查点: ${PHASE2_CHECKPOINT}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"
# 运行训练 (直接运行，因为环境已激活)
python -u scripts/training/train_colbert_igp.py \
    --model_name "${MODEL_NAME}" \
    --train_data "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${CUDA_VISIBLE_DEVICES}" \
    --phase1_epochs ${PHASE1_EPOCHS} \
    --phase2_epochs ${PHASE2_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --eval_ratio ${EVAL_RATIO} \
    --base_lr ${BASE_LR} \
    --gate_lr ${GATE_LR} \
    --enable_phase1 ${ENABLE_PHASE1} \
    --enable_phase2 ${ENABLE_PHASE2} \
    --enable_probe ${ENABLE_PROBE} \
    --enable_adapter ${ENABLE_ADAPTER} \
    --enable_gate ${ENABLE_GATE} \
    --max_ratio ${MAX_RATIO} \
    --bottleneck_dim ${BOTTLENECK_DIM} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --log_interval ${LOG_INTERVAL} \
    --phase2_patience ${PHASE2_PATIENCE} \
    --phase2_early_stop_threshold ${PHASE2_EARLY_STOP_THRESHOLD} \
    $([ -n "${SAVE_TOTAL_LIMIT}" ] && echo "--save_total_limit ${SAVE_TOTAL_LIMIT}") \
    $([ -n "${PHASE1_CHECKPOINT}" ] && echo "--phase1_checkpoint ${PHASE1_CHECKPOINT}") \
    $([ -n "${PHASE2_CHECKPOINT}" ] && echo "--phase2_checkpoint ${PHASE2_CHECKPOINT}") \
    --note "${NOTE}"
echo ""
echo "============================================================"
echo "✅ 训练完成!"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "============================================================"
