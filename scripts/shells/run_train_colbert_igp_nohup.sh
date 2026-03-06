#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate
# ColBERT-IGP 两阶段训练 (后台运行版本)
# 需配置以下参数即可运行
# ========== 配置参数 ==========
# 确保环境干净
unset CUDA_VISIBLE_DEVICES
# 模型路径 (基座模型)
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
# 训练数据路径
TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl"
# 输出目录 (基础目录，会在此目录下创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train"
CUSTOM_OUTPUT_PATH=""
NOTE="短指令IGP架构两阶段训练"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="1"
# ============================
# IGP 模块参数
# ============================
ENABLE_PROBE=true
ENABLE_ADAPTER=true
ENABLE_GATE=true
MAX_RATIO=0.2
BOTTLENECK_DIM=64
AUX_LOSS_WEIGHT=0.1
# ============================
# 训练参数
# ============================
PHASE1_EPOCHS=2
PHASE2_EPOCHS=3
BATCH_SIZE=16
BASE_LR=1e-5
GATE_LR=1e-2
# Phase 2 早停配置
PHASE2_PATIENCE=5
PHASE2_EARLY_STOP_THRESHOLD=0.005
# 检查点保存配置 (设为空则自动使用 epoch 总数)
SAVE_TOTAL_LIMIT=""
# Phase 1 检查点路径 (为空则从头训练)
PHASE1_CHECKPOINT=""
# Phase 2 检查点路径 (为空则从头训练)
PHASE2_CHECKPOINT=""
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
# 日志文件
LOG_FILE="${OUTPUT_DIR}/train_log.txt"
echo "============================================================"
echo "📊 ColBERT-IGP 两阶段训练 (后台运行)"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "训练数据: ${TRAIN_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "GPU: cuda:${CUDA_VISIBLE_DEVICES}"
echo "日志文件: ${LOG_FILE}"
echo "--- IGP 模块配置 ---"
echo "启用 Probe: ${ENABLE_PROBE}, Adapter: ${ENABLE_ADAPTER}, Gate: ${ENABLE_GATE}"
echo "门控最大比率: ${MAX_RATIO}, Adapter瓶颈: ${BOTTLENECK_DIM}"
echo "--- 训练参数 ---"
echo "Phase 1: ${PHASE1_EPOCHS} epochs, Phase 2: ${PHASE2_EPOCHS} epochs"
echo "批次大小: ${BATCH_SIZE}, 基础LR: ${BASE_LR}, 门控LR: ${GATE_LR}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"
# 使用 accelerate launch 在后台运行训练
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
cd /home/luwa/Documents/pylate
setsid bash -c "
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    scripts/training/train_colbert_igp.py \
    --model_name '${MODEL_NAME}' \
    --train_data '${TRAIN_DATA}' \
    --output_dir '${OUTPUT_DIR}' \
    --device 'cuda:${CUDA_VISIBLE_DEVICES}' \
    --phase1_epochs ${PHASE1_EPOCHS} \
    --phase2_epochs ${PHASE2_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --base_lr ${BASE_LR} \
    --gate_lr ${GATE_LR} \
    --enable_probe ${ENABLE_PROBE} \
    --enable_adapter ${ENABLE_ADAPTER} \
    --enable_gate ${ENABLE_GATE} \
    --max_ratio ${MAX_RATIO} \
    --bottleneck_dim ${BOTTLENECK_DIM} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --phase2_patience ${PHASE2_PATIENCE} \
    --phase2_early_stop_threshold ${PHASE2_EARLY_STOP_THRESHOLD} \
    $([ -n "${SAVE_TOTAL_LIMIT}" ] && echo "--save_total_limit ${SAVE_TOTAL_LIMIT}") \
    $([ -n "${PHASE1_CHECKPOINT}" ] && echo "--phase1_checkpoint ${PHASE1_CHECKPOINT}") \
    $([ -n "${PHASE2_CHECKPOINT}" ] && echo "--phase2_checkpoint ${PHASE2_CHECKPOINT}") \
    --note '${NOTE}'
" > "${LOG_FILE}" 2>&1 &
PID=$!
# 保存 PID
echo $PID > "${OUTPUT_DIR}/train.pid"
echo "🚀 训练已在后台启动"
echo "📝 PID: ${PID}"
echo "� 输出目录: ${OUTPUT_DIR}"
echo "📄 日志文件: ${LOG_FILE}"
echo ""
echo "查看日志: tail -f ${LOG_FILE}"
echo "停止训练: kill ${PID}"
