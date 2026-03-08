#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# ColBERT FollowIR 微调脚本
# 只需配置以下参数即可运行

# ========== 配置参数 ==========
# 模型路径 (基座模型)
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"

# 训练数据路径
TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/colbert_train_final.jsonl"

# 输出目录 (基础目录，会在此目录下创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_finetune_followir"
CUSTOM_OUTPUT_PATH=""

NOTE="短指令训练的最佳模型，在FollowIR数据集上训一下看看能不能提升效果"

# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="0"

# 训练参数
NUM_EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=1e-5

# 从检查点继续训练 (填检查点路径，如: /path/to/checkpoint-405，设为空字符串或不填则从头训练)
RESUME_FROM_CHECKPOINT=""

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
echo "📊 ColBERT FollowIR 微调"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "训练数据: ${TRAIN_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "GPU: cuda:${CUDA_VISIBLE_DEVICES}"
echo "批次大小: ${BATCH_SIZE}"
echo "学习率: ${LEARNING_RATE}"
echo "训练轮数: ${NUM_EPOCHS}"
[ -n "${RESUME_FROM_CHECKPOINT}" ] && echo "从检查点继续: ${RESUME_FROM_CHECKPOINT}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python scripts/training/train_followir.py \
    --model_name '${MODEL_NAME}' \
    --train_data '${TRAIN_DATA}' \
    --output_dir '${OUTPUT_DIR}' \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --device 'cuda:${CUDA_VISIBLE_DEVICES}'"

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    CMD="${CMD} --resume_from_checkpoint '${RESUME_FROM_CHECKPOINT}'"
fi

if [ -n "${NOTE}" ]; then
    CMD="${CMD} --note \"${NOTE}\""
fi

eval ${CMD}

echo ""
echo "============================================================"
echo "✅ 训练完成!"
echo "📁 模型目录: ${OUTPUT_DIR}"
echo "============================================================"
