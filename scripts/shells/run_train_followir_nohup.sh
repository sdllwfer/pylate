#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate
# 设置无缓冲输出
export PYTHONUNBUFFERED=1
# ============================================================
# ColBERT FollowIR 后台训练脚本
# ============================================================
# 使用 nohup 在后台运行训练任务，支持多组实验并行
# ============================================================
# ========== 配置参数 ==========
# 模型路径 (基座模型)
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
# 训练数据路径
TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/colbert_train_final.jsonl"
# 输出目录 (基础目录，会在此目录下创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train"
CUSTOM_OUTPUT_PATH=""
NOTE=""
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
# 创建日志目录
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
# 日志文件名
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "============================================================"
echo "📊 ColBERT FollowIR 后台训练"
echo "============================================================"
echo "模型: ${MODEL_NAME}"
echo "训练数据: ${TRAIN_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${LOG_FILE}"
echo "GPU: cuda:${CUDA_VISIBLE_DEVICES}"
echo "批次大小: ${BATCH_SIZE}"
echo "学习率: ${LEARNING_RATE}"
echo "训练轮数: ${NUM_EPOCHS}"
[ -n "${RESUME_FROM_CHECKPOINT}" ] && echo "从检查点继续: ${RESUME_FROM_CHECKPOINT}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"
# 构建命令
CMD="cd /home/luwa/Documents/pylate && "
CMD="${CMD} conda run -n pylate python -u scripts/training/train_followir.py "
CMD="${CMD} --model_name '${MODEL_NAME}' "
CMD="${CMD} --train_data '${TRAIN_DATA}' "
CMD="${CMD} --output_dir '${OUTPUT_DIR}' "
CMD="${CMD} --num_epochs ${NUM_EPOCHS} "
CMD="${CMD} --batch_size ${BATCH_SIZE} "
CMD="${CMD} --learning_rate ${LEARNING_RATE} "
CMD="${CMD} --device 'cuda:${CUDA_VISIBLE_DEVICES}' "
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    CMD="${CMD} --resume_from_checkpoint '${RESUME_FROM_CHECKPOINT}' "
fi
if [ -n "${NOTE}" ]; then
    CMD="${CMD} --note \"${NOTE}\" "
fi
# 清理之前的进程 (排除当前脚本进程)
pkill -f "train_followir.py" --older-than 5s 2>/dev/null
sleep 2
# 使用 tee 同时输出到终端和日志文件
nohup bash -c "${CMD} 2>&1 | tee -a ${LOG_FILE}" > /dev/null 2>&1 &
PID=$!
# 保存 PID
echo $PID > "${OUTPUT_DIR}/train.pid"
echo "🚀 训练任务已启动!"
echo "📝 进程 PID: ${PID}"
echo "📁 日志文件: ${LOG_FILE}"
echo ""
echo "========== 查看日志命令 =========="
echo "实时查看日志: tail -f ${LOG_FILE}"
echo "查看最后100行: tail -n 100 ${LOG_FILE}"
echo "=================================="
echo ""
echo "========== 管理进程命令 =========="
echo "查看运行中的训练: ps aux | grep train_followir"
echo "终止训练进程: kill ${PID}"
echo "强制终止: kill -9 ${PID}"
echo "=================================="
