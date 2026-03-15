#!/bin/bash

# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES="1"

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 验证环境
if [ "$CONDA_DEFAULT_ENV" != "pylate" ]; then
    echo "❌ 错误: 无法激活 pylate 环境"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"
echo "✅ Python 路径: $(which python)"

# ============================================================
# ColBERT-IGP 过拟合训练脚本 (使用 v3 数据集)
# 使用改进的难负样本数据集进行训练
# ============================================================

# ========== 配置参数 ==========
# 模型路径 (基座模型)
MODEL_NAME="lightonai/ColBERT-Zero"

# 训练数据 (v3 版本 - 改进的难负样本)
TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/overfit_test_data/train_overfit_mixed_instructions_v3.jsonl"

# 训练参数
EPOCHS=50
BATCH_SIZE=128
BASE_LR=1e-5
GATE_LR=2e-2

# 输出目录
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train"
TIMESTAMP=$(date +%m%d%H%M%S)
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}_overfit_v3"

# 备注
NOTE="过拟合训练-v3数据集-改进难负样本-正负比例1:5-基于真实相似度分数"

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
# 早停配置
# ============================
PATIENCE=15
EARLY_STOP_THRESHOLD=0.0001

# 验证集比例
EVAL_RATIO=0.1

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

echo ""
echo "============================================================"
echo "🚀 ColBERT-IGP 过拟合训练 (v3 数据集)"
echo "============================================================"
echo ""
echo "📊 训练配置:"
echo "   数据: ${TRAIN_DATA}"
echo "   Epochs: ${EPOCHS}"
echo "   Batch Size: ${BATCH_SIZE}"
echo "   Base LR: ${BASE_LR}"
echo "   Gate LR: ${GATE_LR}"
echo "   输出目录: ${OUTPUT_DIR}"
echo ""
echo "🔧 IGP 配置:"
echo "   Probe: ${ENABLE_PROBE}"
echo "   Adapter: ${ENABLE_ADAPTER}"
echo "   Gate: ${ENABLE_GATE}"
echo "   Max Ratio: ${MAX_RATIO}"
echo "   Probe Layers: ${PROBE_NUM_LAYERS}"
echo ""
echo "============================================================"

# 检查数据文件是否存在
if [ ! -f "${TRAIN_DATA}" ]; then
    echo "❌ 错误: 数据文件不存在: ${TRAIN_DATA}"
    exit 1
fi

# 统计数据
NUM_SAMPLES=$(wc -l < "${TRAIN_DATA}")
echo "📊 训练样本数: ${NUM_SAMPLES}"
echo ""

cd /home/luwa/Documents/pylate

# 启动训练
python scripts/training/train_colbert_igp.py \
    --model_name "${MODEL_NAME}" \
    --train_data "${TRAIN_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --phase1_epochs 0 \
    --phase2_epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --eval_ratio ${EVAL_RATIO} \
    --base_lr ${BASE_LR} \
    --gate_lr ${GATE_LR} \
    --enable_phase1 false \
    --enable_phase2 true \
    --enable_probe ${ENABLE_PROBE} \
    --enable_adapter ${ENABLE_ADAPTER} \
    --enable_gate ${ENABLE_GATE} \
    --max_ratio ${MAX_RATIO} \
    --bottleneck_dim ${BOTTLENECK_DIM} \
    --probe_num_layers ${PROBE_NUM_LAYERS} \
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --phase2_patience ${PATIENCE} \
    --phase2_early_stop_threshold ${EARLY_STOP_THRESHOLD} \
    --log_interval ${LOG_INTERVAL} \
    --device "cuda:0" \
    --note "${NOTE}" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "🎉 训练完成!"
else
    echo "❌ 训练失败，退出码: ${EXIT_CODE}"
fi
echo "============================================================"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
