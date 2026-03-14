#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# ============================================================
# ColBERT-IGP V3 训练脚本 - 动态感知门控 + L1稀疏正则
# ============================================================

# ========== 配置参数 ==========
# 模型路径 (基座模型)
MODEL_NAME="lightonai/ColBERT-Zero"

# 训练数据
TRAIN_DATA="/home/luwa/Documents/pylate/dataset/colbert_data/igp_hard_synthetic_dataset/final_hard_easy_mixed_train_augmented_instrmask.jsonl"

# 训练轮数
EPOCHS=50
BATCH_SIZE=64

# 输出目录
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/output/colbert_igp_train/col_v1_max0.2_RatioGateV3_动态感知门控"
NOTE="v3-动态感知门控-L1稀疏正则-自动按需开启"

# GPU 设备编号
CUDA_VISIBLE_DEVICES="1"

# ============================
# IGP 模块参数
# ============================
ENABLE_PROBE=true
ENABLE_ADAPTER=true
ENABLE_GATE=true
MAX_RATIO=0.2
BOTTLENECK_DIM=128
AUX_LOSS_WEIGHT=0

# ============================
# 训练参数
# ============================
BASE_LR=2e-5
GATE_LR=5e-2

# L1稀疏正则化系数 (关键参数！)
# 0.01 = 默认，增大则门控更稀疏（更倾向于关闭）
GATE_L1_COEFF=0.01

# 早停配置
PHASE2_PATIENCE=20
PHASE2_EARLY_STOP_THRESHOLD=0.0001

# 验证集比例
EVAL_RATIO=0.05

# 日志间隔
LOG_INTERVAL=10

# 自动生成时间戳
TIMESTAMP=$(date +%m%d%H%M%S)

# 确定最终输出路径
if [ -n "${CUSTOM_OUTPUT_PATH}" ]; then
    OUTPUT_DIR="${CUSTOM_OUTPUT_PATH}"
    echo "📂 使用自定义输出路径: ${OUTPUT_DIR}"
else
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TIMESTAMP}_RatioGateV3"
    echo "📂 使用时间戳输出路径: ${OUTPUT_DIR}"
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "🚀 ColBERT-IGP V3 训练脚本 - 动态感知门控"
echo "============================================================"
echo ""
echo "📊 训练配置:"
echo "   数据: ${TRAIN_DATA}"
echo "   Epochs: ${EPOCHS}"
echo "   Batch Size: ${BATCH_SIZE}"
echo "   Base LR: ${BASE_LR}"
echo "   Gate LR: ${GATE_LR}"
echo "   L1 Coeff: ${GATE_L1_COEFF}"
echo ""
echo "🔧 IGP 配置:"
echo "   Gate类型: RatioGateV3 (动态感知门控)"
echo "   Max Ratio: ${MAX_RATIO}"
echo "   Bottleneck Dim: ${BOTTLENECK_DIM}"
echo ""
echo "============================================================"

cd /home/luwa/Documents/pylate

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
    --aux_loss_weight ${AUX_LOSS_WEIGHT} \
    --gate_l1_coeff ${GATE_L1_COEFF} \
    --phase2_patience ${PHASE2_PATIENCE} \
    --phase2_early_stop_threshold ${PHASE2_EARLY_STOP_THRESHOLD} \
    --log_interval ${LOG_INTERVAL} \
    --device "cuda:0" \
    --note "${NOTE}" \
    2>&1 | tee "${OUTPUT_DIR}/training.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "❌ 训练失败，退出码: ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi

echo ""
echo "============================================================"
echo "✅ 训练完成!"
echo "============================================================"
echo ""
echo "📊 输出目录: ${OUTPUT_DIR}"
echo ""
echo "🔍 检查 Gate 值分布:"
echo "   可以查看训练日志中的 'gate_ratio' 和 'gate_l1_loss' 字段"
echo "   理想的 Gate 值分布:"
echo "   - 有指令样本: 0.10-0.20 (高)"
echo "   - 无指令样本: 0.00-0.05 (低)"
echo ""
echo "============================================================"
