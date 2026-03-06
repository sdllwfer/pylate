#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate
# FollowIR 评测脚本
# 整合了重排和评测过程，只需配置以下参数即可运行
# ========== 配置参数 ==========
# 模型路径
# MODEL_PATH="lightonai/GTE-ModernColBERT-v1"
MODEL_PATH="/home/luwa/Documents/pylate/output/colbert_finetune_followir/短指令检查点训FollowIR数据集/best_model"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="1"
# 要评测的数据集 (可用: Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval)
# 设为空字符串或注释掉则评测全部三个数据集
# TASKS=("Core17InstructionRetrieval")
TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
# 输出目录 (会自动创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/evaluation_data/followir"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/evaluation_data/followir/test"
# 自定义输出路径 (可选)
NOTE='短指令训练后再在FL数据集上训练测试指标'
# ==============================
# 从模型路径中提取时间戳 (模型路径格式: .../colbert_finetune_followir/时间戳/best_model)
# 如果模型路径不包含时间戳（如 HuggingFace 模型名），则使用默认前缀
if [[ "${MODEL_PATH}" =~ [0-9]{6,} ]]; then
    MODEL_TIMESTAMP=$(echo "${MODEL_PATH}" | grep -oE '[0-9]{6,}' | head -1)
else
    MODEL_TIMESTAMP="huggingface"
fi
# 评估开始时间戳
EVAL_TIMESTAMP=$(date +%m%d%H%M%S)
# 确定最终输出路径
if [ -n "${CUSTOM_OUTPUT_PATH}" ]; then
    OUTPUT_DIR="${CUSTOM_OUTPUT_PATH}"
    echo "📂 使用自定义输出路径: ${OUTPUT_DIR}"
else
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${MODEL_TIMESTAMP}_eval${EVAL_TIMESTAMP}"
    echo "📂 使用时间戳输出路径: ${OUTPUT_DIR}"
fi
echo "============================================================"
echo "📊 FollowIR 完整评测流程"
echo "============================================================"
echo "模型: ${MODEL_PATH}"
echo "数据集: ${TASKS[*]}"
echo "输出目录: ${OUTPUT_DIR}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"
# 创建输出目录
mkdir -p "${OUTPUT_DIR}"
# 构建任务字符串
TASKS_STR=""
for task in "${TASKS[@]}"; do
    TASKS_STR="${TASKS_STR} --tasks ${task}"
done
# 运行评估
cd /home/luwa/Documents/pylate
python -u -m eval_followir \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${CUDA_VISIBLE_DEVICES}" \
    ${TASKS_STR} \
    --note "${NOTE}"
echo ""
echo "============================================================"
echo "✅ 评测完成!"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "============================================================"
# 显示结果摘要
if [ -f "${OUTPUT_DIR}/results.json" ]; then
    echo ""
    echo "📊 结果摘要:"
    python -c "
import json
with open('${OUTPUT_DIR}/results.json') as f:
    results = json.load(f)
for task, metrics in results.items():
    print(f'  {task}:')
    for k, v in metrics.items():
        print(f'    {k}: {v}')
"
fi
