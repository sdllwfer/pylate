#!/bin/bash
# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate
# IGP 评测脚本
# 整合了重排和评测过程，只需配置以下参数即可运行
# ========== 配置参数 ==========
# 模型路径 (IGP模型检查点目录)
MODEL_PATH="/home/luwa/Documents/pylate/output/colbert_igp_train/short_instruction/final_model"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="1"
# 要评测的数据集 (可用: Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval)
TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
# 输出目录 (会自动创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/evaluation_data/colbert_igp"
CUSTOM_OUTPUT_PATH=""
# 自定义输出路径 (可选)
NOTE='IGP模型评测'
# ==============================
# 从模型路径中提取时间戳
if [[ "${MODEL_PATH}" =~ [0-9]{6,} ]]; then
    MODEL_TIMESTAMP=$(echo "${MODEL_PATH}" | grep -oE '[0-9]{6,}' | head -1)
else
    MODEL_TIMESTAMP="igp_model"
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
echo "📊 IGP 完整评测流程"
echo "============================================================"
echo "模型: ${MODEL_PATH}"
echo "数据集: ${TASKS[*]}"
echo "输出目录: ${OUTPUT_DIR}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"
mkdir -p "${OUTPUT_DIR}"
# 构建任务字符串
TASKS_STR=""
for task in "${TASKS[@]}"; do
    TASKS_STR="${TASKS_STR} --tasks ${task}"
done
# 运行评估
cd /home/luwa/Documents/pylate
python -u -m eval_followir_igp \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${CUDA_VISIBLE_DEVICES}" \
    --task "${TASKS[0]}" \
    --note "${NOTE}"
echo ""
echo "============================================================"
echo "✅ 评测完成!"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "============================================================"
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
