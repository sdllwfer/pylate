#!/bin/bash
# 激活 conda 环境
export PATH="/home/luwa/.conda/bin:$PATH"
source /home/luwa/.conda/etc/profile.d/conda.sh
conda activate pylate
# IGP 评测脚本
# 整合了重排和评测过程，只需配置以下参数即可运行
# ========== 配置参数 ==========
# 模型路径 (IGP模型检查点目录)
MODEL_PATH="/home/luwa/Documents/pylate/output/colbert_igp_train/col_phase2_aux0_maxratio0.2/phase2/best_model_phase2_20260306_204712"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="1"
# 要评测的数据集 (可用: Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval)
TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
# 输出目录 (会自动创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/evaluation_data/colbert_igp"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/evaluation_data/colbert_igp/col_phase2_aux0_maxratio0.2"
# 自定义输出路径 (可选)
NOTE='端到端模型评测'
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
# 步骤 1: 运行重排产生 TREC 文件
echo ""
echo "============================================================"
echo "步骤 1/2: 运行重排产生 TREC 结果文件"
echo "============================================================"
cd /home/luwa/Documents/pylate
export PYTHONPATH="/home/luwa/Documents/pylate/scripts/evaluation:$PYTHONPATH"
python -u -m eval_followir_igp \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "cuda:${CUDA_VISIBLE_DEVICES}" \
    --task "${TASKS[0]}" \
    --note "${NOTE}"

# 步骤 2: 计算 FollowIR 指标
echo ""
echo "============================================================"
echo "步骤 2/2: 计算 FollowIR 指标 (p-MRR)"
echo "============================================================"

# 构建任务参数
TASKS_ARG=""
for task in "${TASKS[@]}"; do
    TASKS_ARG="${TASKS_ARG} --tasks ${task}"
done

python -u -m eval_followir_pmr \
    --run_dir "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --note "${NOTE}" \
    ${TASKS_ARG}

echo ""
echo "============================================================"
echo "✅ 完整评测流程完成!"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "============================================================"

# 显示结果摘要
if [ -f "${OUTPUT_DIR}/results_summary.json" ]; then
    echo ""
    echo "📊 结果摘要:"
    python -c "
import json
with open('${OUTPUT_DIR}/results_summary.json') as f:
    results = json.load(f)
for task, metrics in results.items():
    print(f'  {task}:')
    print(f'    p-MRR: {metrics.get(\"p-MRR\", 0):.4f}')
    if 'original' in metrics:
        print(f'    og nDCG@5: {metrics[\"original\"].get(\"ndcg_at_5\", 0):.4f}')
    if 'changed' in metrics:
        print(f'    changed nDCG@5: {metrics[\"changed\"].get(\"ndcg_at_5\", 0):.4f}')
"
fi
