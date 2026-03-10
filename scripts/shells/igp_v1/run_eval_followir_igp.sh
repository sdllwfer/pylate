#!/bin/bash
# ============================================================
# IGP V1 评测脚本 - 流式评估模式
# ============================================================

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 设置无缓冲输出
export PYTHONUNBUFFERED=1

# 验证环境
if [ "$CONDA_DEFAULT_ENV" != "pylate" ]; then
    echo "❌ 错误: 无法激活 pylate 环境，当前环境: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "✅ 已激活环境: $CONDA_DEFAULT_ENV"
echo "✅ Python 路径: $(which python)"

# IGP 评测脚本
# 整合了重排和评测过程，只需配置以下参数即可运行
# ========== 配置参数 ==========
# 模型路径 (IGP模型检查点目录)
MODEL_PATH="/home/luwa/Documents/pylate/output/colbert_igp_train/col_v1_max0.2_长短混合_两阶段/stage1_short_data/phase2/best_model_phase2_20260308_222534"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="1"
# 要评测的数据集 (可用: Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval)
TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
# 批处理大小 (根据GPU显存调整，默认64，可增大到128或256)
BATCH_SIZE=128
# 输出目录 (会自动创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/evaluation_data/colbert_igp"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/evaluation_data/colbert_igp/col_v1_max0.2_长短混合_只有一阶段"
# 自定义输出路径 (可选)
NOTE='端到端模型评测-v1-短长混合-bestmodel-max0.2-只有第一阶段训练'
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
echo "📊 IGP V1 流式评测流程"
echo "   模式: 每完成一个数据集立即计算指标"
echo "============================================================"
echo "模型: ${MODEL_PATH}"
echo "数据集: ${TASKS[*]}"
echo "批处理大小: ${BATCH_SIZE}"
echo "输出目录: ${OUTPUT_DIR}"
[ -n "${NOTE}" ] && echo "备注: ${NOTE}"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}"

cd /home/luwa/Documents/pylate
export PYTHONPATH="/home/luwa/Documents/pylate/scripts/evaluation:$PYTHONPATH"

# 存储所有结果
all_results=()

# 流式处理：每完成一个数据集立即计算指标
for task in "${TASKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "📚 正在处理数据集: ${task}"
    echo "============================================================"
    
    # 步骤 1: 重排生成 TREC 文件
    echo ""
    echo -e "\033[48;5;208m\033[97m============================================================\033[0m"
    echo -e "\033[48;5;208m\033[97m  步骤 1/2: 运行重排产生 TREC 结果文件                      \033[0m"
    echo -e "\033[48;5;208m\033[97m============================================================\033[0m"
    python -u -m eval_followir_igp \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "cuda:${CUDA_VISIBLE_DEVICES}" \
        --task "${task}" \
        --batch_size "${BATCH_SIZE}" \
        --note "${NOTE}"
    
    if [ $? -ne 0 ]; then
        echo "❌ 数据集 ${task} 重排失败，跳过指标计算"
        continue
    fi
    
    echo "✅ 数据集 ${task} 重排完成"
    
    # 步骤 2: 立即计算该数据集的指标
    echo ""
    echo -e "\033[42m\033[97m============================================================\033[0m"
    echo -e "\033[42m\033[97m  步骤 2/2: 计算 FollowIR 指标 p-MRR                       \033[0m"
    echo -e "\033[42m\033[97m============================================================\033[0m"
    python -u -m eval_followir_pmr \
        --run_dir "${OUTPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --note "${NOTE}" \
        --tasks "${task}"
    
    if [ $? -eq 0 ]; then
        echo "✅ 数据集 ${task} 指标计算完成"
        all_results+=("${task}")
        
        # 显示当前数据集的指标
        result_file="${OUTPUT_DIR}/results_${task}.json"
        if [ -f "${result_file}" ]; then
            echo ""
            echo "📊 ${task} 结果:"
            python -c "
import json
with open('${result_file}') as f:
    result = json.load(f)
print(f\"  p-MRR: {result.get('p-MRR', 0):.4f}\")
if 'original' in result:
    print(f\"  og nDCG@5: {result['original'].get('ndcg_at_5', 0):.4f}\")
if 'changed' in result:
    print(f\"  changed nDCG@5: {result['changed'].get('ndcg_at_5', 0):.4f}\")
"
        fi
    else
        echo "❌ 数据集 ${task} 指标计算失败"
    fi
    
    echo ""
    echo "------------------------------------------------------------"
    echo "✅ 数据集 ${task} 完整评测完成"
    echo "------------------------------------------------------------"
done

# 生成汇总结果
echo ""
echo "============================================================"
echo "📊 生成汇总结果"
echo "============================================================"

if [ ${#all_results[@]} -gt 0 ]; then
    # 构建 Python 列表字符串
    tasks_list="$(printf "'%s', " "${all_results[@]}")"
    tasks_list="[${tasks_list%, }]"
    
    python << PYEOF
import json
import os

all_results = {}
output_dir = '${OUTPUT_DIR}'
tasks = ${tasks_list}

for task in tasks:
    result_file = os.path.join(output_dir, f'results_{task}.json')
    if os.path.exists(result_file):
        with open(result_file) as f:
            all_results[task] = json.load(f)

if all_results:
    summary_path = os.path.join(output_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'💾 汇总结果已保存至: {summary_path}')
    
    print('\n📊 汇总 p-MRR:')
    for task_name, result in all_results.items():
        print(f'  {task_name}: {result.get("p-MRR", 0):.4f}')
PYEOF
fi

echo ""
echo "============================================================"
echo "✅ 完整评测流程完成!"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "============================================================"

# 显示最终结果摘要
if [ -f "${OUTPUT_DIR}/results_summary.json" ]; then
    echo ""
    echo "📊 最终结果摘要:"
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
