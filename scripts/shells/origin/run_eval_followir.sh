#!/bin/bash
# ============================================================
# FollowIR 评测脚本
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

# FollowIR 评测脚本
# 整合了重排和评测过程，只需配置以下参数即可运行
# ========== 配置参数 ==========
# 模型路径
# MODEL_PATH="lightonai/GTE-ModernColBERT-v1"
MODEL_PATH="lightonai/ColBERT-Zero"
# MODEL_PATH="/home/luwa/Documents/pylate/output/colbert_finetune_followir/短指令检查点训FollowIR数据集/best_model"
# GPU 设备编号 (0, 1, 2, 3)
CUDA_VISIBLE_DEVICES="2"
# 要评测的数据集 (可用: Core17InstructionRetrieval Robust04InstructionRetrieval News21InstructionRetrieval)
# 设为空字符串或注释掉则评测全部三个数据集
# TASKS=("Core17InstructionRetrieval")
TASKS=("Core17InstructionRetrieval" "Robust04InstructionRetrieval" "News21InstructionRetrieval")
# 输出目录 (会自动创建时间戳子目录)
OUTPUT_BASE_DIR="/home/luwa/Documents/pylate/evaluation_data/followir"
CUSTOM_OUTPUT_PATH="/home/luwa/Documents/pylate/evaluation_data/origin_colbert/ColBERT-Zero原始评测"
# 自定义输出路径 (可选)
NOTE='ColBERT-Zero直接评测'
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

# 循环处理每个数据集
all_results=()
for task in "${TASKS[@]}"; do
    echo ""
    echo "============================================================"
    echo "� 正在处理数据集: ${task}"
    echo "============================================================"
    
    # 步骤 1: 重排生成 TREC 文件
    echo ""
    echo -e "\033[48;5;208m\033[97m============================================================\033[0m"
    echo -e "\033[48;5;208m\033[97m  步骤 1/2: 运行重排产生 TREC 结果文件                      \033[0m"
    echo -e "\033[48;5;208m\033[97m============================================================\033[0m"
    
    cd /home/luwa/Documents/pylate
    python -u scripts/evaluation/eval_followir.py \
        --model_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --device "cuda:${CUDA_VISIBLE_DEVICES}" \
        --task "${task}" \
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
    python -u scripts/evaluation/eval_followir_pmr.py \
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

echo ""
echo "============================================================"
echo "✅ 所有数据集评测完成!"
echo "📁 输出目录: ${OUTPUT_DIR}"
echo "============================================================"

# 显示汇总结果
echo ""
echo "📊 汇总结果:"
for task in "${all_results[@]}"; do
    result_file="${OUTPUT_DIR}/results_${task}.json"
    if [ -f "${result_file}" ]; then
        echo ""
        echo "  ${task}:"
        python << PYEOF
import json
with open('${result_file}') as f:
    result = json.load(f)
print(f"    p-MRR: {result.get('p-MRR', 0):.4f}")
if 'original' in result:
    print(f"    og nDCG@5: {result['original'].get('ndcg_at_5', 0):.4f}")
if 'changed' in result:
    print(f"    changed nDCG@5: {result['changed'].get('ndcg_at_5', 0):.4f}")
PYEOF
    fi
done
