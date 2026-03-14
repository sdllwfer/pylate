#!/bin/bash
# 实时监控训练脚本 - 特别关注 Stage 1 到 Stage 2 的过渡

LOG_FILE="/tmp/two_stage_training_v3.log"
OUTPUT_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令"

echo "========================================"
echo "$(date): 实时监控报告"
echo "========================================"

# 检查进程
PROCESS_COUNT=$(ps aux | grep "3.14-col_v1_之前的长指令超过配置" | grep -v grep | wc -l)
echo "运行中的进程数: $PROCESS_COUNT"

if [ $PROCESS_COUNT -eq 0 ]; then
    echo "⚠️ 警告: 没有训练进程在运行！"
fi

# 检查GPU状态
echo ""
echo "GPU状态:"
nvidia-smi | grep -E "(GPU|MiB|%)" | head -8

# 检查训练进度
echo ""
echo "训练进度:"
tail -5 $LOG_FILE | grep -E "(it/s|epoch|loss)" | tail -3

# 检查当前阶段
echo ""
echo "当前阶段:"
if grep -q "阶段2: 在长数据集上微调" $LOG_FILE 2>/dev/null; then
    if grep -q "阶段2已结束\|阶段2训练完成" $LOG_FILE 2>/dev/null; then
        echo "✅ Stage 2 已完成"
    else
        echo "🔄 正在运行 Stage 2"
        # 检查 Stage 2 进度
        tail -50 $LOG_FILE | grep -E "it/s" | tail -1
    fi
elif grep -q "阶段1已结束\|阶段1训练完成" $LOG_FILE 2>/dev/null; then
    echo "⏳ Stage 1 已完成，正在启动 Stage 2..."
else
    echo "🔄 正在运行 Stage 1"
fi

# 检查输出目录
echo ""
echo "模型保存状态:"
if [ -d "$OUTPUT_DIR/stage1_short_data/phase2" ]; then
    STAGE1_MODELS=$(find $OUTPUT_DIR/stage1_short_data/phase2 -name "best_model_*" -type d | wc -l)
    echo "  Stage 1: 已保存 $STAGE1_MODELS 个最佳模型"
fi

if [ -d "$OUTPUT_DIR/stage2_long_data/phase2" ]; then
    STAGE2_MODELS=$(find $OUTPUT_DIR/stage2_long_data/phase2 -name "best_model_*" -type d | wc -l)
    echo "  Stage 2: 已保存 $STAGE2_MODELS 个最佳模型"
fi

# 检查关键错误
echo ""
RECENT_ERRORS=$(tail -100 $LOG_FILE | grep -iE "(error|exception|failed|失败|错误)" | wc -l)
if [ $RECENT_ERRORS -gt 0 ]; then
    echo "⚠️ 发现 $RECENT_ERRORS 个错误/异常:"
    tail -100 $LOG_FILE | grep -iE "(error|exception|failed|失败|错误)" | tail -3
else
    echo "✅ 无错误/异常"
fi

echo ""
echo "$(date): 监控完成"
echo "========================================"
