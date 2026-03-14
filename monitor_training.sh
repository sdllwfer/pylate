#!/bin/bash
# 训练监控脚本

LOG_FILE="/tmp/training_monitor.log"
OUTPUT_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令"

echo "========================================" >> $LOG_FILE
echo "$(date): 检查训练状态" >> $LOG_FILE
echo "========================================" >> $LOG_FILE

# 检查进程
PROCESS_COUNT=$(ps aux | grep "3.14-col_v1_之前的长指令超过配置" | grep -v grep | wc -l)
echo "运行中的进程数: $PROCESS_COUNT" >> $LOG_FILE

# 检查GPU状态
echo "GPU状态:" >> $LOG_FILE
nvidia-smi | grep -E "(GPU|MiB|%)" | head -10 >> $LOG_FILE

# 检查训练日志
echo "" >> $LOG_FILE
echo "最近训练进度:" >> $LOG_FILE
tail -20 /tmp/two_stage_training_v2.log 2>/dev/null | grep -E "(epoch|loss|it/s|Stage|阶段|✅|❌|%)" | tail -10 >> $LOG_FILE

# 检查输出目录
echo "" >> $LOG_FILE
echo "输出目录状态:" >> $LOG_FILE
if [ -d "$OUTPUT_DIR/stage1_short_data/phase2" ]; then
    echo "Stage 1 输出存在" >> $LOG_FILE
    ls -la $OUTPUT_DIR/stage1_short_data/phase2/ | grep -E "(best_model|checkpoints)" >> $LOG_FILE
fi

if [ -d "$OUTPUT_DIR/stage2_long_data/phase2" ]; then
    echo "Stage 2 输出存在" >> $LOG_FILE
    ls -la $OUTPUT_DIR/stage2_long_data/phase2/ | grep -E "(best_model|checkpoints)" >> $LOG_FILE
fi

echo "" >> $LOG_FILE
echo "$(date): 监控完成" >> $LOG_FILE
echo "========================================" >> $LOG_FILE
echo "" >> $LOG_FILE

# 显示当前状态
cat $LOG_FILE | tail -50
