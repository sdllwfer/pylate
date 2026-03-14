#!/bin/bash
# 持续监控脚本 - 每5分钟检查一次

LOG_FILE="/tmp/two_stage_training_v3.log"
OUTPUT_DIR="/home/luwa/Documents/pylate/output/colbert_igp_train/3.14-col_v1_之前的长指令超过配置-stage2重新训练长指令"
MONITOR_LOG="/tmp/continuous_monitor.log"

# 检查是否已经运行过Stage 2
STAGE2_STARTED=false
STAGE2_COMPLETED=false
STAGE1_COMPLETED=false

echo "$(date): 开始持续监控..." >> $MONITOR_LOG

while true; do
    echo "" >> $MONITOR_LOG
    echo "========================================" >> $MONITOR_LOG
    echo "$(date): 监控检查" >> $MONITOR_LOG
    echo "========================================" >> $MONITOR_LOG
    
    # 检查进程
    PROCESS_COUNT=$(ps aux | grep "3.14-col_v1_之前的长指令超过配置" | grep -v grep | wc -l)
    echo "运行进程数: $PROCESS_COUNT" >> $MONITOR_LOG
    
    if [ $PROCESS_COUNT -eq 0 ]; then
        echo "⚠️ 警告: 训练进程已停止！" >> $MONITOR_LOG
        # 检查是否成功完成
        if grep -q "两阶段训练全部完成" $LOG_FILE 2>/dev/null; then
            echo "✅ 训练已成功完成！" >> $MONITOR_LOG
            echo "$(date): ✅ 训练成功完成 - 可以交付" >> $MONITOR_LOG
            break
        else
            echo "❌ 训练异常终止！" >> $MONITOR_LOG
            echo "$(date): ❌ 训练异常终止" >> $MONITOR_LOG
            tail -50 $LOG_FILE | grep -iE "(error|exception|failed|失败)" >> $MONITOR_LOG
            break
        fi
    fi
    
    # 检查当前阶段
    if grep -q "阶段2: 在长数据集上微调" $LOG_FILE 2>/dev/null; then
        if [ "$STAGE2_STARTED" = false ]; then
            echo "🎉 Stage 2 已启动！" >> $MONITOR_LOG
            STAGE2_STARTED=true
        fi
        
        if grep -q "阶段2训练完成\|两阶段训练全部完成" $LOG_FILE 2>/dev/null; then
            if [ "$STAGE2_COMPLETED" = false ]; then
                echo "✅ Stage 2 已完成！" >> $MONITOR_LOG
                STAGE2_COMPLETED=true
                echo "$(date): ✅ Stage 2 完成 - 可以交付" >> $MONITOR_LOG
            fi
        else
            echo "🔄 Stage 2 运行中" >> $MONITOR_LOG
            tail -3 $LOG_FILE | grep -E "(it/s|epoch)" | tail -1 >> $MONITOR_LOG
        fi
    elif grep -q "阶段1训练完成\|阶段1已结束" $LOG_FILE 2>/dev/null; then
        if [ "$STAGE1_COMPLETED" = false ]; then
            echo "✅ Stage 1 已完成，等待 Stage 2 启动..." >> $MONITOR_LOG
            STAGE1_COMPLETED=true
        fi
    else
        echo "🔄 Stage 1 运行中" >> $MONITOR_LOG
        tail -3 $LOG_FILE | grep -E "(it/s|epoch)" | tail -1 >> $MONITOR_LOG
    fi
    
    # 检查错误
    RECENT_ERRORS=$(tail -50 $LOG_FILE | grep -iE "(error|exception|failed|失败|错误)" | wc -l)
    if [ $RECENT_ERRORS -gt 0 ]; then
        echo "⚠️ 发现错误:" >> $MONITOR_LOG
        tail -50 $LOG_FILE | grep -iE "(error|exception|failed|失败|错误)" | tail -3 >> $MONITOR_LOG
    fi
    
    echo "$(date): 检查完成" >> $MONITOR_LOG
    
    # 如果训练完成，退出循环
    if [ "$STAGE2_COMPLETED" = true ]; then
        break
    fi
    
    # 每5分钟检查一次
    sleep 300
done

echo "$(date): 监控结束" >> $MONITOR_LOG
