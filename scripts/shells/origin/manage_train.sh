#!/bin/bash
# 训练任务管理脚本
# 用于查看和管理后台训练进程
BASE_DIR="/home/luwa/Documents/pylate/output/colbert_finetune_followir"
echo "============================================================"
echo "🔧 训练任务管理工具"
echo "============================================================"
echo ""
# 选项
case "$1" in
    "list")
        echo "📋 正在运行/最近完成的训练任务:"
        echo ""
        ls -ltd ${BASE_DIR}/2* 2>/dev/null | head -10 | while read dir; do
            echo "📁 $(basename $dir)"
            if [ -f "$dir/logs/train_$(basename $dir).log" ]; then
                echo "   📝 日志: $dir/logs/train_$(basename $dir).log"
            fi
            # 查找进程
            pid_file="$dir/train.pid"
            if [ -f "$pid_file" ]; then
                pid=$(cat $pid_file)
                if ps -p $pid > /dev/null 2>&1; then
                    echo "   ✅ 运行中 (PID: $pid)"
                else
                    echo "   ⏹️ 已完成"
                fi
            fi
            echo ""
        done
        ;;
    "log")
        if [ -z "$2" ]; then
            echo "用法: $0 log <实验目录名>"
            echo "示例: $0 log 0227164946"
            exit 1
        fi
        LOG_FILE="${BASE_DIR}/${2}/logs/train_${2}.log"
        if [ -f "$LOG_FILE" ]; then
            echo "📄 日志文件: ${LOG_FILE}"
            echo ""
            tail -n 100 "${LOG_FILE}"
        else
            echo "❌ 日志文件不存在: ${LOG_FILE}"
        fi
        ;;
    "status")
        if [ -z "$2" ]; then
            echo "用法: $0 status <实验目录名>"
            echo "示例: $0 status 0227164946"
            exit 1
        fi
        EXP_DIR="${BASE_DIR}/${2}"
        if [ -d "$EXP_DIR" ]; then
            echo "📁 实验: ${2}"
            echo ""
            # 检查参数文件
            if [ -f "$EXP_DIR/training_params.txt" ]; then
                echo "📋 训练参数:"
                cat "$EXP_DIR/training_params.txt"
            fi
            # 检查损失曲线
            if [ -f "$EXP_DIR/loss_history.csv" ]; then
                echo ""
                echo "📉 损失历史:"
                cat "$EXP_DIR/loss_history.csv"
            fi
            # 检查最佳模型
            if [ -f "$EXP_DIR/best_model_info.json" ]; then
                echo ""
                echo "🏆 最佳模型信息:"
                cat "$EXP_DIR/best_model_info.json"
            fi
        else
            echo "❌ 实验目录不存在: ${EXP_DIR}"
        fi
        ;;
    "stop")
        if [ -z "$2" ]; then
            echo "用法: $0 stop <实验目录名>"
            echo "示例: $0 stop 0227164946"
            exit 1
        fi
        EXP_DIR="${BASE_DIR}/${2}"
        if [ -f "$EXP_DIR/train.pid" ]; then
            pid=$(cat "$EXP_DIR/train.pid")
            if ps -p $pid > /dev/null 2>&1; then
                kill $pid
                echo "✅ 已终止进程 PID: $pid"
            else
                echo "⚠️ 进程已不存在"
            fi
        else
            echo "❌ 未找到 PID 文件"
        fi
        ;;
    *)
        echo "用法: $0 <命令> [参数]"
        echo ""
        echo "命令:"
        echo "  list              列出所有训练任务"
        echo "  log <目录名>      查看训练日志"
        echo "  status <目录名>   查看训练状态"
        echo "  stop <目录名>     停止训练任务"
        echo ""
        echo "示例:"
        echo "  $0 list"
        echo "  $0 log 0227164946"
        echo "  $0 status 0227164946"
        echo "  $0 stop 0227164946"
        ;;
esac
