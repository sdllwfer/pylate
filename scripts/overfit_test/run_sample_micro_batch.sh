#!/bin/bash
# 运行过拟合测试数据采样脚本

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 运行数据采样
python /home/luwa/Documents/pylate/scripts/overfit_test/sample_micro_batch.py
