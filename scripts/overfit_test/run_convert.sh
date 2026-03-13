#!/bin/bash
# 转换测试集为训练格式

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 运行转换脚本
python /home/luwa/Documents/pylate/scripts/overfit_test/convert_test_to_train.py
