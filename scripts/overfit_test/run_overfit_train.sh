#!/bin/bash
# 运行过拟合训练脚本

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate pylate

# 设置环境变量
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# 运行过拟合训练
python /home/luwa/Documents/pylate/scripts/overfit_test/overfit_train.py \
    --model_path "/home/luwa/Documents/pylate/output/colbert_igp_train/col_two_stage_short_then_long_v2/stage2_long_data/final_model_20260308_144406" \
    --data_path "/home/luwa/Documents/pylate/dataset/colbert_data/overfit_micro_batch.json" \
    --output_dir "/home/luwa/Documents/pylate/output/colbert_igp_train/overfit_test" \
    --epochs 100 \
    --batch_size 8 \
    --lr 5e-4 \
    --lr_multiplier 4.0 \
    --max_ratio 1.0 \
    --device cuda
