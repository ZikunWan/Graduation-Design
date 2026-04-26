#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONWARNINGS='ignore:The cuda.cudart module is deprecated and will be removed in a future release:FutureWarning'

python train.py \
    --root_dir      /data/zikun_workspace/preprocessed \
    --seed          42 \
    --client_names  BraTS Shanghai Figshare Brisc2025 \
    --global_rounds 50 \
    --eval_gap      1 \
    --local_epochs  3 \
    --local_learning_rate 1e-3 \
    --client_batch_size_map BraTS=8 Shanghai=64 Figshare=128 Brisc2025=128 \
    --val_ratio     0.1 \
    --model_name    resnet18 \
    --model_mode    baseline \
    --num_classes   5 \
    --dropout       0.0 \
    --save_gap      1 \
    --save_total_limit 2 \
    --save_dir      checkpoints/fedtgp \
    --algo          fedtgp \
    --proto_lambda  0.1 \
    --brats_ddp \
    --client_gpu_map "BraTS=3,4,5,6,7" Shanghai=0 Figshare=1 Brisc2025=2 \
    --server_learning_rate 1e-2 \
    --server_epochs 10 \
    --margin_threshold 100 \
    --server_early_stopping_patience 10 \
    --server_early_stopping_min_delta 0.0 \
    --num_workers 8 \
    "$@"
