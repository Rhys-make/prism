#!/usr/bin/env bash
set -euo pipefail

# Replace these paths before running. This file is a command template only.
MODEL_PATH="models/llava-v1.5-7b"
DATA_PATH="data/compression/llava_v1_5_mix665k"
INIT_CHECKPOINT="outputs/sgcsr_k144_stage2_mix665k_90_5_5/best.pt"

# Keep all non-topk settings identical across the five runs.
COMMON_ARGS=(
    --model_name_or_path "$MODEL_PATH"
    --data_path "$DATA_PATH"
    --init_checkpoint_path "$INIT_CHECKPOINT"
    --local_radius 0.15
    --allow_checkpoint_config_mismatch
    --batch_size 1
    --epochs 1
    --seed 42
)

# local_topk = 8
python train_sgcsr.py \
    "${COMMON_ARGS[@]}" \
    --local_topk 8 \
    --output_dir outputs/sgcsr_local_topk_8

# local_topk = 16
python train_sgcsr.py \
    "${COMMON_ARGS[@]}" \
    --local_topk 16 \
    --output_dir outputs/sgcsr_local_topk_16

# local_topk = 32
python train_sgcsr.py \
    "${COMMON_ARGS[@]}" \
    --local_topk 32 \
    --output_dir outputs/sgcsr_local_topk_32

# local_topk = 64
python train_sgcsr.py \
    "${COMMON_ARGS[@]}" \
    --local_topk 64 \
    --output_dir outputs/sgcsr_local_topk_64

# local_topk = 0: disable top-k, while keeping local_radius=0.15.
python train_sgcsr.py \
    "${COMMON_ARGS[@]}" \
    --local_topk 0 \
    --output_dir outputs/sgcsr_local_topk_0
