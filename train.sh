#!/bin/bash
set -euo pipefail

nproc_per_node=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=$((20000 + RANDOM % 20000))
export MASTER_ADDR=127.0.0.1

RDZV_ID="sft_${USER}_$RANDOM"

echo "[train_sft] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} RDZV_ID=${RDZV_ID}"


torchrun \
  --nproc_per_node "${nproc_per_node}" \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id "${RDZV_ID}" \
  train.py \
    --model 'Qwen/Qwen2.5-VL-7B-Instruct' \
    --train_type lora \
    --dataset my_training_data \
    --new_special_tokens 'special_tokens/action_and_var_tokens.txt' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --max_pixels 1166400 \
    --split_dataset_ratio 0.0 \
    --freeze_vit true \
    --freeze_aligner true \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_steps 100 \
    --logging_steps 5 \
    --lr_scheduler_type cosine \
    --output_dir './outputs/train' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 7 \
    --deepspeed zero2 \
    --add_version \
    --modules_to_save embed_tokens lm_head \
    --template my_qwen2_5_vl \
