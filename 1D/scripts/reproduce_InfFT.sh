#!/bin/bash

script_path="./run_inference_ft.py"
gpu_id=1
exp_id="turbo-1"
tuning_id="reproduce-ft"
alpha=0.98
finetune_lr=1e-5
weight_decay=1e-4
InfFT_iters=3
guidance_weights='{"w_score": 500}'

python "$script_path" \
    --gpu_id="$gpu_id" \
    --exp_id="$exp_id" \
    --tuning_id="$tuning_id" \
    --alpha="$alpha" \
    --finetune_lr="$finetune_lr" \
    --weight_decay="$weight_decay" \
    --InfFT_iters="$InfFT_iters" \
    --guidance_weights="$guidance_weights"