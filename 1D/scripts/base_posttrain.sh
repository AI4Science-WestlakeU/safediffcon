#!/bin/bash

script_path="./run_posttrain.py"
gpu_id=1
exp_id="turbo-1"
tuning_id="sweep_posttrain"
finetune_lr=0.0001
weight_decay=0.0001
cosine_epoch=4
finetune_epoch=5
finetune_steps=3200
finetune_batch_size=32
finetune_subset_size=10240
guidance_weights='{"w_score": 2500}'
loss_weights='{"loss_train": 1.0, "loss_test": 0.0}'

python "$script_path" \
    --gpu_id="$gpu_id" \
    --exp_id="$exp_id" \
    --tuning_id="$tuning_id" \
    --finetune_lr="$finetune_lr" \
    --weight_decay="$weight_decay" \
    --cosine_epoch="$cosine_epoch" \
    --finetune_epoch="$finetune_epoch" \
    --finetune_steps="$finetune_steps" \
    --finetune_batch_size="$finetune_batch_size" \
    --finetune_subset_size="$finetune_subset_size" \
    --guidance_weights="$guidance_weights" \
    --loss_weights="$loss_weights"