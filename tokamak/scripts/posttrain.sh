cd tokamak

EXP_ID="SafeDiffCon"
TUNING_DIR='posttrain'
GPU_ID=0
CHECKPOINT=190
DDIM_SAMPLING_STEPS=200
TRAIN_BATCH_SIZE=32
ALPHA=0.9

GUIDANCE_SCALER_VALUES=(5)
FINETUNE_STEPS_VALUES=(1)
FINETUNE_EPOCH_VALUES=(8)
FINETUNE_LR_VALUES=(7e-6)

for GUIDANCE_SCALER in "${GUIDANCE_SCALER_VALUES[@]}"; do
    for FINETUNE_EPOCH in "${FINETUNE_EPOCH_VALUES[@]}"; do
        for FINETUNE_STEPS in "${FINETUNE_STEPS_VALUES[@]}"; do
            for FINETUNE_LR in "${FINETUNE_LR_VALUES[@]}"; do
                TUNING_ID="${FINETUNE_LR}-${GUIDANCE_SCALER}-${FINETUNE_STEPS}-${FINETUNE_EPOCH}"

                    echo "Running inference with:"
                    echo "Experiment ID: $EXP_ID"
                    echo "GPU ID: $GPU_ID"
                    echo "Checkpoint: $CHECKPOINT"
                    echo "Finetune LR: $FINETUNE_LR"
                    echo "Guidance scaler: $GUIDANCE_SCALER"
                    echo "Finetune steps: $FINETUNE_STEPS"
                    echo "Finetune epoch: $FINETUNE_EPOCH"

                    python run_inference.py \
                        --gpu_id "$GPU_ID" \
                        --exp_id "$EXP_ID" \
                        --tuning_dir "$TUNING_DIR" \
                        --tuning_id "$TUNING_ID" \
                        --checkpoint "$CHECKPOINT" \
                        --train_batch_size "$TRAIN_BATCH_SIZE" \
                        --alpha "$ALPHA" \
                        --ddim_sampling_steps "$DDIM_SAMPLING_STEPS" \
                        --finetune_epoch "$FINETUNE_EPOCH" \
                        --finetune_steps "$FINETUNE_STEPS" \
                        --finetune_lr "$FINETUNE_LR" \
                        --guidance_scaler "$GUIDANCE_SCALER" 
            done
        done
    done
done