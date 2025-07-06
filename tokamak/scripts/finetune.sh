cd tokamak

EXP_ID='SafeDiffCon'
TUNING_DIR='finetune'
POST_TRAIN_ID='posttrain/7e-6-5-1-8'
GPU_ID=0
ALPHA=0.9
FINETUNE_EPOCH=5

CHECKPOINT_VALUES=(7)
DDIM_SAMPLING_STEPS_VALUES=(250)
FINETUNE_LR_VALUES=(9e-6)
GUIDANCE_SCALER_VALUES=(0.01)

for GUIDANCE_SCALER in "${GUIDANCE_SCALER_VALUES[@]}"; do
    for CHECKPOINT in "${CHECKPOINT_VALUES[@]}"; do
        for FINETUNE_LR in "${FINETUNE_LR_VALUES[@]}"; do
            for DDIM_SAMPLING_STEPS in "${DDIM_SAMPLING_STEPS_VALUES[@]}"; do
                TUNING_ID="${CHECKPOINT}-${DDIM_SAMPLING_STEPS}-${FINETUNE_LR}-${GUIDANCE_SCALER}"

                echo "Running inference with:"
                echo "Experiment ID: $EXP_ID"
                echo "GPU ID: $GPU_ID"
                echo "Checkpoint: $CHECKPOINT"
                echo "DDIM sampling steps: $DDIM_SAMPLING_STEPS"
                echo "Finetune LR: $FINETUNE_LR"
                echo "Guidance scaler: $GUIDANCE_SCALER"

                python run_inference.py \
                    --finetune_set "test" \
                    --use_guidance \
                    --backward_finetune \
                    --gpu_id "$GPU_ID" \
                    --exp_id "$EXP_ID" \
                    --tuning_dir "$TUNING_DIR" \
                    --tuning_id "$TUNING_ID" \
                    --post_train_id "$POST_TRAIN_ID" \
                    --checkpoint "$CHECKPOINT" \
                    --finetune_epoch "$FINETUNE_EPOCH" \
                    --finetune_lr "$FINETUNE_LR" \
                    --guidance_scaler "$GUIDANCE_SCALER" \
                    --ddim_sampling_steps "$DDIM_SAMPLING_STEPS" \
                    --alpha "$ALPHA"
            done
        done
    done
done