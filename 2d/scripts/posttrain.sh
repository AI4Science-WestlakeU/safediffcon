cd 2d

cp_id_ls=(20)
steps=(100)

for k in "${!steps[@]}"; do
    for j in "${!cp_id_ls[@]}"; do
        echo ${cp_id_ls[j]} ${steps[k]} 
        step=${steps[k]}
        cp_id=${cp_id_ls[j]}
        python inference_2d.py \
        --id 'posttrain' \
        --test_batch_size 50 \
        --cal_batch_size 50 \
        --N_cal_batch 4 \
        --diffusion_model_path "./results/train/" \
        --inference_result_path "./results/test/" \
        --ddim_sampling_steps $step \
        --diffusion_checkpoint $cp_id \
        --standard_fixed_ratio_list 100 \
        --w_safe_list 0.9 \
        --alpha 0.04 \
        --finetune_steps 4000 \
        --finetune_batch_size 14 \
        --finetune_lr 1e-4 \
        --epochs 8 
    done
done
