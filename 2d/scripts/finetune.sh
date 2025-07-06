cd 2d

cp_id_ls=(5)
steps=(100)
alphas=(0.01)
standard_fixed_ratio_list=(495)
test_backward_batch_size_list=(13)

for k in "${!steps[@]}"; do
    for j in "${!cp_id_ls[@]}"; do
        for m in "${!alphas[@]}"; do
            for n in "${!standard_fixed_ratio_list[@]}"; do
                for l in "${!test_backward_batch_size_list[@]}"; do
                    echo ${cp_id_ls[j]} ${steps[k]}
                    step=${steps[k]}
                    cp_id=${cp_id_ls[j]}
                    alpha=${alphas[m]}
                    test_backward_batch_size=${test_backward_batch_size_list[l]}
                    python inference_2d.py \
                    --seed 1 \
                    --id 'finetune' \
                    --finetune_set 'test' \
                    --backward_finetune \
                    --cal_batch_size 40 \
                    --N_cal_batch 1 \
                    --test_batch_size 50 \
                    --N_test_batch 1 \
                    --test_backward_batch_size $test_backward_batch_size \
                    --diffusion_model_path "./results/test/posttrain" \
                    --inference_result_path "./results/test/" \
                    --ddim_sampling_steps $step \
                    --diffusion_checkpoint $cp_id \
                    --standard_fixed_ratio_list ${standard_fixed_ratio_list[n]}\
                    --w_safe_list 1 \
                    --alpha $alpha \
                    --finetune_steps 1 \
                    --epochs 4 
                done
            done
        done
    done
done
