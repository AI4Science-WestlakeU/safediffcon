cd 2d

accelerate launch --config_file default_config.yaml \
--main_process_port 29500 \
--gpu_ids 0,1 \
train_2d.py 