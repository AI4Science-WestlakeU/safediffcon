import torch
import argparse
import json
import logging
from datetime import datetime
import os

from data.burgers import BurgersDataset
from configs.posttrain_config import get_posttrain_config
from posttrain.post_train import PostTrainPipeline
from utils.common import set_seed, setup_logging, load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=2, help='GPU ID to use')
    parser.add_argument('--exp_id', type=str, default="turbo-repeat", help='Experiment ID')
    parser.add_argument('--tuning_id', type=str, default="test", help='Tuning ID')

    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='Finetune learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--cosine_epoch', type=int, default=4, help='Cosine epoch')

    parser.add_argument('--finetune_epoch', type=int, default=5, help='Finetune epoch')
    parser.add_argument('--finetune_steps', type=int, default=1000, help='Finetune steps')
    parser.add_argument('--finetune_batch_size', type=int, default=16, help='Finetune batch size')
    parser.add_argument('--finetune_subset_size', type=int, default=10000, help='Finetune subset size')

    parser.add_argument('--guidance_weights', type=str, default=None, help='JSON string of guidance weights')
    parser.add_argument('--loss_weights', type=str, default=None, help='JSON string of loss weights')
    args = parser.parse_args()
    
    # basic config
    config = get_posttrain_config(model_size='turbo', exp_id=args.exp_id, tuning_id=args.tuning_id)

    # update command line arguments
    if args.guidance_weights:
        config.guidance_weights = json.loads(args.guidance_weights)
    if args.loss_weights:
        config.loss_weights = json.loads(args.loss_weights)
    
    config.exp_id = args.exp_id
    config.tuning_id = args.tuning_id
    config.gpu_id = args.gpu_id

    config.finetune_lr = args.finetune_lr
    config.weight_decay = args.weight_decay
    config.cosine_epoch = args.cosine_epoch

    config.finetune_epoch = args.finetune_epoch
    config.finetune_steps = args.finetune_steps
    config.finetune_batch_size = args.finetune_batch_size
    config.finetune_subset_size = args.finetune_subset_size
    
    log_dir = os.path.join(
        config.experiments_dir,
        config.exp_id,
        'post_train',
        config.tuning_id
    )
    setup_logging(log_dir)

    # print replaced config
    logging.info(f"guidance_weights: {config.guidance_weights}")
    logging.info(f"loss_weights: {config.loss_weights}")
    logging.info(f"weight_decay: {config.weight_decay}")
    logging.info(f"cosine_epoch: {config.cosine_epoch}")
    logging.info(f"finetune_lr: {config.finetune_lr}")
    logging.info(f"finetune_epoch: {config.finetune_epoch}")
    logging.info(f"finetune_steps: {config.finetune_steps}")
    logging.info(f"finetune_batch_size: {config.finetune_batch_size}")
    logging.info(f"finetune_subset_size: {config.finetune_subset_size}")

    torch.cuda.set_device(config.gpu_id)
    config.device = torch.device(f"cuda:{config.gpu_id}")
    logging.info(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    
    set_seed(config.seed)
    
    test_dataset = BurgersDataset(
        split="test",
        root_path=config.datasets_dir,
        dataset=config.dataset,
        is_normalize=True,
        config=config
    )   # only used for loading trainer
    model, _ = load_model(config, test_dataset)

    # Reset AcceleratorState after loading model
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state()

    pipeline = PostTrainPipeline(
        config=config,
        model=model,
        mixed_precision_type='fp16',
        split_batches=True,
        ema_decay=0.9999,
        ema_update_every=10,
        max_grad_norm=1.0
    )

    metrics = pipeline.run()

    logging.info("Finetune Done!")
    
if __name__ == '__main__':
    main()