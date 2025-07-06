import torch
import argparse
import json
import logging
from datetime import datetime
import os

from data.burgers import BurgersDataset
from configs.inference_config import get_inference_config
from inference.inference_ft import InferenceFT
from utils.common import set_seed, setup_logging, load_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--exp_id', type=str, default="turbo-1", help='Experiment ID')
    parser.add_argument('--tuning_id', type=str, default="reproduce-ft", help='Tuning ID')
    parser.add_argument('--seed', type=int, default=5169, help='Seed')

    parser.add_argument('--finetune_lr', type=float, default=1e-5, help='Finetune learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')

    parser.add_argument('--InfFT_iters', type=int, default=6, help='Finetune steps')
    parser.add_argument('--InfFT_Q', type=float, default=None, help='Finetune Q')

    parser.add_argument('--alpha', type=float, default=0.98, help='probability of safe control')
    parser.add_argument('--guidance_weights', type=str, default=None, help='JSON string of guidance weights')
    args = parser.parse_args()
    
    # basic config
    config = get_inference_config(model_size='turbo', exp_id=args.exp_id, tuning_id=args.tuning_id)

    # update command line arguments
    if args.guidance_weights:
        config.guidance_weights = json.loads(args.guidance_weights)
    config.alpha = args.alpha
    config.seed = args.seed
    config.exp_id = args.exp_id
    config.tuning_id = args.tuning_id
    config.gpu_id = args.gpu_id

    config.finetune_lr = args.finetune_lr
    config.weight_decay = args.weight_decay

    config.InfFT_iters = args.InfFT_iters
    config.InfFT_Q = args.InfFT_Q
    
    log_dir = os.path.join(
        config.experiments_dir,
        config.exp_id,
        'test',
        'log',
        config.tuning_id
    )
    setup_logging(log_dir)

    # print replaced config
    logging.info(f"alpha: {config.alpha}")
    logging.info(f"guidance_weights: {config.guidance_weights}")
    logging.info(f"weight_decay: {config.weight_decay}")
    logging.info(f"finetune_lr: {config.finetune_lr}")
    logging.info(f"InfFT_iters: {config.InfFT_iters}")
    logging.info(f"InfFT_Q: {config.InfFT_Q}")

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

    pipeline = InferenceFT(
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
