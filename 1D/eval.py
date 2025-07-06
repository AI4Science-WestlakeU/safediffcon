import json
import os
import logging
import torch
import torch.utils.data
import argparse

from data.burgers import BurgersDataset
from model.diffusion import GaussianDiffusion
from model.unet import Unet2D
from model.trainer import Trainer
from utils.common import set_seed, get_target, setup_logging, load_model
from utils.guidance import get_gradient_guidance
from utils.metrics import (
    evaluate_samples,
    control_trajectories,
)
from configs.eval_config import EvalConfig, get_eval_config
from model.model_utils import get_scheduler

def diffuse_samples(
    model: GaussianDiffusion,
    dataset: BurgersDataset,
    dataloader: torch.utils.data.DataLoader,
    config: EvalConfig,
    device: torch.device
) -> torch.Tensor:
    """Generate samples using the model
    return: (batch, channels, padded_time, space), scaled, GPU
    """
    samples = []

    model.eval()
    
    for i, batch in enumerate(dataloader):
        if i * config.batch_size >= config.n_test_samples:
            break
            
        batch = batch.to(device)
        sample = model.sample(
            batch_size=batch.shape[0],
            clip_denoised=True,
            device=device,
            u_init=batch[:,0,0,:],
            u_final=batch[:,0,dataset.nt_total-1,:],
            guidance_u0=True,
            nablaJ=None,
            J_scheduler=None,
            w_scheduler=None,
            ddim_sampling_eta=config.ddim_eta if config.using_ddim else None,
            timesteps=config.ddim_sampling_steps if config.using_ddim else None
        )
        samples.append(sample)
        logging.info(f'Generated batch {i+1}/{len(dataloader)}')
    
    samples = torch.cat(samples[:config.n_test_samples//config.batch_size + 1])
    samples = samples[:config.n_test_samples]
    
    return samples

def save_results(metrics: dict, config: EvalConfig, exp_dir: str):
    """Save evaluation results"""
    results_file = os.path.join(exp_dir, 'eval_results.json')
    
    existing_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
            
    existing_results[f'checkpoint_{config.checkpoint}'] = metrics
    
    sorted_results = dict(sorted(existing_results.items(), key=lambda item: int(item[0].split('_')[1])))
    
    with open(results_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

def evaluate(config: EvalConfig):
    """Main evaluation function"""
    # Setup
    exp_dir = os.path.join(config.experiments_dir, config.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    setup_logging(exp_dir)
    set_seed(config.seed)

    torch.cuda.set_device(config.gpu_id)
    config.device = torch.device(f"cuda:{config.gpu_id}")
    logging.info(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    
    # Load dataset and model
    dataset = BurgersDataset(
        split="test",
        root_path=config.datasets_dir,
        dataset=config.dataset,
        config=config
    )
    logging.info(f'Test dataset loaded: {len(dataset)} samples')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,  
        num_workers=4,
        pin_memory=True  
    )
    
    model, model_path = load_model(config, dataset)
    logging.info(f'Model loaded from checkpoint {model_path}')
    
    # Generate samples and unnormalize
    diffused = diffuse_samples(model, dataset, dataloader, config, config.device)
    diffused = diffused * dataset.scaler
    logging.info(f'Generated {len(diffused)} samples')
    
    # Get controlled trajectories using solver
    u_controlled = control_trajectories(diffused, dataset.nt_total)
    u_target = get_target(list(range(config.n_test_samples)), 
                          dataset=config.dataset,
                          is_normalize=False).to(config.device)
    
    metrics = evaluate_samples(
        diffused, 
        u_controlled, 
        u_target, 
        dataset.nt_total,
        config.u_bound,
        use_max_safety=config.use_max_safety
    )
    save_results(metrics, config, exp_dir)

def main():
    """Main entry point"""
    config = get_eval_config(model_size="turbo")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--exp_id", type=str, required=True, help="Experiment ID")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use")
    parser.add_argument("--use_max_safety", type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    # pdb.set_trace()
    if args.checkpoint:
        config.checkpoint = args.checkpoint
    if args.exp_id:
        config.exp_id = args.exp_id
    if args.gpu_id:
        config.gpu_id = args.gpu_id
    if args.use_max_safety:
        config.use_max_safety = args.use_max_safety
    
    print(f"Whether use the max score: {config.use_max_safety}")
    evaluate(config)

if __name__ == "__main__":
    main()
