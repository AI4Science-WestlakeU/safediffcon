import json
import os
import logging
import torch
import torch.utils.data
import argparse
from data.tokamak_dataset import TokamakDataset
from model.diffusion import GaussianDiffusion
from utils.common import set_seed, get_target, setup_logging, load_model, SCALER
from utils.guidance import get_gradient_guidance
from utils.metrics import (
    evaluate_samples,
    control_trajectories,
)
from configs.eval_config import EvalConfig, get_eval_config
from model.model_utils import get_scheduler

def diffuse_samples(
    model: GaussianDiffusion,
    dataset: TokamakDataset,
    dataloader: torch.utils.data.DataLoader,
    config: EvalConfig,
    device: torch.device
) -> torch.Tensor:
    """Generate samples using the model"""
    samples = []

    model.eval()
    
    for i, batch in enumerate(dataloader):
        if i * config.batch_size >= config.n_test_samples:
            break
            
        batch = batch.to(device)
        sample = model.sample(
            batch_size=batch.shape[0],
            clip_denoised=True,
            guidance_u0=True,
            device=device,
            u_init=batch[:,:3,0],
            u_final=batch[:,[0,2],:dataset.nt_total],
            nablaJ=get_gradient_guidance(
                target_i=list(range(i * config.batch_size, 
                                  min((i + 1) * config.batch_size, config.n_test_samples))),
                w_obj=config.guidance_weights["w_obj"],
                w_safe=config.guidance_weights["w_safe"],
                nt=dataset.nt_total,
                safety_threshold=config.safety_threshold,
                device=device,
            ) if any(config.guidance_weights.values()) else None,
            J_scheduler=get_scheduler(config.J_scheduler),
            w_scheduler=get_scheduler(config.w_scheduler),
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
    dataset = TokamakDataset(split='test')
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
    diffused = diffused * SCALER.to(config.device)
    logging.info(f'Generated {len(diffused)} samples')
    
    # Get controlled trajectories using solver
    state_controlled = control_trajectories(diffused, dataset.nt_total)
    state_target = get_target(list(range(config.n_test_samples)), 
                          device=config.device,
                          is_normalize=False)
    
    metrics = evaluate_samples(
        diffused, 
        state_controlled, 
        state_target, 
        config.safety_threshold,
        dataset
    )
    save_results(metrics, config, exp_dir)

def main():
    """Main entry point"""
    config = get_eval_config(model_size="turbo")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=int, help="Checkpoint to evaluate")
    parser.add_argument("--exp_id", type=str, help="Experiment ID")
    parser.add_argument("--gpu_id", type=int, help="GPU ID to use")
    args = parser.parse_args()
    if args.checkpoint:
        config.checkpoint = args.checkpoint
    if args.exp_id:
        config.exp_id = args.exp_id
    if args.gpu_id:
        config.gpu_id = args.gpu_id
    print(f"\nStarting evaluation for checkpoint {config.checkpoint} in experiment {config.exp_id} on GPU {config.gpu_id}")
    evaluate(config)

if __name__ == "__main__":
    main()
