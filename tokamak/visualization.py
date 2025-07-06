import json
import os
import logging
import torch
import torch.utils.data
from data.tokamak_dataset import TokamakDataset
from model.trainer import Trainer
from model.diffusion import GaussianDiffusion
from utils.common import set_seed, get_target, build_model, load_model, SCALER
from utils.guidance import get_gradient_guidance
from utils.metrics import (
    control_trajectories,
)
from configs.inference_config import InferenceConfig
# from model.model_utils import get_scheduler

import torch
from typing import Optional, Callable

def get_scheduler(scheduler_type: str) -> Optional[Callable]:
    """Get scheduler function
    
    Args:
        scheduler_type: Type of scheduler
        
    Returns:
        Scheduler function or None
    """
    if scheduler_type == 'constant':
        return lambda t: 1.0
    elif scheduler_type == 'linear':
        print('Linear scheduler is not recommended.')
        return lambda t: t
    elif scheduler_type == 'cosine':
        print('Cosine scheduler is not recommended.')
        return lambda t: torch.cos(t * torch.pi / 2)
    return None


from kstar_solver_vis import KSTARSolver
from multiprocessing import Pool


def process_single(args):
    idx, action = args
    solver = KSTARSolver(sample_id=idx)
    solver.simulate(action)

def control_trajectories(diffused: torch.Tensor, nt_total: int) -> torch.Tensor:
    """Generate controlled trajectories using solver
    
    Args:
        diffused: Diffusion model output, shape (batch, channels, padded_time), original scale
            0: beta_p, 1: q_95, 2: l_i
            3~11: actions
    Returns:
        state_controlled: Controlled state trajectories from solver, shape (batch, state_dim, time)
    """
    diffused_action = diffused[:, 3:, :nt_total-1].permute(0, 2, 1).cpu().numpy()

    with Pool() as pool:
        args = [(idx, diffused_action[idx]) for idx in range(diffused_action.shape[0])]
        pool.map(process_single, args)
        

def load_model(config,
               model_path,
               dataset: TokamakDataset,
               ) -> GaussianDiffusion:
    model = build_model(config, dataset)
    cp = torch.load(model_path, map_location=config.device)
    model.load_state_dict(cp['model'])
    model.to(config.device)
    config.finetune_quantile = cp['quantile']
    config.finetune_alpha = cp['config'].alpha
    config.finetune_guidance_scaler = cp['config'].guidance_scaler if not cp['config'].use_guidance \
                                    else 2 * cp['config'].guidance_scaler
    config.finetune_guidance_weights = cp['config'].guidance_weights
    return model, model_path



def diffuse_samples(
    model: GaussianDiffusion,
    dataset: TokamakDataset,
    dataloader: torch.utils.data.DataLoader,
    config,
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

            nablaJ=lambda x: get_gradient_guidance(x, 
                target_i=list(range(config.n_test_samples)),
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

def save_results(metrics: dict, config, exp_dir: str):
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

def evaluate(config, model_path: str):
    """Main evaluation function"""
    # Setup
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
    
    model, model_path = load_model(config, model_path, dataset)
    logging.info(f'Model loaded from checkpoint {model_path}')
    
    # Generate samples and unnormalize
    with torch.no_grad():
        diffused = diffuse_samples(model, dataset, dataloader, config, config.device)
        diffused = diffused * SCALER.to(config.device)
        logging.info(f'Generated {len(diffused)} samples')
        
        # Get controlled trajectories using solver
        control_trajectories(diffused, dataset.nt_total)
        state_target = get_target(list(range(config.n_test_samples)), 
                            device=config.device,
                            is_normalize=False)
    

def main():
    import json
    
    """Main entry point"""
    # config = get_eval_config(model_size="turbo")
    model_path = '/home/conformal_diffcon/tokamak_new/experiments/best_final_checkpoints/7-{"w_obj": 0.0, "w_safe": 1.0}-9e-6-120@200-0.9-{"w_obj": 0.0, "w_safe": 1.0}-8e-6-10-1-10/model-3.pth'
    gpu_id = 0
    
    config_path = '/home/conformal_diffcon/tokamak_new/experiments/best_final_checkpoints/7-{"w_obj": 0.0, "w_safe": 1.0}-9e-6-120@200-0.9-{"w_obj": 0.0, "w_safe": 1.0}-8e-6-10-1-10/config.json'
    config_dict = json.load(open(config_path, 'r'))
    config = InferenceConfig(
        tuning_dir=config_dict['tuning_dir'],
        tuning_id=config_dict['tuning_id'],
        exp_id=config_dict['exp_id'],
        seed=config_dict['seed'],
        gpu_id=config_dict['gpu_id'],
        device=config_dict['device'],
        dataset=config_dict['dataset'],
        nt_total=config_dict['nt_total'],
        pad_size=config_dict['pad_size'],
        safety_threshold=config_dict['safety_threshold'],
        n_test_samples=config_dict['n_test_samples'],
        test_batch_size=config_dict['test_batch_size'],
        n_cal_samples=config_dict['n_cal_samples'],
        cal_batch_size=config_dict['cal_batch_size'],
        num_cal_batch=config_dict['num_cal_batch'],
        train_batch_size=config_dict['train_batch_size'],
        finetune_set=config_dict['finetune_set'],
        use_guidance=config_dict['use_guidance'],
        backward_finetune=config_dict['backward_finetune'],
        optimizer=config_dict['optimizer'],
        finetune_lr=config_dict['finetune_lr'],
        finetune_epoch=config_dict['finetune_epoch'],
        finetune_steps=config_dict['finetune_steps'],
        loss_weights=config_dict['loss_weights'],
        use_grad_norm=config_dict['use_grad_norm'],
        alpha=config_dict['alpha'],
        checkpoint_dir=config_dict['checkpoint_dir'],
        wo_post_train=config_dict['wo_post_train'],
        post_train_id=config_dict['post_train_id'],
        checkpoint=config_dict['checkpoint'],
        train_num_steps=config_dict['train_num_steps'],
        checkpoint_interval=config_dict['checkpoint_interval'],
        using_ddim=config_dict['using_ddim'],
        ddim_eta=config_dict['ddim_eta'],
        ddim_sampling_steps=config_dict['ddim_sampling_steps'],
        J_scheduler=config_dict['J_scheduler'],
        w_scheduler=config_dict['w_scheduler'],
        guidance_weights=config_dict['guidance_weights'],
        guidance_scaler=config_dict['guidance_scaler'],
        is_condition_u0=config_dict['is_condition_u0'],
        is_condition_uT=config_dict['is_condition_uT'],
        is_condition_u0_zero_pred_noise=config_dict['is_condition_u0_zero_pred_noise'],
        is_condition_uT_zero_pred_noise=config_dict['is_condition_uT_zero_pred_noise'],
        dim=config_dict['dim'],
        resnet_block_groups=config_dict['resnet_block_groups'],
        dim_mults=config_dict['dim_mults']
    )
    
    config.gpu_id = gpu_id
    config.batch_size = config.test_batch_size
    print(f"\nStarting evaluation for checkpoint {config.checkpoint} in experiment {config.exp_id} on GPU {config.gpu_id}")
    evaluate(config, model_path)

if __name__ == "__main__":
    main()
