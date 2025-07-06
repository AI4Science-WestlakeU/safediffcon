import os
import torch
from typing import Optional, Callable
import matplotlib.pyplot as plt
import yaml
import pandas as pd

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
        return lambda t: t
    elif scheduler_type == 'cosine':
        return lambda t: torch.cos(t * torch.pi / 2)
    return None

def print_guidance_info(pred: torch.Tensor, Q: torch.Tensor, safe_bound: float):
    """Print guidance related information
    
    Args:
        pred: Model predictions
        Q: Quantile value
        safe_bound: Safety boundary
    """
    print(f"Q value: {Q.item():.4f}")
    print(f"Safe bound: {safe_bound:.4f}") 

def GradNorm(model, batch_metrics, device, norm: float=1, norm_type: float = 2.0,):
    r"""
    Calculate normalized gradients accumulated over all losses.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.
    """
    loss_names = ["diff_mse", "safe_mse"]
    grads = []
    norms = []
    for i, loss in enumerate(loss_names):
        loss = batch_metrics[loss].mean()
        parameters = model.parameters()
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in gradients]), norm_type)
        clip_coef = norm / (total_norm + 1e-6)
        for g in gradients:
            g.detach().mul_(clip_coef.to(g.device))
        grads.append(gradients)
        norms.append(total_norm.item())

    for param, grad_list in zip(model.parameters(), zip(*grads)):
        combined_grad = sum(grad_list) / len(grad_list)  # assume weighted average
        param.grad = combined_grad

    return norms

def log_reweights(reweights, epoch, config):
    log_dir = os.path.join(config.experiments_dir, config.exp_id, 'finetune', config.tuning_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'reweights.txt')
    with open(log_file, 'a') as f:
        f.write(f"{epoch}th epoch: {reweights.cpu().numpy()}\n")

def plot_loss_train(loss_train_history, epoch, finetune_dir):
    steps, losses = zip(*loss_train_history)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.xlabel('Finetune Step')
    plt.ylabel('Loss Train')
    plt.title(f'Loss Train vs Finetune Step (Iteration {epoch})')
    plt.grid(True)
    
    plot_path = os.path.join(finetune_dir, f'loss_train_iteration_{epoch}.png')
    plt.savefig(plot_path)
    plt.close()
    
    loss_data_path = os.path.join(finetune_dir, f'loss_train_iteration_{epoch}.yaml')
    with open(loss_data_path, 'w') as f:
        yaml.dump({
            'steps': [int(s) for s in steps],
            'losses': [float(l) for l in losses]
        }, f, default_flow_style=False)

def save_traj(traj, path, epoch):
    '''
    traj: tensor [batch, 11, 128], unscaled, GPU
    '''
    os.makedirs(path, exist_ok=True)
    u_array = traj.detach().cpu().numpy()
    reshaped_u = u_array.reshape(-1, u_array.shape[-1])
    pd.DataFrame(reshaped_u).to_csv(os.path.join(path,f'epoch_{epoch}.csv'), index=False)