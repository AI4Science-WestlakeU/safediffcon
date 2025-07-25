import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
import pdb
from multiprocessing import Pool

from data.tokamak_dataset import TokamakDataset
from kstar_solver import KSTARSolver

def evaluate_samples(
    diffused: torch.Tensor,
    state_controlled: torch.Tensor,
    state_target: torch.Tensor,
    safety_threshold: float,
    dataset: TokamakDataset,
) -> Dict[str, float]:
    """Evaluate generated samples against ground truth
    
    Args:
        diffused: Samples generated by diffusion model, shape (batch, channels, padded_time), original scale
        state_controlled: State trajectories from solver, shape (batch, state_dim, time), original scale
        state_target: Ground truth trajectories, shape (batch, state_dim, time), original scale
        safety_threshold: Safety bound for state values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    metrics = {}

    # 1. Calculate MSE between diffusion output and controlled trajectory
    diffusion_mse = (state_controlled - diffused[:, :3, :dataset.nt_total]).square().mean((-1,-2))
    metrics['diffusion_mse_mean'] = diffusion_mse.mean().item()
    metrics['diffusion_mse_std'] = diffusion_mse.std().item()

    # 2. Calculate MSE between controlled trajectory and ground truth
    beta_p_mse = (state_target[:, 0, :dataset.nt_total] - state_controlled[:, 0, :dataset.nt_total]).square().mean((-1))
    metrics['beta_p_mse_mean'] = beta_p_mse.mean().item()
    metrics['beta_p_mse_std'] = beta_p_mse.std().item()
    
    l_i_mse = (state_target[:, 2, :dataset.nt_total] - state_controlled[:, 2, :dataset.nt_total]).square().mean((-1))
    metrics['l_i_mse_mean'] = l_i_mse.mean().item()
    metrics['l_i_mse_std'] = l_i_mse.std().item()

    metrics['obj_mse_mean'] = metrics['beta_p_mse_mean'] + metrics['l_i_mse_mean']
    metrics['obj_mse_std'] = (beta_p_mse + l_i_mse).std().item()

    # 3. Calculate safety metrics (using controlled trajectory)
    safety_metrics = calculate_safety_metrics(state_controlled[:, 1, :], safety_threshold, diffused[:, 1, :dataset.nt_total])
    metrics.update(safety_metrics)
    
    return metrics

def process_single(args):
    idx, action, seed = args
    solver = KSTARSolver(random_seed=seed)
    outputs = solver.simulate(action)
    return idx, outputs

def control_trajectories(diffused: torch.Tensor, nt_total: int, seed: int) -> torch.Tensor:
    """Generate controlled trajectories using solver
    
    Args:
        diffused: Diffusion model output, shape (batch, channels, padded_time), original scale
            0: beta_p, 1: q_95, 2: l_i
            3~11: actions
    Returns:
        state_controlled: Controlled state trajectories from solver, shape (batch, state_dim, time)
    """
    diffused_state = diffused[:, :3, :nt_total]
    diffused_action = diffused[:, 3:, :nt_total-1].permute(0, 2, 1).cpu().numpy()

    state_controlled = torch.zeros_like(diffused_state)
    # with Pool() as pool:
    #     args = [(idx, diffused_action[idx], seed) for idx in range(diffused_action.shape[0])]
    #     results = pool.map(process_single, args)
        
    results = []
    for idx in range(diffused_action.shape[0]):
        results.append(process_single((idx, diffused_action[idx], seed)))

    for idx, outputs in results:
        state_controlled[idx, :, :] = torch.tensor(outputs[:, [1, 4, 6]], device=diffused_state.device).permute(1, 0)

    return state_controlled

def calculate_safety_metrics(controlled_q95: torch.Tensor, threshold: float, diffused_s: torch.Tensor) -> Dict[str, float]:
    """Calculate safety metrics
    
    Args:
        controlled_q95: Controlled q95, shape (batch, time)
        threshold: Safety threshold
        diffused_s: Safety score, shape (batch, time)
        
    Returns:
        Dictionary containing safety metrics
    """
    metrics = {}
    
    # 1. Calculate ratio of data below threshold
    below_mask = (controlled_q95 < threshold)
    
    # Statistics by time
    time_below = (below_mask).float().mean()
    metrics['time_below_ratio'] = time_below.item()
    
    # Statistics by sample
    sample_below = (below_mask.any(dim=(-1))).float().mean()
    metrics['sample_below_ratio'] = sample_below.item()
    
    # 2. Calculate safety score
    safety_score = controlled_q95.amin(dim=-1)
    metrics['safety_score_mean'] = safety_score.mean().item()
    metrics['safety_score_std'] = safety_score.std().item()
    
    # 3. Calculate diffused safety score mse
    diffused_score_mse = (diffused_s.amin(dim=-1) - safety_score).square().mean()
    metrics['diffused_score_mse'] = diffused_score_mse.item()

    # 4. Calculate reported safety metric
    reported_safe_metric = calculate_reported_safe_metric(controlled_q95, threshold)
    metrics['reported_safe_metric'] = reported_safe_metric
    
    return metrics

def calculate_reported_safe_metric(controlled_q95, threshold):
    """
    Compute safe metric reported during evaluation.
    Args:
        controlled_q95: Controlled q95, shape (batch, time)
        threshold: Safety threshold
    Returns:
        metric: scalar
    """
    scores = torch.min(controlled_q95, dim=1)[0]
    normalized_scores = threshold / scores
    mask_safe = (scores >= threshold).float()
    mask_unsafe = (scores < threshold).float()
    inside_mean = (normalized_scores * mask_safe).sum() / mask_safe.sum().clamp(min=1)
    outside_mean = (normalized_scores * mask_unsafe).sum() / mask_unsafe.sum().clamp(min=1)
    metric = inside_mean + outside_mean
    return metric.item()

def calculate_safety_score(x: torch.Tensor) -> torch.Tensor:
    """Calculate safety score
    Returns:
        safety_score: Safety score, shape (batch,)
    """
    q_95 = x[:, 1, :]
    safety_score = q_95.amin(dim=-1)
    return safety_score

