import torch
from torch.autograd import grad
import os
from datetime import datetime

from configs.posttrain_config import PostTrainConfig
from utils.common import SCALER

def calculate_guidance(state: torch.Tensor, Q: float, config: PostTrainConfig) -> torch.Tensor:
    """Calculate guidance value
    
    Args:
        state: Input tensor from training set or calibration set
        Q: Q value for guidance
        u_bound: Upper bound for guidance
        guidance_weights: Dictionary of guidance weights
        
    Returns:
        guidance: Guidance value
    """
    state = state * SCALER
    if config.use_max_safety:
        s = state[:,2,:11,:].mean(dim=(-1,-2))
    else:
        s = state[:,2,:11,:].amax(dim=(-1,-2))
    guidance_safe = torch.maximum(
        s + Q - config.u_bound**2,
        torch.zeros_like(s)
    )
    
    # try:
    #     if state.shape[0] in [250]:
    #         log_guidance_safe(guidance_safe, state, config)
    # except Exception as e:
    #     print(f"Error logging guidance_safe: {e}")

    return guidance_safe * config.guidance_weights["w_score"]

def get_weight(state: torch.Tensor, Q: float, config: PostTrainConfig) -> torch.Tensor:
    """Calculate weight for a state
        
    Returns:
        weight: Weight for the state [B]
    """
    guidance = calculate_guidance(state, Q, config)
    return torch.exp(-guidance)

def normalize_weights(weights):
    '''
    Args:
        weights: torch.Tensor, [B]
    Returns:
        normalized_weights: torch.Tensor, [B]
    '''
    # if inf, replace with max
    if torch.isinf(weights).any():
        non_inf_mask = ~torch.isinf(weights)
        max_non_inf = weights[non_inf_mask].max()
        weights[torch.isinf(weights)] = max_non_inf

    if weights.sum() == 0:
        normalized_weights = torch.ones_like(weights)
    else:
        normalized_weights = weights.shape[0] * weights / weights.sum()

    return normalized_weights

def log_guidance_safe(guidance_safe, state, config):
    """Log guidance_safe values to a file."""
    log_dir = os.path.join(config.experiments_dir, config.exp_id, 'finetune', config.tuning_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'guidance_safe.txt')
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        guidance_safe_str = ' '.join([f"{x:.6f}" for x in guidance_safe.cpu().numpy()])
        f.write(f"{timestamp} Batch size {state.shape[0]}: {guidance_safe_str}\n")