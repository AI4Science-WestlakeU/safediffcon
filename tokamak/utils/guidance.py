from typing import Union, List, Optional, Callable
import torch
import pdb
import time
from configs.inference_config import InferenceConfig
from utils.common import get_target, SCALER
from utils.metrics import calculate_safety_score

class GradientGuidance:
    """Gradient guidance calculator"""
    
    def __init__(
        self,
        target_i: List[int],
        w_obj: float = 0,
        w_safe: float = 0,
        guidance_scaler: float = 1.0,
        nt: int = 122,
        mode: str = "test",
        Q: float = 0.0,
        safety_threshold: float = 5.0,
        device: str = 'cuda',
    ):
        self.w_obj = w_obj
        self.w_safe = w_safe
        self.guidance_scaler = guidance_scaler
        self.Q = Q
        self.safety_threshold = safety_threshold
        self.state_target = get_target(target_i, device=device, split=mode)
        self.nt = nt

    def calculate_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        args: x: [B, channel, padded_time], scaled
        return: [B]
        """
        x = x * SCALER.to(x.device)
        state = x[:, :3, :self.nt]

        beta_p_final = state[:, 0, :]
        l_i_final = state[:, 2, :]

        beta_p_final_gt = self.state_target[:, 0, :]
        l_i_final_gt = self.state_target[:, 2, :]
        
        objective_beta_p = (beta_p_final - beta_p_final_gt).square().mean(-1)
        objective_l_i = (l_i_final - l_i_final_gt).square().mean(-1)
        objective = objective_beta_p + objective_l_i

        s = calculate_safety_score(state)
        safe_cost = torch.maximum(
            self.safety_threshold - s + self.Q,
            torch.zeros_like(s)
        )

        return self.w_obj * objective + self.w_safe * safe_cost
    
    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        """
        args: x: [B, channel, padded_time], scaled
        return: [B]
        """
        loss = self.calculate_loss(x)
        return torch.exp(- loss * self.guidance_scaler)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            loss = self.calculate_loss(x) * self.guidance_scaler
            # the grad_x of loss.sum() is same as the grad_x of loss
            grad_x = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

            return grad_x

def get_gradient_guidance(
    x: torch.Tensor,
    target_i: List[int],
    w_obj: float = 0,
    w_safe: float = 0,
    guidance_scaler: float = 1.0,
    device: str = 'cuda',
    nt: int = 122,
    Q: float = 0.0,
    safety_threshold: float = 5.0,
) -> Callable:
    """Build guidance gradient function"""
    return GradientGuidance(
        target_i=target_i,
        w_obj=w_obj,
        w_safe=w_safe,
        guidance_scaler=guidance_scaler,
        device=device,
        nt=nt,
        Q=Q,
        safety_threshold=safety_threshold,
    )(x)

def calculate_weight(x: torch.Tensor, state_target: torch.Tensor, 
                     nt: int, Q: float, safety_threshold: float,
                     w_obj: float, w_safe: float, guidance_scaler: float) -> torch.Tensor:
    """
    args: 
        x: [B, channel, padded_time], scaled, GPU
        state_target: [B, state_dim, time], original, GPU
    return: [B]
    """
    x = x * SCALER.to(x.device)
    state = x[:, :3, :nt]

    beta_p_final = state[:, 0, :]
    l_i_final = state[:, 2, :]

    beta_p_final_gt = state_target[:, 0, :]
    l_i_final_gt = state_target[:, 2, :]
    
    objective_beta_p = (beta_p_final - beta_p_final_gt).square().mean(-1)
    objective_l_i = (l_i_final - l_i_final_gt).square().mean(-1)
    objective = objective_beta_p + objective_l_i

    s = calculate_safety_score(state)
    safe_cost = torch.maximum(
        safety_threshold - s + Q,
        torch.zeros_like(s)
    )

    loss = w_obj * objective + w_safe * safe_cost
    weight = torch.exp(- loss * guidance_scaler)
    return weight

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

