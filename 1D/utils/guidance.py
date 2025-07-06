from typing import Union, List, Optional, Callable
import torch

from utils.common import get_target, SCALER
from utils.metrics import calculate_safety_score

class GradientGuidance:
    """Gradient guidance calculator"""
    
    def __init__(
        self,
        target_i: List[int],
        wu: float = 0,
        wf: float = 0,
        dataset: str = 'free_u_f_1e5',
        mode: str = "test",
    ):
        self.wu = wu
        self.wf = wf
        self.u_target = get_target(target_i, dataset=dataset, split=mode)
        
    def calculate_loss(self, x: torch.Tensor) -> torch.Tensor:
        u = x[:, 0, :11, :]
        f = x[:, 1, :10, :]
        
        u0 = u[:, 0, :]
        uf = u[:, -1, :]
        u0_gt = self.u_target[:, 0, :]
        uf_gt = self.u_target[:, -1, :]
        
        loss_u = (u0 - u0_gt).square() + (uf - uf_gt).square()
        loss_u = loss_u.mean()
        
        loss_f = f.square().sum((-1, -2)).mean()
        
        return self.wu * loss_u + self.wf * loss_f
        
    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            loss = self.calculate_loss(x)
            return torch.autograd.grad(loss, x)[0]

def get_gradient_guidance(
    target_i: List[int],
    wu: float = 0,
    wf: float = 0,
    dataset: str = 'free_u_f_1e5',
) -> Callable:
    """Build guidance gradient function"""
    return GradientGuidance(
        target_i=target_i,
        wu=wu,
        wf=wf,
        dataset=dataset,
    ) 

def calculate_guidance(state: torch.Tensor, Q: float, config) -> torch.Tensor:
    """Calculate guidance value
    
    Args:
        state: Input tensor
        
    Returns:
        guidance: Guidance value
    """
    state = state * SCALER
    if config.use_max_safety:
        s = state[:,2,:11,:].mean(dim=(-1,-2)) # if use_max_safety = False, use amax instead of mean
    else:
        s = state[:,2,:11,:].amax(dim=(-1,-2))
    guidance_safe = torch.maximum(
        s + Q - config.u_bound**2,
        torch.zeros_like(s)
    )
    
    return guidance_safe * config.guidance_weights["w_score"]

def get_finetune_guidance(config, x: torch.Tensor, Q: float) -> torch.Tensor:
    guidance = calculate_guidance(x, 
                                  Q,
                                  config).sum()
    grad_x = torch.autograd.grad(guidance, x, grad_outputs=torch.ones_like(guidance), retain_graph=True)[0]
    # print("==guidance==: ", guidance.norm().item(), guidance.max().item(), guidance.min().item())
    # print("==grad_x==: ", grad_x.norm().item(), grad_x.max().item(), grad_x.min().item())
    return grad_x