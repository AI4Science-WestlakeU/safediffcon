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