import os
from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path

@dataclass
class EvalConfig:
    """Configuration for evaluation
    
    Args:

    """
    # Paths and IDs
    exp_id: str
    dataset: str = "tokamak"
    
    # Evaluation settings
    seed: int = 42
    device: str = "cuda"
    gpu_id: int = 1
    n_test_samples: int = 50
    batch_size: int = 50
    safety_threshold: float = 5.0
    
    # Model settings
    checkpoint: int = 10
    train_num_steps: int = 100000
    checkpoint_interval: int = 1000

    # Diffusion settings
    using_ddim: bool = True
    ddim_eta: float = 1.0
    ddim_sampling_steps: int = 200
    J_scheduler: str = None
    w_scheduler: str = None
             
    # Guidance settings
    guidance_weights: dict = None
    
    # Model conditioning
    is_condition_u0: bool = True
    is_condition_uT: bool = True
    is_condition_u0_zero_pred_noise: bool = True
    is_condition_uT_zero_pred_noise: bool = True
    
    # UNet settings
    dim: int = 64
    resnet_block_groups: int = 1
    dim_mults: List[int] = None

    @property
    def base_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def datasets_dir(self) -> str:
        return os.path.join(self.base_dir, "datasets")
    
    @property
    def experiments_dir(self) -> str:
        return os.path.join(self.base_dir, "experiments")
    
    @property
    def checkpoints_dir(self) -> str:
        return os.path.join(self.experiments_dir, "checkpoints")

    def __post_init__(self):
        """Set default values that depend on other fields"""
        if self.guidance_weights is None:
            self.guidance_weights = {
                "w_obj": 0.0,  # Force guidance
                "w_safe": 1.0,  # Safety score guidance
            }
        
        if self.dim_mults is None:
            self.dim_mults = [1, 2, 4, 8]

def get_eval_config(exp_id: Optional[str] = None, model_size: str = "base") -> EvalConfig:
    """Get evaluation configuration"""
    if model_size == "turbo":
        return EvalConfig(
            exp_id=exp_id,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            train_num_steps=200000,
            ddim_eta=1.0,
            ddim_sampling_steps=200,
        )
