import os
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class TrainConfig:
    """Training configuration for diffusion models."""
    
    # Basic settings
    exp_id: str                           # Experiment folder id
    seed: int = 42                        # Random seed for reproducibility
    date_time: str = datetime.today().strftime('%Y-%m-%d')  # Date for experiment folder
    device: str = "cuda"                  # Device to use
    gpu_id: int = 6                      # GPU ID to use

    # Training settings
    train_num_steps: int = 100000         # Total number of training steps
    checkpoint_interval: int = 1000      # Save checkpoint every N steps
    lr: float = 1e-5                      # Learning rate for training
    
    # Dataset settings
    dataset: str = "free_u_f_1e5"        # Dataset name for training
    use_max_safety: bool = True          # Whether to use maximum safety score for the entire sample
    
    # UNet hyperparameters
    dim: int = 64                         # Base dimension for UNet features
    resnet_block_groups: int = 1          # Number of groups in GroupNorm
    dim_mults: List[int] = (1, 2, 4, 8)   # Channel multipliers for each UNet level
    
    # Conditioning settings
    train_on_padded_locations: bool = False # Whether to train on padded locations
    is_condition_u0: bool = True         # Whether to learn p(u_[1, T] | u0)
    is_condition_uT: bool = True         # Whether to learn p(u_[0, T-1] | uT)
    is_condition_u0_zero_pred_noise: bool = True  # Enforce zero pred_noise for conditioned data when learning p(u_[1, T-1] | u0)
    is_condition_uT_zero_pred_noise: bool = True  # Enforce zero pred_noise for conditioned data when learning p(u_[1, T-1] | uT)
    
    # Residual settings
    condition_on_residual: Optional[str] = None  # Options: None, residual_gradient
    residual_on_u0: bool = False         # Whether to feed u0 or ut into UNet when using residual conditioning
    
    # Sampling settings
    recurrence: bool = False             # Whether to use recurrence in Universal Guidance
    recurrence_k: int = 1                # Number of recurrence iterations
    using_ddim: bool = False             # Whether to use DDIM sampler
    ddim_eta: float = 0.0                # DDIM eta parameter
    ddim_sampling_steps: int = 1000      # Number of DDIM sampling steps
    
    @property
    def base_dir(self) -> str:
        """Get the base directory path."""
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @property
    def datasets_dir(self) -> str:
        """Get the datasets directory path."""
        return os.path.join(self.base_dir, "datasets")
    
    @property
    def experiments_dir(self) -> str:
        """Get the experiments directory path."""
        return os.path.join(self.base_dir, "experiments")
    
    @property
    def checkpoints_dir(self) -> str:
        """Get the model checkpoints directory path."""
        return os.path.join(self.experiments_dir, "checkpoints")

def get_train_config(exp_id: str, model_size: str) -> TrainConfig:
    """Get configuration template based on some input values."""
    if model_size == "turbo":
        return TrainConfig(
            exp_id=exp_id,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            train_num_steps=200000,
        )
    else:
        return TrainConfig(
            exp_id=exp_id,
        ) 