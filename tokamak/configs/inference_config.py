import os
from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path

@dataclass
class InferenceConfig:
    """Configuration for inference
    
    Args:
  
    """
    # Experiment settings
    tuning_dir: str
    tuning_id: str
    exp_id: str = "SafeDiffCon"
    seed: int = 42
    gpu_id: int = 0
    device: str = "cuda"
    
    # Dataset settings
    dataset: str = "tokamak"
    nt_total: int = 122
    pad_size: int = 128
    safety_threshold: float = 4.98

    n_test_samples: int = 50
    test_batch_size: int = 50

    n_cal_samples: int = 1000
    cal_batch_size: int = 1000
    num_cal_batch: int = 1

    train_batch_size: int = 1000    # batch_size of training data in per finetune step
    
    # finetuning settings
    finetune_set: str = "train"
    use_guidance: bool = False
    backward_finetune: bool = False
    optimizer: str = "adam"
    finetune_lr: float = 1e-5
    finetune_epoch: int = 2    # number of finetune epochs
    finetune_steps: int = 1   # number of finetune steps in one epoch
    loss_weights: Optional[dict] = None
    use_grad_norm: bool = False

    # conformal settings
    alpha: float = 0.9

    # Load model settings
    checkpoint_dir: str = "checkpoints"
    wo_post_train: bool = True
    post_train_id: str = "total"
    checkpoint: int = 190
    train_num_steps: int = 200000
    checkpoint_interval: int = 1000

    # Diffusion settings
    using_ddim: bool = True
    ddim_eta: float = 1.0
    ddim_sampling_steps: int = 200
    J_scheduler: str = "constant"
    w_scheduler: str = "constant"
             
    # Guidance settings
    guidance_weights: Optional[dict] = None
    guidance_scaler: Optional[float] = None
    
    # Model conditioning
    is_condition_u0: bool = True
    is_condition_uT: bool = True
    is_condition_u0_zero_pred_noise: bool = True
    is_condition_uT_zero_pred_noise: bool = True
    
    # UNet settings
    dim: int = 64
    resnet_block_groups: int = 1
    dim_mults: Optional[List[int]] = None

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
        if self.finetune_set == "train" or self.wo_post_train:
            return os.path.join(self.experiments_dir, self.checkpoint_dir)
        else:
            return os.path.join(self.experiments_dir, self.exp_id, self.post_train_id)

    def __post_init__(self):
        """Set default values that depend on other fields"""        
        if self.loss_weights is None:
            self.loss_weights = {
                "loss_train": 1.0,
                "loss_test": 0.0,
            }

        if self.guidance_weights is None:
            self.guidance_weights = {
                "w_obj": 0.0,
                "w_safe": 1.0,
            }
        if self.guidance_scaler is None:
            self.guidance_scaler = 1.0
            
        if self.dim_mults is None:
            self.dim_mults = [1, 2, 4, 8]

def get_inference_config(model_size: str = "large", exp_id: str = "posttrain", tuning_dir: str = "finetune", tuning_id: str = "test") -> InferenceConfig:
    """Get evaluation configuration"""
    if model_size == "large":
        return InferenceConfig(
            exp_id=exp_id,
            tuning_dir=tuning_dir,
            tuning_id=tuning_id,
            dim=256,
            dim_mults=(1, 2, 4, 8),
            train_num_steps=200000,
            ddim_eta=1.0,
            ddim_sampling_steps=200,
        )
    elif model_size == "turbo":
        return InferenceConfig(
            exp_id=exp_id,
            tuning_dir=tuning_dir,
            tuning_id=tuning_id,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            train_num_steps=200000,
            ddim_eta=1.0,
            ddim_sampling_steps=200,
        )