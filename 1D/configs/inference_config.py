import os
from dataclasses import dataclass
from typing import Optional, List, Union
from pathlib import Path

@dataclass
class InferenceConfig:
    """Configuration for inference time finetuning
    """
    # Experiment settings
    tuning_id: str
    exp_id: str = "pretrain-turbo"
    seed: int = 42
    gpu_id: int = 0
    device: str = "cuda"
    
    # Dataset settings
    dataset: str = "free_u_f_1e5"
    nt: int = 11
    pad_size: int = 16

    u_bound: float = 0.8
    use_max_safety: bool = True

    train_batch_size: int = 380

    n_cal_samples: int = 1000
    cal_batch_size: int = 250
    num_cal_batch: int = 4

    n_test_samples: int = 50
    test_batch_size: int = 50

    finetune_subset_size: int = 10000
    finetune_batch_size: int = 380
    
    # training settings
    optimizer: str = "adamW"
    finetune_lr: float = 1e-5
    weight_decay: float = 1e-4

    use_grad_norm: bool = False
    loss_weights: Optional[dict] = None

    InfFT_iters: int = 5
    InfFT_Q: Optional[float] = None
    cosine_ratio: float = 1

    # conformal settings
    alpha: float = 0.98

    # Load model settings
    checkpoint: int = 171
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
    train_on_padded_locations: bool = False
    is_condition_u0: bool = True
    is_condition_uT: bool = True
    is_condition_u0_zero_pred_noise: bool = True
    is_condition_uT_zero_pred_noise: bool = True
    
    # UNet settings
    dim: int = 128
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
        return os.path.join(self.experiments_dir, "checkpoints")

    def __post_init__(self):
        """Set default values that depend on other fields"""        
        if self.loss_weights is None:
            self.loss_weights = {
                "loss_train": 1.0,
                "loss_test": 0.0,
            }
        
        if self.guidance_scaler is None:
            self.guidance_scaler = 1.0

        if self.guidance_weights is None:
            self.guidance_weights = {
                "wf": 0.0,
                "wu": 0.0,
                "w_score": 1.0,
            }
            
        if self.dim_mults is None:
            self.dim_mults = [1, 2, 4, 8]

def get_inference_config(model_size: str = "base", exp_id: str = "turbo-repeat", tuning_id: str = "test") -> InferenceConfig:
    if tuning_id == "reproduce-ft":
        return InferenceConfig(
            exp_id="turbo-1",
            tuning_id=tuning_id,
            guidance_weights={"w_score": 500},
        )        
    
    if model_size == "turbo" and tuning_id != "reproduce-ft":
        return InferenceConfig(
            exp_id=exp_id,
            tuning_id=tuning_id,
            dim=128,
            dim_mults=(1, 2, 4, 8),
            train_num_steps=200000,
            ddim_eta=1.0,
            ddim_sampling_steps=200,
        )

def load_config_from_args(args) -> InferenceConfig:
    """Load configuration from argparse arguments"""
    config = InferenceConfig(
        exp_id=args.exp_id,
        tuning_id=args.tuning_id,
        gpu_id=args.gpu_id,
    )
    
    return config