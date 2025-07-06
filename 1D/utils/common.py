import os
import torch
import random
import yaml
import logging
import numpy as np
from typing import Optional, Dict, Any, Union, List

from data.burgers import BurgersDataset
from configs.train_config import TrainConfig
from configs.inference_config import InferenceConfig
from configs.eval_config import EvalConfig
from configs.posttrain_config import PostTrainConfig
from model.diffusion import GaussianDiffusion
from model.unet import Unet2D
from model.trainer import Trainer
SCALER = 10.0   # normalize data into [-1, 1], before input into diffusion model
UNSCALER = 1 / SCALER # unnormalize data, after output from diffusion model

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def none_or_str(value: str) -> Optional[str]:
    """Convert 'none' string to None, otherwise return the string.
    
    Args:
        value: Input string
        
    Returns:
        None if value.lower() is 'none', otherwise the original string
    """
    if value.lower() == 'none':
        return None
    return value

def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing

def setup_logging(exp_dir: str):
    """Setup logging configuration"""
    os.makedirs(exp_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(exp_dir, 'run.log')),
            logging.StreamHandler()
        ]
    )

def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        return yaml.safe_load(f) 
    
def get_target(
    target_i: Union[int, List[int]], 
    dataset: str = 'free_u_f_1e5',
    device: Optional[torch.device] = None,
    is_normalize: Optional[bool] = False,
    split: str = "test"
) -> torch.Tensor:
    """Get target trajectory from test dataset
    
    Args:
        is_normalize: guidance->False, eval->False
    """
    dataset = BurgersDataset(
        split=split,
        root_path="datasets",
        dataset=dataset,
        is_normalize=is_normalize
    )
    
    if isinstance(target_i, int):
        target = dataset[target_i]
        target = target.unsqueeze(0)
    else:
        target = torch.stack([dataset[i] for i in target_i], dim=0)
    
    target = target[:, 0, :dataset.nt_total, :]
    
    if device is not None:
        target = target.to(device)
    
    return target

def build_model(config: Union[EvalConfig, InferenceConfig, PostTrainConfig], 
                dataset: BurgersDataset,
                ) -> GaussianDiffusion:
    """Build diffusion model"""
    channels = dataset[0].shape[0]
    sim_time_stamps, sim_space_grids = dataset.pad_size, dataset.nx

    unet = Unet2D(
        dim=config.dim,
        dim_mults=config.dim_mults,
        channels=channels,
        resnet_block_groups=config.resnet_block_groups,
    )

    model = GaussianDiffusion(
        unet,
        seq_length=(sim_time_stamps, sim_space_grids),
        use_conv2d=True,
        temporal=True,
        train_on_padded_locations=config.train_on_padded_locations,
        is_condition_u0=config.is_condition_u0,
        is_condition_uT=config.is_condition_uT,
        condition_idx=dataset.nt_total - 1,
        is_condition_u0_zero_pred_noise=config.is_condition_u0_zero_pred_noise,
        is_condition_uT_zero_pred_noise=config.is_condition_uT_zero_pred_noise,
        sampling_timesteps=config.ddim_sampling_steps if config.using_ddim else 1000,
        ddim_sampling_eta=config.ddim_eta,
    ).to(config.device)

    return model

def load_model(config: Union[TrainConfig, EvalConfig, InferenceConfig, PostTrainConfig],
               dataset: BurgersDataset,
               ) -> GaussianDiffusion:
    """Load model"""

    model = build_model(config, dataset)

    model_path = os.path.join(config.checkpoints_dir, config.exp_id)
    
    trainer = Trainer(
        model,
        dataset,
        results_folder=model_path, 
        train_num_steps=config.train_num_steps, 
        save_and_sample_every=config.checkpoint_interval,
    )
    # load state dict into model in class Trainer
    trainer.load(config.checkpoint)
    
    return model, model_path