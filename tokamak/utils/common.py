import os
import torch
import random
import yaml
import logging
import numpy as np
from typing import Optional, Dict, Any, Union, List

from data.tokamak_dataset import TokamakDataset
from configs.pretrain_config import TrainConfig
from configs.inference_config import InferenceConfig
from configs.eval_config import EvalConfig
from model.diffusion import GaussianDiffusion
from model.unet import Unet1D
from model.trainer import Trainer
SCALER = torch.tensor([2, 7, 2, 1, 2, 2, 2, 2, 1, 1, 2, 3]).reshape(12,1)

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
    device: Optional[torch.device] = None,
    is_normalize: bool = False,
    split: str = "test"
) -> torch.Tensor:
    """Get target trajectory from test dataset
    
    Args:
        is_normalize: guidance->False, eval->False
    Returns:
        target: Target trajectory, shape (batch, state_dim, time)
    """
    dataset = TokamakDataset(
        split=split,
        is_normalize=is_normalize
    )
    
    if isinstance(target_i, int):
        target = dataset[target_i]
        target = target.unsqueeze(0)
    else:
        target = torch.stack([dataset[i] for i in target_i], dim=0)
    
    state_target = target[:, :3, :dataset.nt_total]
    
    if device is not None:
        state_target = state_target.to(device)
    
    return state_target

def build_model(config: Union[EvalConfig, InferenceConfig], 
                dataset: TokamakDataset,
                ) -> GaussianDiffusion:
    channels = dataset[0].shape[0]
    sim_time_stamps = dataset.pad_size

    unet = Unet1D(
        dim=config.dim,
        dim_mults=config.dim_mults,
        channels=channels,
        resnet_block_groups=config.resnet_block_groups,
    )

    model = GaussianDiffusion(
        unet,
        seq_length=sim_time_stamps,
        nt=dataset.nt_total,
        use_conv2d=False,
        temporal=False,
        guidance_u0=True,
        is_condition_u0=config.is_condition_u0,
        is_condition_uT=config.is_condition_uT,
        is_condition_u0_zero_pred_noise=config.is_condition_u0_zero_pred_noise,
        is_condition_uT_zero_pred_noise=config.is_condition_uT_zero_pred_noise,
        sampling_timesteps=config.ddim_sampling_steps if config.using_ddim else 1000,
        ddim_sampling_eta=config.ddim_eta,
    ).to(config.device)

    return model

def load_model(config: Union[TrainConfig, EvalConfig, InferenceConfig],
               dataset: TokamakDataset,
               ) -> GaussianDiffusion:
    model = build_model(config, dataset)

    if config.finetune_set == "train" or config.wo_post_train:
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
    else:
        model_path = config.checkpoints_dir
        cp = torch.load(os.path.join(model_path, f'model-{config.checkpoint}.pth'), map_location=config.device)
        model.load_state_dict(cp['model'])
        model.to(config.device)
        config.finetune_quantile = cp['quantile']
        config.finetune_alpha = cp['config'].alpha
        config.finetune_guidance_scaler = cp['config'].guidance_scaler if not cp['config'].use_guidance \
                                        else 2 * cp['config'].guidance_scaler
        config.finetune_guidance_weights = cp['config'].guidance_weights
    return model, model_path