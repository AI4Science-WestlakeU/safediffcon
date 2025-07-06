import os
import logging
import torch
import json
from typing import Optional
from typing import Tuple
from datetime import datetime

from data.tokamak_dataset import TokamakDataset
from model.unet import Unet1D
from model.diffusion import GaussianDiffusion
from model.trainer import Trainer
from utils.common import set_seed, save_config, setup_logging, build_model
from configs.pretrain_config import TrainConfig, get_train_config

# new version
def setup_experiment(config: TrainConfig) -> Tuple[str, str]:
    """Setup experiment directory and save metadata
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple[str, str]: Paths to experiment and model directories
    """
    # Create directories
    exp_dir = os.path.join(config.experiments_dir, config.exp_id)
    model_dir = os.path.join(config.checkpoints_dir, config.exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save experiment config
    save_config(config.__dict__, os.path.join(exp_dir, 'config.json'))
    
    # Update pretrain metadata
    metadata_path = os.path.join(config.experiments_dir, 'metadata/pretrain.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
    metadata[config.exp_id] = {
        'date': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'description': config.description if hasattr(config, 'description') else '',
        'config': config.__dict__
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    return exp_dir, model_dir

def train(config: TrainConfig):
    """Main training function
    
    Args:
        config: Training configuration
        device: Device to place model and data on
    """
    # Setup
    exp_dir, model_dir = setup_experiment(config)
    setup_logging(exp_dir)
    set_seed(config.seed)

    # setup device
    torch.cuda.set_device(config.gpu_id)
    config.device = torch.device(f"cuda:{config.gpu_id}")
    logging.info(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    
    # Load dataset
    dataset = TokamakDataset(split='train')
    logging.info(f'Dataset loaded: {len(dataset)} samples')
    logging.info(f'Sample shape: {dataset[0].shape}')
    
    # Build model
    model = build_model(config, dataset)
    
    # Setup trainer
    trainer = Trainer(
        model,
        dataset,
        results_folder=model_dir,
        train_num_steps=config.train_num_steps,
        save_and_sample_every=config.checkpoint_interval,
        train_lr=config.lr,
    )
    
    # Train
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed")

def main():
    """Main entry point"""
    
    config = get_train_config(exp_id="SafeDiffCon", model_size="turbo")
    
    train(config)

if __name__ == "__main__":
    main()
