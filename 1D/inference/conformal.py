import torch
import logging
import numpy as np
from typing import Tuple, List
from torch.utils.data import DataLoader

from configs.inference_config import InferenceConfig
from inference.guidance import get_weight, normalize_weights
from utils.common import SCALER

class ConformalCalculator:
    """Class for calculating conformal scores and quantiles"""
    
    def __init__(self, model, config: InferenceConfig):
        """Initialize conformal calculator
        
        Args:
            model: Diffusion model
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.device = config.device
        
    def get_conformal_scores(self, dataloader, Q: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate weighted conformal scores
        
        Args:
            dataloader: Cyclic dataloader for calibration dataset
            
        Returns:
            Tuple containing:
            - weighted_scores: Weighted conformal scores
            - normalized_weights: Normalized weights
            - states: Original states
        """
        conformal_scores = []
        weights = []
        states = []
        
        logging.info("===Start calculating conformal scores...")
        
        for i in range(self.config.num_cal_batch):
            logging.info(f"====Calculate {i}-th Batch in Calibration set")
            
            # Get next batch from cyclic dataloader
            state = next(dataloader)  # [B, 3, 16, 128]
            states.append(state)
            state = state.to(self.device)
            
            # Generate samples without guidance
            with torch.no_grad():
                output = self.model.sample(
                    batch_size=state.shape[0],
                    clip_denoised=True,
                    guidance_u0=False,
                    device=self.device,
                    u_init=state[:,0,0,:],
                    u_final=state[:,0,self.config.nt-1,:],
                    w_groundtruth=state[:,1,:,:],   # NOTE: use cleaning actions for sampling on calibration set
                    nablaJ=None,
                    J_scheduler=None,
                    w_scheduler=None,
                    enable_grad=False,
                )
            
            # NOTE: calculate weights for distribution shift
            weight = get_weight(state, Q, self.config)
            if self.config.InfFT_Q is not None:
                weight = weight * get_weight(state, self.config.InfFT_Q, self.config)
            else:
                pass
            weights.append(weight)
            
            # Calculate conformal scores
            pred = output * SCALER
            state = state * SCALER
            if self.config.use_max_safety:
                c_pred = pred[:, 2, :11, :].mean(dim=(-1,-2))
                c_target = state[:,2,:11,:].mean(dim=(-1,-2))
            else:
                c_pred = pred[:, 2, :11, :].amax(dim=(-1,-2))
                c_target = state[:,2,:11,:].amax(dim=(-1,-2))
            batch_scores = (c_pred - c_target).abs()
            conformal_scores.append(batch_scores)
            
        # Concatenate and normalize weights
        weights = torch.cat(weights)
        normalized_weights = normalize_weights(weights)
        
        return (normalized_weights * torch.cat(conformal_scores), 
                normalized_weights, 
                torch.cat(states))
                
    def calculate_quantile(self, scores: torch.Tensor, weights: torch.Tensor, 
                          states: torch.Tensor, alpha: float) -> torch.Tensor:
        """Calculate quantile
        
        Args:
            scores: Conformal scores of test set
            weights: Weights
            states: States
            alpha: Confidence level
            
        Returns:
            quantile: Calculated quantile
        """
        n = scores.shape[0]
        
        # Get index of quantile
        sorted_scores, sorted_indices = torch.sort(scores)
        rank = min(int(np.ceil(alpha * (n + 1))), n) - 1 # 'n-1' to avoid the worst case
        q_index = sorted_indices[rank]
        
        quantile = scores[q_index]
        logging.info(f"===Calculate {alpha}-th quantile, No.{rank}")
        logging.info(f"Quantile of Test set: {quantile:.4f}")
        return quantile