import torch
import logging
import numpy as np
from typing import Tuple, List
from torch.utils.data import DataLoader

from configs.inference_config import InferenceConfig
from utils.guidance import calculate_weight, normalize_weights
from utils.common import SCALER
from utils.metrics import calculate_safety_score

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
        self.nt_total = config.nt_total
        self.w_obj = config.guidance_weights['w_obj']
        self.w_safe = config.guidance_weights['w_safe']
        self.guidance_scaler = config.guidance_scaler
        self.safety_threshold = config.safety_threshold

        self.state = None
        self.idx = None

    def get_conformal_scores(self, dataloader, cal_targets, Q) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate weighted conformal scores
        
        Args:
            dataloader: Cyclic dataloader for calibration dataset
            cal_targets: Calibration targets, shape (batch, state_dim, time), scaled, cpu
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
            state, self.idx = next(dataloader)  # [B, channels, padded_time]
            states.append(state)
            state = state.to(self.device)
            state_target_batch = cal_targets[self.idx].to(self.device)

            with torch.no_grad():
                output = self.model.sample(
                    batch_size=state.shape[0],
                    clip_denoised=True,
                    guidance_u0=False,
                    device=self.device,
                    u_init=state[:, :3, 0],
                    u_final=state[:, [0, 2], :self.nt_total],
                    w_groundtruth=state[:, 3:, :],    # NOTE: use cleaning actions for sampling on calibration set
                    nablaJ=None,
                    J_scheduler=None,
                    w_scheduler=None,
                    enable_grad=False,
                )
            
            # Calculate weights for distribution shift
            weight = calculate_weight(state, state_target_batch, 
                                      self.nt_total,
                                      Q, 
                                      self.safety_threshold, 
                                      self.w_obj, 
                                      self.w_safe, 
                                      self.guidance_scaler)
            if self.config.finetune_set == 'train' and self.config.use_guidance:
                weight = weight * calculate_weight(state, state_target_batch, 
                                            self.nt_total,
                                            Q, 
                                            self.safety_threshold, 
                                            self.w_obj, 
                                            self.w_safe, 
                                            self.guidance_scaler)
            if self.config.finetune_set == 'test' and not self.config.wo_post_train:
                weight = weight * calculate_weight(state, 
                                            state_target_batch, 
                                            self.config.nt_total, 
                                            self.config.finetune_quantile, 
                                            self.config.safety_threshold, 
                                            self.config.finetune_guidance_weights['w_obj'], 
                                            self.config.finetune_guidance_weights['w_safe'], 
                                            self.config.finetune_guidance_scaler)
            weights.append(weight)
            
            pred = output * SCALER.to(self.device)
            state = state * SCALER.to(self.device)
            
            s_pred = calculate_safety_score(pred[:,:,:self.config.nt_total])
            s_target = calculate_safety_score(state[:,:,:self.config.nt_total])
            batch_scores = (s_pred - s_target).abs()
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
            scores: Conformal scores set
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
        
        logging.info(f"===Calculate quantile, No.{rank}")
        # logging.info(f"Sorted conformal scores: {sorted_scores}")
        logging.info(f"{alpha}-th quantile: {sorted_scores[rank]}")
        
        quantile = scores[q_index]

        return quantile
