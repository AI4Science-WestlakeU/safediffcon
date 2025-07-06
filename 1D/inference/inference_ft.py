import os
import json
import yaml
import torch
import pdb
from typing import Dict, Optional, Tuple
import logging
import time
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from itertools import cycle
from accelerate import Accelerator
from ema_pytorch import EMA
import torch.optim as optim

from data.burgers import BurgersDataset
from configs.inference_config import InferenceConfig
from .conformal import ConformalCalculator
from .guidance import get_weight, normalize_weights
from .utils import get_scheduler
from utils.common import SCALER, get_target
from utils.guidance import get_finetune_guidance
from utils.metrics import evaluate_samples, control_trajectories

class InferenceFT:
    """InferenceFT class for inference and fine-tuning
    
    This implementation follows the same acceleration strategy as trainer.py,
    using the accelerate library for mixed precision and distributed training.
    """
    def __init__(
        self,
        config: InferenceConfig,
        model,
        *,
        mixed_precision_type: str = 'fp16',
        split_batches: bool = True,
        ema_decay: float = 0.995,
        ema_update_every: int = 10,
        max_grad_norm: float = 1.0,
    ):
        # Save config
        self.config = config
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision_type,
            split_batches=split_batches,
            device_placement=True,
        )
        
        # Setup model
        self.model = model
        
        # Initialize optimizer
        if self.config.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.finetune_lr,
                momentum=0.9
            )
        elif self.config.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.finetune_lr,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer == 'adamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.finetune_lr,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        self.cosine_steps = self.config.InfFT_iters * self.config.cosine_ratio

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cosine_steps, eta_min=1e-6)
        
        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        
        # Initialize EMA
        if self.accelerator.is_main_process:
            self.ema = EMA(
                model,
                beta=ema_decay,
                update_every=ema_update_every
            )
            self.ema.to(self.device)
        
        # Setup data
        self.setup_data()
        
        # Initialize conformal calculator
        self.conformal_calculator = ConformalCalculator(self.get_model_for_inference(), self.config)
        
        # Setup guidance
        self.setup_guidance()
        
        # Training parameters
        self.max_grad_norm = max_grad_norm
        self.step = 0
        self.Q = 0  # Initialize quantile
        self.crt_epoch=0
        
    @property
    def device(self):
        return self.accelerator.device
    
    def warmup_lr_lambda(self, step):
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return 1

    def setup_data(self):
        """Setup datasets and dataloaders"""
        # Calibration dataset
        self.cal_dataset = BurgersDataset(
            split="cal",
            root_path=self.config.datasets_dir,
            dataset=self.config.dataset,
            is_normalize=True,
            config=self.config
        )
        self.cal_loader = DataLoader(
            self.cal_dataset,
            batch_size=self.config.cal_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.cal_loader = self.accelerator.prepare(self.cal_loader)
        self.cal_loader_iter = cycle(self.cal_loader)
        
        # Test dataset
        self.test_dataset = BurgersDataset(
            split="test",
            root_path=self.config.datasets_dir,
            dataset=self.config.dataset,
            is_normalize=True,
            is_need_idx=True,
            config=self.config
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = self.accelerator.prepare(self.test_loader)
        
        logging.info(f"Calibration batch size: {self.config.cal_batch_size}")
        logging.info(f"Test batch size: {self.config.test_batch_size}")
        
    def setup_guidance(self):
        """Setup guidance function for test sampling"""
        self.guidance_fn = lambda x: get_finetune_guidance(
            self.config,
            x,
            self.Q
        ) if any(self.config.guidance_weights.values()) else None
        
        self.J_scheduler = get_scheduler(self.config.J_scheduler)
        self.w_scheduler = get_scheduler(self.config.w_scheduler)

    def get_model_for_inference(self) -> torch.nn.Module:
        """Get the appropriate model for inference (EMA if available)"""
        if self.accelerator.is_main_process and hasattr(self, 'ema'):
            return self.ema.ema_model
        return self.accelerator.unwrap_model(self.model)

    def get_finetune_reweights(self, data_loader, mode):
        """Get reweights for fine-tuning loss"""
        weights = []
        for state, idx in tqdm(data_loader, desc=f"Getting {mode} loss reweights"):
            state = state.to(self.device)
            weights.append(get_weight(state, self.Q, self.config))
        weights = torch.cat(weights)
        normalized_weights = normalize_weights(weights)
        return normalized_weights

    def _compute_loss(self, state: torch.Tensor, reweight: torch.Tensor) -> torch.Tensor:
        """Compute loss with accelerator's autocast"""
        with self.accelerator.autocast():
            loss_diff = self.model(state, mean=False)
            return (reweight * loss_diff).mean()

    def finetune_step(self, prediction):
        """Single fine-tuning step with acceleration"""
        self.model.train()

        if self.config.use_max_safety:
            s = prediction[:,2,:11,:].amax(dim=(-1,-2))
        else:
            s = prediction[:,2,:11,:].amax(dim=(-1,-2))
        
        with self.accelerator.autocast():
            obj = torch.maximum(s + self.Q - self.config.u_bound**2, torch.zeros_like(s))
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(obj, torch.zeros_like(s))
        
        self.accelerator.backward(loss)

        logging.info(f"loss: {loss.item()}")
        
        # Clip gradients
        if self.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Wait for all processes
        self.accelerator.wait_for_everyone()
        
        # Update EMA model
        self.step += 1
        if self.accelerator.is_main_process and hasattr(self, 'ema'):
            self.ema.update()
        
        return {
            'loss': loss.item(),
        }

    def run_epoch(self) -> Dict:
        """Run one epoch of fine-tuning and evaluation"""

        all_prediction = []
        train_metrics = []
        
        for test_state, idx in self.test_loader:
            prediction = self.inference(test_state, is_backward=True)
            all_prediction.append(prediction)

            batch_loss = self.finetune_step(prediction)
            train_metrics.append(batch_loss)

        if self.crt_epoch == self.config.InfFT_iters - 1:
            pass
        else:
            quantile = self.calibrate()
            self.Q = quantile

        eval_metrics = self.evaluate_model()

        # Calculate metrics
        avg_train_metrics = {}
        for key in train_metrics[0].keys():
            avg_train_metrics[key] = sum(m[key] for m in train_metrics) / len(train_metrics)
        
        epoch_metrics = {
            'epoch': self.crt_epoch,
            'train': {k: round(v, 6) for k, v in avg_train_metrics.items()},
            'eval': eval_metrics,
            'quantile': round(self.Q.item() if isinstance(self.Q, torch.Tensor) else self.Q, 6)
        }
        
        return epoch_metrics

    def evaluate_model(self) -> Dict:
        """Evaluate current model on test dataset"""
        logging.info("Starting evaluation...")
        all_predictions = []
        all_controlled = []
        
        model = self.get_model_for_inference()
        model.eval()
        
        with torch.no_grad():
            for test_state, idx in self.test_loader:
                test_state = test_state.to(self.device)
                predictions = self.inference(test_state)
                u_controlled = control_trajectories(predictions, self.config.nt)
                
                all_predictions.append(predictions)
                all_controlled.append(u_controlled)
        
        # Concatenate results
        predictions = torch.cat(all_predictions)
        controlled = torch.cat(all_controlled)
        u_target = get_target(
            list(range(self.config.n_test_samples)), 
            dataset=self.config.dataset,
            is_normalize=False
        ).to(self.device)

        metrics = evaluate_samples(
            diffused=predictions,
            u_controlled=controlled,
            u_target=u_target,
            nt=self.config.nt,
            u_bound=self.config.u_bound,
            use_max_safety=self.config.use_max_safety
        )

        logging.info("Evaluation completed")
        return metrics

    def calibrate(self) -> float:
        """Run calibration phase"""
        logging.info("Starting calibration phase...")
        
        scores, weights, states = self.conformal_calculator.get_conformal_scores(
            self.cal_loader_iter, self.Q
        )
        
        quantile = self.conformal_calculator.calculate_quantile(
            scores, weights, states, self.config.alpha
        )
        
        return quantile
        
    def inference(self, state: torch.Tensor, is_backward: bool = False) -> torch.Tensor:
        """Run inference phase"""
        state = state.to(self.device)
        if is_backward:
            output = self.model.sample(
                batch_size=state.shape[0],
                clip_denoised=True,
                u_init=state[:,0,0,:],
                u_final=state[:,0,self.config.nt-1,:],
                guidance_u0=True,
                nablaJ=self.guidance_fn,    # CHOICE: None or self.guidance_fn
                J_scheduler=self.J_scheduler,
                w_scheduler=self.w_scheduler,
                enable_grad=is_backward,
                device=self.device,
            )
        else:
            model = self.get_model_for_inference()
            output = model.sample(
                batch_size=state.shape[0],
                clip_denoised=True,
                u_init=state[:,0,0,:],
                u_final=state[:,0,self.config.nt-1,:],
                guidance_u0=True,
                nablaJ=self.guidance_fn,    # CHOICE: None or self.guidance_fn
                J_scheduler=self.J_scheduler,
                w_scheduler=self.w_scheduler,
                enable_grad=is_backward,
                device=self.device,
            )
        pred = output * SCALER
        return pred
    
    def save(self, path: str, milestone: Optional[int] = None):
        """Save model checkpoint"""
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.optimizer.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if hasattr(self.accelerator, 'scaler') else None
        }
        
        # Save EMA state if available
        if self.accelerator.is_main_process and hasattr(self, 'ema'):
            data['ema'] = self.ema.state_dict()
            
        if milestone is not None:
            path = f"{path}/model-{milestone}.pt"
        torch.save(data, path)
        
    def load(self, path: str, milestone: Optional[int] = None):
        """Load model checkpoint"""
        if milestone is not None:
            path = f"{path}/model-{milestone}.pt"
            
        data = torch.load(path, map_location=self.device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        
        self.step = data['step']
        self.optimizer.load_state_dict(data['opt'])
        
        if hasattr(self.accelerator, 'scaler') and 'scaler' in data and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])
            
        if self.accelerator.is_main_process and hasattr(self, 'ema') and 'ema' in data:
            self.ema.load_state_dict(data['ema'])
    
    def run(self) -> Dict:
        """Run multiple training epochs and record results"""
        # Setup finetune directory
        self.finetune_dir = os.path.join(
            self.config.experiments_dir,
            self.config.exp_id,
            'test',
            'seed' + str(self.config.seed),
            'alpha' + str(self.config.alpha),
            'lr' + str(self.config.finetune_lr) + '_iters' + str(self.config.InfFT_iters),
            'wd' + str(self.config.weight_decay),
            'guidance' + str(self.config.guidance_weights["w_score"])
        )
        os.makedirs(self.finetune_dir, exist_ok=True)
        
        # Save config
        config_dict = self.config.__dict__.copy()
        config_dict['device'] = str(config_dict['device'])
        with open(os.path.join(self.finetune_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Run training epochs
        start_time = time.time()
        all_metrics = []
        
        for epoch in range(self.config.InfFT_iters):
            self.crt_epoch = epoch
            if self.crt_epoch == self.config.InfFT_iters - 1:
                pass
            else:
                epoch_metrics = self.run_epoch()
                all_metrics.append(epoch_metrics)
                
                # Save results
                results_file = os.path.join(self.finetune_dir, 'results.yaml')
                with open(results_file, 'w') as f:
                    yaml.dump(all_metrics, f, default_flow_style=False)
                
            # Save checkpoint
            # if (epoch + 1) % self.config.save_every == 0:
            #     self.save(self.finetune_dir, milestone=epoch)

        total_time = time.time() - start_time
        logging.info(f"Fine-tuning completed in {total_time/60:.2f} minutes")

        return all_metrics
    