import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from accelerate import Accelerator
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
from ema_pytorch import EMA
import logging

from ddpm.data_2d import Smoke


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def get_device():
    return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        frames,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        standard_fixed_ratio=1, # used in standard sampling
        device=None
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.image_size = image_size
        self.frames = frames
        self.standard_fixed_ratio = standard_fixed_ratio
        
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        register_buffer('loss_weight', maybe_clipped_snr / snr)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def model_predictions(self, shape, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, design_fn = None):
        pred_noise = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        # guidance
        if design_fn is not None:
            with torch.enable_grad():
                x_clone = x_start.clone().detach().requires_grad_()
                g = design_fn(x_clone)
                grad_final = self.standard_fixed_ratio * g
                pred_noise = pred_noise + grad_final

        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)
        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, shape, x, t, x_self_cond = None, clip_denoised = True, design_fn = None):
        preds = self.model_predictions(shape, x, t, x_self_cond, design_fn = design_fn)
        x_start = preds.pred_x_start
        if clip_denoised:
            x_start.clamp_(-1., 1.) 
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def sample_noise(self, shape, device):
        return torch.randn(shape, device = device)

    @torch.no_grad()
    def p_sample(self, shape, x, t: int, x_self_cond = None, clip_denoised = True, design_fn = None):
        b, *_, device = *x.shape, x.device 
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        p_mean_variance = self.p_mean_variance(shape, x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised, design_fn = design_fn)

        model_mean, _, model_log_variance, x_start = p_mean_variance
        noise = self.sample_noise(model_mean.shape, device) if t > 0 else 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred, x_start
            

    @torch.no_grad()
    def p_sample_loop(self, shape, design_fn=None, return_all_timesteps=None, enable_grad=False, init=None, control=None, device=None):
        b, f, c, h, w = shape 
        batch, device = shape[0], self.betas.device
        # self.betas = self.betas.to(device)
        noise_state = self.sample_noise([b, f, c, h, w], device) 
        # condition
        assert init is not None
        init = init.to(device)
        noise_state[:, 0, 0] = init

        if control is not None:
            control = control.to(device)
            noise_state[:, :, 3:5] = control

        x = noise_state
        x_start = None
        # for t in tqdm(reversed(range(0, 2)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            if t != 0 or not enable_grad:
                self_cond = x_start if self.self_condition else None
                x, x_start = self.p_sample(shape, x, t, self_cond, design_fn=design_fn)
                x[:, 0, 0] = init
                if control is not None:
                    x[:, :, 3:5] = control
                final_result = x
            else: # enable grad in the final loop
                with torch.enable_grad():
                    self_cond = x_start if self.self_condition else None
                    x, x_start = self.p_sample(shape, x, t, self_cond, design_fn=design_fn)
                    x[:, 0, 0] = init
                    if control is not None:
                        x[:, :, 3:5] = control
                    final_result = x
        return final_result

    @torch.no_grad()
    def ddim_sample(self, shape, design_fn=None, enable_grad=False, init=None, control=None, device=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
            
        # condition
        init = init.to(device)
        img[:, 0, 0] = init

        if control is not None:
            control = control.to(device)
            img[:, :, 3:5] = control

        x_start = None
        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            if time_next >= 0 or not enable_grad:
                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

                self_cond = x_start if self.self_condition else None

                pred_noise, x_start, *_ = self.model_predictions(shape, img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, \
                                                design_fn = design_fn)

                if time_next < 0:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

                img[:, 0, 0] = init
                if control is not None:
                    img[:, :, 3:5] = control
            else: # enable grad in the final loop
                # print("time_next < 0")
                with torch.enable_grad():
                    time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

                    self_cond = x_start if self.self_condition else None

                    pred_noise, x_start, *_ = self.model_predictions(shape, img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, \
                                                    design_fn = design_fn)

                    if time_next < 0:
                        img = x_start
                        continue

                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]

                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()

                    noise = torch.randn_like(img)

                    img = x_start * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise

                    img[:, 0, 0] = init
                    if control is not None:
                        img[:, :, 3:5] = control
        if control is not None:
            img[:, :, 3:5] = control
        ret = img 

        return ret

    @torch.no_grad()
    def sample(self, batch_size = 16, design_fn = None, enable_grad = False, init = None, control = None, device = None):
        image_size, channels, frames = self.image_size, self.channels, self.frames
        
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        assert batch_size == init.shape[0]
        sample_size = (batch_size, frames, channels, image_size, image_size)
        
        return sample_fn(sample_size, design_fn, enable_grad=enable_grad, init=init, control=control, device = device)
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )        


    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, state_start, t, noise = None, mean = True):
        b, f, c, h, w = state_start.shape 
        noise_state = default(noise, lambda: torch.randn_like(state_start))
        # noisy sample
        state = self.q_sample(x_start = state_start, t = t, noise = noise_state)
        # condition on initial state (and reward)        
        state[:, 0, 0] = state_start[:, 0, 0]
        noise_state[:, 0, 0] = torch.zeros_like(noise_state[:, 0, 0])
        
        x_self_cond = None
        model_out = self.model(state, t, x_self_cond)
        
        loss = self.loss_fn(model_out, noise_state, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        if mean:
            return loss.mean()
        else:
            return loss

    def forward(self, state, *args, **kwargs):
        b, f, c, h, w, device, img_size, = *state.shape, state.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(state, t, *args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        dataset_path,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_path = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        is_schedule = True,
        resume = False,
        resume_step = 0,
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp
        self.dataset = dataset

        # model
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader
        if dataset == "Smoke":
            self.ds = Smoke(
                dataset_path,
                is_train=True,
            )
        else:
            assert False

        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = 16)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.resume = resume
        self.resume_step = resume_step
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        if is_schedule == True:
            self.scheduler = lr_scheduler.MultiStepLR(self.opt, milestones=[50000, 150000, 300000], gamma=0.1)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok = True)

        # step counter state

        self.step = 0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_path / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device
        print("model path: ", str(self.results_path / f'model-{milestone}.pt'))
        data = torch.load(str(self.results_path / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        print("model loaded: ", str(self.results_path / f'model-{milestone}.pt'))

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...')

        mu = torch.mean(features, dim = 0).cpu()
        sigma = torch.cov(features).cpu()
        return mu, sigma

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        log_filename = os.path.join(self.results_path, "info.log")
        logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    state, _ = next(self.dl)
                    state = state.to(device) # [batch, 32, 6, 64, 64]

                    with self.accelerator.autocast():
                        loss = self.model(state)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.step != 0 and self.step % 10 == 0:
                    pbar.set_description(f'loss: {total_loss:.4f}, LR: {self.opt.param_groups[0]["lr"]}')
                    logging.info(f'step: {self.step}, loss: {total_loss:.4f}, LR: {self.opt.param_groups[0]["lr"]}')
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        milestone = self.step // self.save_and_sample_every

                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

