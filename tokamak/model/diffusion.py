import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

from functools import partial
from collections import namedtuple
from torch.cuda.amp import autocast
from einops import reduce

from model.model_utils import linear_beta_schedule, cosine_beta_schedule,\
        default, extract, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one


# constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# gaussian diffusion trainer class
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        nt = 122,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = False, # should be false
        guidance_u0 = True,
        residual_on_u0 = False, 
        # conv choice
        temporal = False, # Must be True when using 2d conv
        use_conv2d = False, # enabled when temporal == True
        # condition choice
        is_condition_u0 = True, 
        is_condition_uT = True, 
        is_condition_u0_zero_pred_noise = True, 
        is_condition_uT_zero_pred_noise = True, 
        # unnecessary
        train_on_padded_locations=True, 
        expand_condition=False, 
    ):
        '''
        Arguments:
            temporal: if conv along the time dimension
            use_conv2d: if using space+time 2d conv

        '''

        super().__init__()

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        if temporal:
            # use conv on the temporal axis to capture time correlation
            self.temporal = True
            # if False: first conv along time, then conv along space
            # True:  \int f(x, t) g(x, t)       dxdt
            # False: \int f(x, t) g_x(x) g_t(t) dxdt
            # The second one looks like decoupling x and t in some sense
            # \int f gx gt dxdt = \int gt (\int f gx dx) dt 
            # (FT of x and t)-> F(f) F(gx) F(gt)
            # I am thinking about if there are any relations that this above 
            # formulation cannot capture... Not sure at all
            self.conv2d = use_conv2d 
            assert type(seq_length) is tuple and len(seq_length) == 2, \
                "should be a tuple of (Nt, Nx) (time evolution of a 1-d function)"
            self.traj_size = seq_length

        else:
            assert not use_conv2d, 'must set temporal to True when using 2d conv!'
            self.seq_length = seq_length
            self.temporal = False

        self.objective = objective
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, (
            'objective must be either pred_noise (predict noise) or '
            'pred_x0 (predict image start) or pred_v (predict v '
            '[v-parameterization as defined in appendix D of progressive '
            'distillation paper, used in imagen-video successfully])'
        )

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_prev = F.pad(alphas[:-1], (1, 0), value = 1.)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) 
        # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        self.alphas = alphas.to(torch.float32).clone() # to make compatible with previous trained models
        self.alphas_prev = alphas_prev.to(torch.float32).clone() # to make compatible with previous trained models
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

        # calculate loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)


        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.guidance_u0 = guidance_u0 # guidance calculated on predicted u. 0: diffusion step
        self.is_condition_u0 = is_condition_u0 # condition on u_{t=0}
        self.is_condition_uT = is_condition_uT # condition on u_{t=T}
        self.is_condition_u0_zero_pred_noise = is_condition_u0_zero_pred_noise
        self.is_condition_uT_zero_pred_noise = is_condition_uT_zero_pred_noise
        self.expand_condition = expand_condition
        self.train_on_padded_locations = train_on_padded_locations
        self.nt = nt

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
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

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False, **kwargs):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        nablaJ, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
        
        if self.objective == 'pred_noise':
            if 'pred_noise' in kwargs and kwargs['pred_noise'] is not None:
                pred_noise = kwargs['pred_noise']
                assert self.guidance_u0 is False, 'guidance should be w.r.t. ut'
            else:
                pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            
            # guidance
            if self.guidance_u0:

                with torch.enable_grad():
                    x_clone = x_start.clone().detach().requires_grad_()
                    pred_noise = may_proj_guidance(pred_noise, nablaJ(x_clone) * nablaJ_scheduler(t[0].item()))
                
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, **kwargs):
        preds = self.model_predictions(x, t, x_self_cond, **kwargs)
        x_start = preds.pred_x_start

        # NOTE: seems that if no clamp, result would be problematic
        if kwargs['clip_denoised']:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start, preds.pred_noise

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, **kwargs):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(
                                                                    x = x, 
                                                                    t = batched_times, 
                                                                    x_self_cond = x_self_cond, 
                                                                    **kwargs)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start, pred_noise

    def recurrent_sample(self, x_tm1, t: int):
        b, *_, device = *x_tm1.shape, x_tm1.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)

        alpha_t = extract(self.alphas.to(device), batched_times, x_tm1.shape)
        alpha_tm1 = extract(self.alphas_prev.to(device), batched_times, x_tm1.shape)

        xtm1_coef, noise_coef = torch.sqrt(alpha_t / alpha_tm1), torch.sqrt(1 - (alpha_t / alpha_tm1))
        noise = noise_coef * torch.randn_like(x_tm1) if t > 0 else 0. # no noise if t == 0? 
        x_t = xtm1_coef * x_tm1 + noise
        return x_t


    def get_guidance_options(self, **kwargs):
        if 'nablaJ' in kwargs and kwargs['nablaJ'] is not None: # guidance
            nabla_J = kwargs['nablaJ']
            assert not self.self_condition, 'self condition not tested with guidance'
        else:
            nabla_J = lambda x: 0
        nablaJ_scheduler = kwargs['J_scheduler'] if ('J_scheduler' in kwargs and kwargs['J_scheduler'] is not None) else lambda t: 1.
        if 'proj_guidance' in kwargs and kwargs['proj_guidance'] is not None:
            may_proj_guidance = kwargs['proj_guidance']
        else:
            # no proj
            may_proj_guidance = lambda ep, nabla_J: ep + nabla_J
        return nabla_J, nablaJ_scheduler, may_proj_guidance

    def set_condition(self, img, u: torch.Tensor, shape, u0_or_uT):
        if u0_or_uT == 'uT':
            # NOTE: use the full trajectory to implement condition guidance.
            if len(shape) == 3 and not self.expand_condition:
                img[:, [0, 2], :self.nt] = u
            else:
                raise ValueError('Bad sample shape')
        elif u0_or_uT == 'u0':
            if len(shape) == 3 and not self.expand_condition:
                img[:, :3, 0] = u
            else:
                raise ValueError('Bad sample shape')
        else:
            assert False

    @torch.no_grad()
    def p_sample_loop(self, shape, w_groundtruth=None, enable_grad=True, **kwargs):
        assert not self.is_ddim_sampling, 'wrong branch!'

        nabla_J, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        x_start = None
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            if t != 0 or not enable_grad:
                # fill u0 into cur sample
                if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                    u0 = kwargs['u_init'] # should be (batch, Nx)
                    self.set_condition(img, u0, shape, 'u0')

                if self.is_condition_uT: # NOTE: uT here means physical time t=T
                    uT = kwargs['u_final'] # should be (batch, Nx)
                    self.set_condition(img, uT, shape, 'uT')

                if not self.train_on_padded_locations:
                    img[..., :3, self.nt :] = torch.zeros_like(img[..., :3, self.nt:])  # three states
                    img[..., 3:, self.nt-1:] = torch.zeros_like(img[..., 3:, self.nt-1:])  # actions
                
                # replace all w with w_groundtruth to complete condition on w_gt
                if w_groundtruth is not None:
                    img[:,1,:,:] = w_groundtruth

                
                self_cond = x_start if self.self_condition else None
                # calculates \hat{u_0} for better guidance calculation
                img_curr, x_start, pred_noise = self.p_sample(img, t, self_cond, **kwargs)

                # controlling diffusion:
                if self.guidance_u0: # 
                    img = img_curr
                else:
                    pred_noise = may_proj_guidance(pred_noise, nabla_J(img_curr) * nablaJ_scheduler(t)) # guidance
                    img, x_start, _ = self.p_sample(img, t, self_cond, pred_noise=pred_noise, **kwargs)
                
                img = img.detach()
                
            else:
                with torch.enable_grad():
                    # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                    if self.is_condition_u0: 
                        u0 = kwargs['u_init'] # should be (batch, Nx)
                        self.set_condition(img, u0, shape, 'u0')
                    if self.is_condition_uT: # NOTE: uT here means physical time t=T
                        uT = kwargs['u_final'] # should be (batch, Nx)
                        self.set_condition(img, uT, shape, 'uT')
                    if not self.train_on_padded_locations:
                        img[..., :3, self.nt:] = torch.zeros_like(img[..., :3, self.nt:])  # three states
                        img[..., 3:, self.nt-1:] = torch.zeros_like(img[..., 3:, self.nt-1:])  # actions
                    if w_groundtruth is not None:
                        img[:,1,:,:] = w_groundtruth
                    self_cond = x_start if self.self_condition else None
                    img_curr, x_start, pred_noise = self.p_sample(img, t, self_cond, **kwargs)
                    if self.guidance_u0:
                        img = img_curr
                    img = img.detach()
        img = self.unnormalize(img)
        return img
    
    @torch.no_grad()
    def ddim_sample(self, shape, 
                    return_all_timesteps = False,
                    w_groundtruth=None, 
                    enable_grad=False,
                    **kwargs):
        nabla_J, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
        batch, device, total_timesteps, sampling_timesteps, eta, objective = (
            shape[0], 
            self.betas.device, 
            self.num_timesteps, 
            self.sampling_timesteps, 
            self.ddim_sampling_eta, 
            self.objective
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
            u0 = kwargs['u_init'] # should be (batch, Nx)
            self.set_condition(img, u0, shape, 'u0')
        if self.is_condition_uT: # NOTE: uT here means physical time t=T
            uT = kwargs['u_final'] # should be (batch, Nx)
            self.set_condition(img, uT, shape, 'uT')
        if not self.train_on_padded_locations:
            img[..., :3, self.nt:] = torch.zeros_like(img[..., :3, self.nt:])  # three states
            img[..., 3:, self.nt-1:] = torch.zeros_like(img[..., 3:, self.nt-1:])  # actions

        if w_groundtruth is not None:
            img[:,3:,:] = w_groundtruth

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            if time_next >= 0 or not enable_grad:

                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(
                    img, time_cond, self_cond, 
                    clip_x_start=True, 
                    rederive_pred_noise=True, 
                    **kwargs
                )

                if time_next < 0:
                    img = x_start
                    imgs.append(img)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

                # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                if self.is_condition_u0: 
                    u0 = kwargs['u_init'] # should be (batch, Nx)
                    self.set_condition(img, u0, shape, 'u0')
                if self.is_condition_uT: # NOTE: uT here means physical time t=T
                    uT = kwargs['u_final'] # should be (batch, Nx)
                    self.set_condition(img, uT, shape, 'uT')
                if not self.train_on_padded_locations:
                    img[..., :3, self.nt:] = torch.zeros_like(img[..., :3, self.nt:])  # three states
                    img[..., 3:, self.nt-1:] = torch.zeros_like(img[..., 3:, self.nt-1:])  # actions
                if w_groundtruth is not None:
                    img[:,3:,:] = w_groundtruth

                imgs.append(img)
            else:
                with torch.enable_grad():
                    time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                    self_cond = x_start if self.self_condition else None
                    pred_noise, x_start, *_ = self.model_predictions(
                        img, 
                        time_cond, 
                        self_cond, 
                        clip_x_start=True, 
                        rederive_pred_noise=True, 
                        **kwargs
                    )
                    if time_next < 0:
                        img = x_start
                        imgs.append(img)
                        continue
                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[time_next]
                    sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    noise = torch.randn_like(img)
                    img = x_start * alpha_next.sqrt() + \
                        c * pred_noise + \
                        sigma * noise
                    # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                    if self.is_condition_u0: 
                        u0 = kwargs['u_init'] # should be (batch, Nx)
                        self.set_condition(img, u0, shape, 'u0')
                    if self.is_condition_uT: # NOTE: uT here means physical time t=T
                        uT = kwargs['u_final'] # should be (batch, Nx)
                        self.set_condition(img, uT, shape, 'uT')
                    if not self.train_on_padded_locations:
                        img[..., :3, self.nt:] = torch.zeros_like(img[..., :3, self.nt:])  # three states
                        img[..., 3:, self.nt-1:] = torch.zeros_like(img[..., 3:, self.nt-1:])  # actions
                    if w_groundtruth is not None:
                        img[:,3:,:] = w_groundtruth
                    imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad
    def sample(self, batch_size=16, clip_denoised=True, w_groundtruth=None, enable_grad=True, **kwargs):
        '''
        Kwargs:
            clip_denoised: 
                boolean, clip generated x
            nablaJ: 
                a gradient function returning nablaJ for diffusion guidance. 
                Can use the function get_nablaJ to construct the gradient function.
            J_scheduler: 
                Optional callable, scheduler for J, returns stepsize given t
            proj_guidance:
                Optional callable, postprocess guidance for better diffusion. 
                E.g., project nabla_J to the orthogonal direction of epsilon_theta
            guidance_u0:
                Optional, boolean. If true, use guidance inside the model_pred
            u_init:
                Optional, torch.Tensor of size (batch, Nx). u at time = 0, applies when self.is_condition_u0 == True
            w_groundtruth:
                Optional, torch.Tensor []. Groundtruth of w in calibration set.
                As condition during sampling p(u,c|w).
        '''
        if 'guidance_u0' in kwargs:
            self.guidance_u0 = kwargs['guidance_u0']
        if self.is_condition_u0:
            assert 'is_condition_u0' not in kwargs, 'specify this value in the model. not during sampling.'
            assert 'u_init' in kwargs and kwargs['u_init'] is not None
        if self.is_condition_uT:
            assert 'is_condition_uT' not in kwargs, 'specify this value in the model. not during sampling.'
            assert 'u_final' in kwargs and kwargs['u_final'] is not None
        # determine sampling size
        if self.temporal:
            sample_size = (batch_size, self.channels, *self.traj_size)
        else:
            seq_length, channels = self.seq_length, self.channels
            sample_size = (batch_size, channels, seq_length)

        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(sample_size, clip_denoised=clip_denoised, 
                         w_groundtruth=w_groundtruth, enable_grad=enable_grad,
                         **kwargs)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start, pred_noise = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, mean=True):
        # b, c, n = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        # 1. BEFORE MODEL_PREDICTION: SET INPUT
        # may fill u0 into cur sample
        # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
        if self.is_condition_u0: 
            self.set_condition(x, x_start[:, :3, 0], x.shape, 'u0')
                
        if self.is_condition_uT: # NOTE: uT here means physical time t=T
            self.set_condition(x, x_start[:, [0, 2], :self.nt], x.shape, 'uT')
        
        if not self.train_on_padded_locations:
            x[..., :3, self.nt:] = x_start[..., :3, self.nt:]  # three states
            x[..., 3:, self.nt-1:] = x_start[..., 3:, self.nt-1:]  # actions

        # 2. MODEL PREDICTION
        model_out = self.model(x, t, x_self_cond)

        # 3. AFTER MODEL_PREDICTION: SET OUTPUT AND TARGET
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # set loss to zero instead of learning zero output, not computing loss for the diffused state!
        if self.is_condition_u0 and self.is_condition_u0_zero_pred_noise:
            self.set_condition(noise, torch.zeros_like(x[:, :3, 0]), x.shape, 'u0')
        if self.is_condition_uT and self.is_condition_uT_zero_pred_noise:
            self.set_condition(noise, torch.zeros_like(x[:, [0, 2], :self.nt]), x.shape, 'uT')
        
        if not self.train_on_padded_locations:
            # Should not train on the zero-padded locations. (Target is still random noise)
            assert not self.expand_condition
            model_out[..., :3, self.nt:] = target[..., :3, self.nt:]  # three states
            model_out[..., 3:, self.nt-1:] = target[..., 3:, self.nt-1:]  # actions

        # 4. COMPUTE LOSS
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        
        if mean:
            return loss.mean()
        else:
            return loss

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        # diffusion timestep
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
