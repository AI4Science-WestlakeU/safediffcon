import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
import pdb

from functools import partial
from collections import namedtuple
from torch.cuda.amp import autocast
from einops import reduce

from model.pinn_loss import residual_gradient
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
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 0.,
        auto_normalize = False, # should be false
        guidance_u0 = True,
        conditioned_on_residual = None, 
        residual_on_u0 = False, 
        # conv choice
        temporal = False, # Must be True when using 2d conv
        use_conv2d = False, # enabled when temporal == True
        # condition choice
        is_condition_u0 = False, 
        is_condition_uT = False, 
        is_condition_u0_zero_pred_noise = True, 
        is_condition_uT_zero_pred_noise = True, 
        condition_idx=10, # when using condtion on uT, this is the index of the conditioned time stamp
        # unnecessary
        recurrence = False, 
        recurrence_k = 1, 
        normalize_beta=False, 
        train_on_padded_locations=False, # NOTE: true: mimic faulty behavior. in principle it should be false.
        train_on_partially_observed = None, 
        set_unobserved_to_zero_during_sampling = False,   
        is_model_w=False, 
        eval_two_models=False, 
        expand_condition=False, 
        prior_beta=1,     
    ):
        '''
        Arguments:
            temporal: if conv along the time dimension
            use_conv2d: if using space+time 2d conv

        '''

        # NOTE: perhaps we cannot always normalize the dataset? 
        # May need to fix this problem.
        super().__init__()

        # if eval_two_models: self.model will be a tuple (model_uw, model_w)
        if not eval_two_models:
            self.model = model
            self.channels = self.model.channels
            self.self_condition = self.model.self_condition
        else:
            self.model_uw = model[0]
            self.model_w = model[1]
            # these shapes will be used to set e.g. initial noisy img. set to the same as model_uw since model_uw should be kept the same as model_w
            self.channels = self.model_uw.channels
            self.self_condition = self.model_uw.self_condition
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

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

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
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

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
        self.train_on_partially_observed = train_on_partially_observed
        self.set_unobserved_to_zero_during_sampling = set_unobserved_to_zero_during_sampling
        self.conditioned_on_residual = conditioned_on_residual
        self.residual_on_u0 = residual_on_u0
        self.recurrence = recurrence
        self.recurrence_k = recurrence_k
        self.is_model_w = is_model_w
        self.eval_two_models = eval_two_models
        self.expand_condition = expand_condition
        self.prior_beta = prior_beta
        self.train_on_padded_locations = train_on_padded_locations
        self.normalize_beta = normalize_beta
        self.condition_idx = condition_idx

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

    def model_predictions(self, x, t, x_self_cond = None, residual=None, clip_x_start = False, rederive_pred_noise = False, **kwargs):
        if self.eval_two_models:
            model_output = self.model_uw(x, t, x_self_cond, residual=residual) # p(u, w)
            x_w = x.clone()
            x_w[..., 0, 1: self.condition_idx, :] = 0 # u[1...T-1] should be zero (consistent with training)
            model_w_output = self.model_w(x_w, t, x_self_cond, residual=residual) # p(w)
            model_w_output[..., 0, :, :] = 0 # only output w, not trained on u

            # step size scheduling. This implementation is shitty (creating a tensor every iteration)...
            eta = kwargs['w_scheduler'](t[0].item()) if ('w_scheduler' in kwargs and kwargs['w_scheduler'] is not None) else 1
            if self.normalize_beta:
                model_output = (model_output - (1 - self.prior_beta) * model_w_output) / self.prior_beta
            else:
                model_output = model_output - (1 - self.prior_beta) * eta * model_w_output
        elif self.is_model_w:
            assert not self.eval_two_models
            x[..., 0, 1: self.condition_idx, :] = 0 # unseen during training (trained on p(w|u0, uT))
            model_output = self.prior_beta * self.model(x, t, x_self_cond, residual=residual)
            model_output[..., 0, :, :] = 0
        else:
            model_output = self.model(x, t, x_self_cond, residual=residual)
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
                    # print(pred_noise.size())
                    # print((nablaJ(x_start) * nablaJ_scheduler(t[0].item())).size())
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

    def p_mean_variance(self, x, t, x_self_cond = None, residual=None, **kwargs):
        preds = self.model_predictions(x, t, x_self_cond, residual=residual, **kwargs)
        x_start = preds.pred_x_start

        # NOTE: seems that if no clamp, result would be problematic
        if kwargs['clip_denoised']:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, preds.pred_noise

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond=None, residual=None, **kwargs):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, residual=residual, **kwargs)
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
            # NOTE: well I am not sure what sign nabla_J should take....
            may_proj_guidance = lambda ep, nabla_J: ep + nabla_J
        return nabla_J, nablaJ_scheduler, may_proj_guidance

    def set_condition(self, img, u: torch.Tensor, shape, u0_or_uT):
        if u0_or_uT == 'uT':
            if len(shape) == 4:
                if self.expand_condition:
                    img[:, 3, :, :] = u.unsqueeze(-2)
                else:
                    img[:, 0, self.condition_idx, :] = u
            elif len(shape) == 3 and not self.expand_condition:
                img[:, self.condition_idx, :] = u
            else:
                raise ValueError('Bad sample shape')
        elif u0_or_uT == 'u0':
            if len(shape) == 4:
                if self.expand_condition:
                    img[:, 2, :, :] = u.unsqueeze(-2)
                else:
                    img[:, 0, 0, :] = u
            elif len(shape) == 3 and not self.expand_condition:
                img[:, 0, :] = u
            else:
                raise ValueError('Bad sample shape')
        else:
            assert False

    def set_pad_condition(self, img, origin_img=None):
        assert not self.expand_condition
        if origin_img is None:
            origin_img = torch.zeros_like(img)
        img[..., 0, self.condition_idx + 1:, :] = origin_img[..., 0, self.condition_idx + 1:, :]
        img[..., 1, self.condition_idx:, :] = origin_img[..., 1, self.condition_idx:, :]
        img[..., 2, self.condition_idx:, :] = origin_img[..., 2, self.condition_idx:, :]

    @torch.no_grad()
    def p_sample_loop(self, shape, w_groundtruth=None, enable_grad=True, **kwargs):
        assert not self.is_ddim_sampling, 'wrong branch!'

        nabla_J, nablaJ_scheduler, may_proj_guidance = self.get_guidance_options(**kwargs)
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        x_start = None
        # pdb.set_trace()
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # for k in range(self.recurrence_k):
            if t != 0 or not enable_grad:
                # fill u0 into cur sample
                if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                    u0 = kwargs['u_init'] # should be (batch, Nx)
                    self.set_condition(img, u0, shape, 'u0')

                if self.is_condition_uT: # NOTE: uT here means physical time t=T
                    uT = kwargs['u_final'] # should be (batch, Nx)
                    self.set_condition(img, uT, shape, 'uT')
                
                # replace all w with w_groundtruth to complete condition on w_gt
                if w_groundtruth is not None:
                    img[:,1,:,:] = w_groundtruth
                if not self.train_on_padded_locations:
                    self.set_pad_condition(img)

                if self.set_unobserved_to_zero_during_sampling:
                    Nx = img.size(-1)
                    if len(shape) == 4:
                        img[:, 0, :, Nx // 4: (Nx * 3) // 4] = 0
                    else:
                        raise ValueError('Bad sample shape')
                
                # condition on residual
                if self.conditioned_on_residual is not None:
                    assert not self.eval_two_models
                    assert len(img.shape) == 4, 'must stack u, f and residual/residual gradient'
                    if self.conditioned_on_residual == 'residual_gradient':
                        residual = residual_gradient(img[..., :2, :, :] if not self.residual_on_u0 else x_start[..., :2, :, :])
                    elif self.conditioned_on_residual == 'residual':
                        raise NotImplementedError
                else:
                    residual = None
                
                self_cond = x_start if self.self_condition else None
                # calculates \hat{u_0} for better guidance calculation
                img_curr, x_start, pred_noise = self.p_sample(img, t, self_cond, residual=residual, **kwargs)

                # controlling diffusion:
                if self.guidance_u0: # 
                    img = img_curr
                else:
                    pred_noise = may_proj_guidance(pred_noise, nabla_J(img_curr) * nablaJ_scheduler(t)) # guidance
                    img, x_start, _ = self.p_sample(img, t, self_cond, pred_noise=pred_noise, residual=residual, **kwargs)
                
                # pdb.set_trace()
                img = img.detach()
                
                if self.recurrence:
                    # self recurrence: add back the noise
                    img = self.recurrent_sample(img, t)
            else:
                with torch.enable_grad():
                    if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                        u0 = kwargs['u_init'] # should be (batch, Nx)
                        self.set_condition(img, u0, shape, 'u0')
                    if self.is_condition_uT: # NOTE: uT here means physical time t=T
                        uT = kwargs['u_final'] # should be (batch, Nx)
                        self.set_condition(img, uT, shape, 'uT')
                    if w_groundtruth is not None:
                        img[:,1,:,:] = w_groundtruth
                    if not self.train_on_padded_locations:
                        self.set_pad_condition(img)
                    self_cond = x_start if self.self_condition else None
                    img_curr, x_start, pred_noise = self.p_sample(img, t, self_cond, residual=residual, **kwargs)
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
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
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
        if w_groundtruth is not None:
            img[:,1,:,:] = w_groundtruth
        if not self.train_on_padded_locations:
            self.set_pad_condition(img)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            if time_next >= 0 or not enable_grad:
                if self.set_unobserved_to_zero_during_sampling:
                    Nx = img.size(-1)
                    if len(shape) == 4:
                        img[:, 0, :, Nx // 4: (Nx * 3) // 4] = 0
                    else:
                        raise ValueError('Bad sample shape')
                assert self.conditioned_on_residual is None
                assert self.eval_two_models == False

                time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, **kwargs)

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

                if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                    u0 = kwargs['u_init'] # should be (batch, Nx)
                    self.set_condition(img, u0, shape, 'u0')
                if self.is_condition_uT: # NOTE: uT here means physical time t=T
                    uT = kwargs['u_final'] # should be (batch, Nx)
                    self.set_condition(img, uT, shape, 'uT')
                if w_groundtruth is not None:
                    img[:,1,:,:] = w_groundtruth
                if not self.train_on_padded_locations:
                    self.set_pad_condition(img)

                imgs.append(img)
            else:
                with torch.enable_grad():
                    time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
                    self_cond = x_start if self.self_condition else None
                    pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True, **kwargs)
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
                    if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
                        u0 = kwargs['u_init'] # should be (batch, Nx)
                        self.set_condition(img, u0, shape, 'u0')
                    if self.is_condition_uT: # NOTE: uT here means physical time t=T
                        uT = kwargs['u_final'] # should be (batch, Nx)
                        self.set_condition(img, uT, shape, 'uT')
                    if w_groundtruth is not None:
                        img[:,1,:,:] = w_groundtruth
                    if not self.train_on_padded_locations:
                        self.set_pad_condition(img)
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
            if self.conditioned_on_residual is not None:
                assert not self.two_ddpm
                # when using conditioned_on_residual, channel = 4, but only samples 2
                if self.conditioned_on_residual == 'residual_gradient':
                    sample_size = (batch_size, self.channels - 2, *self.traj_size)
                else:
                    raise NotImplementedError
            else:
                sample_size = (batch_size, self.channels, *self.traj_size)
        else:
            assert self.conditioned_on_residual is None
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
        if self.is_condition_u0: # NOTE: u0 here means physical time t=0, while the u0 in guidance means the 0th step in diffusion
            self.set_condition(x, x_start[:, 0, 0, :], x.shape, 'u0')
        if self.is_condition_uT: # NOTE: uT here means physical time t=T
            self.set_condition(x, x_start[:, 0, self.condition_idx, :], x.shape, 'uT')
        if not self.train_on_padded_locations:
            self.set_pad_condition(x)

        # condition on residual
        if self.conditioned_on_residual is not None:
            assert len(x.shape) == 4, 'must stack u, f and residual/residual gradient'
            if self.conditioned_on_residual == 'residual_gradient':
                residual = residual_gradient(x[..., :2, :, :] if not self.residual_on_u0 else x_start[..., :2, :, :])
            elif self.conditioned_on_residual == 'residual':
                raise NotImplementedError
        else:
            residual = None

        # training p(w|u0, uT)
        if self.is_model_w:
            x[..., 0, 1:self.condition_idx, :] = 0 # when training p(w | u0, uT), unet does not see u_[1...T-1]

        # 2. MODEL PREDICTION
        model_out = self.model(x, t, x_self_cond, residual=residual)


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

        # set target: does not train on unobserved, conditioned or teach model to output zero
        if self.train_on_partially_observed is not None:
            if self.train_on_partially_observed == 'front_rear_quarter':
                Nx = model_out.size(-1)
                model_out[..., 0, :, Nx // 4: (Nx * 3) // 4] = target[..., 0, :, Nx // 4: (Nx * 3) // 4]
            elif self.train_on_partially_observed == 'front_rear_quarter_u_and_f':
                # mimic faulty behavior in some versions
                Nx = model_out.size(-1)
                model_out[..., Nx // 4: (Nx * 3) // 4] = target[..., Nx // 4: (Nx * 3) // 4]
            else:
                raise NotImplementedError
            
        # TODO: set loss to zero instead of learning zero output
        if self.is_condition_u0 and self.is_condition_u0_zero_pred_noise:
            # not computing loss for the diffused state!
            self.set_condition(noise, torch.zeros_like(x[:, 0, 0, :]), x.shape, 'u0')
        if self.is_condition_uT and self.is_condition_uT_zero_pred_noise:
            # not computing loss for the diffused state!
            self.set_condition(noise, torch.zeros_like(x[:, 0, 0, :]), x.shape, 'uT')
        
        if self.is_model_w:
            # do not train on pred noise u of model_w
            model_out[..., 0, :, :] = target[..., 0, :, :]
        
        if not self.train_on_padded_locations:
            self.set_pad_condition(model_out, origin_img=target)

        # 4. COMPUTE LOSS
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)

        # add this if for finetune loss
        if mean:
            return loss.mean()
        else:
            return loss

    def forward(self, img, *args, **kwargs):
        if self.temporal:
            b, c, nt, nx, device, traj_size = *img.shape, img.device, self.traj_size
            assert (nt, nx) == traj_size, f'traj size must be (nt, nx) of ({nt, nx}), but got {traj_size}'
        else:
            b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
            assert n == seq_length, f'seq length must be {seq_length}'
        # diffusion timestep
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# class GaussianDiffusion1D(GaussianDiffusion):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)