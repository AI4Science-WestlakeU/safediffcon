import torch
from torch.autograd import grad
import sys, os
import datetime
import time
import numpy as np
import multiprocess as mp
from tqdm import tqdm
import argparse

from ddpm.data_2d import Smoke, cycle
from ddpm.diffusion_2d import GaussianDiffusion, Trainer
from dataset.apps.evaluate_solver import *
from video_diffusion_pytorch.video_diffusion_pytorch_conv3d import Unet3D_with_Conv3D


def load_ddpm_model(args, RESCALER):
    # load model
    model = Unet3D_with_Conv3D(
        dim = 64,
        dim_mults = (1, 2, 4),
        channels=7
    )
    print("number of parameters Unet3D_with_Conv3D: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(args.device)
    diffusion = GaussianDiffusion(
        model,
        image_size = 64,
        frames = 32,
        timesteps = 1000,            # number of steps
        sampling_timesteps=args.ddim_sampling_steps if args.using_ddim else 1000, # ddim, accelerate sampling
        ddim_sampling_eta=args.ddim_eta,
        loss_type = 'l2',            # L1 or L2
        standard_fixed_ratio = args.standard_fixed_ratio,
    )
    diffusion.eval()
    if args.finetune_set == 'train':
        # load trainer
        trainer = Trainer(
            diffusion,
            dataset = args.dataset,
            dataset_path = args.dataset_path,
            results_path = args.diffusion_model_path, 
            amp = False,                       # turn on mixed precision
        )
        trainer.load(args.diffusion_checkpoint) 
    else:
        cp = torch.load(os.path.join(args.diffusion_model_path, f'model-{args.diffusion_checkpoint}.pt'), map_location=args.device)
        diffusion.load_state_dict(cp['model'])
        diffusion.to(args.device)
        args.finetune_Q = cp['Q']
        args.finetune_alpha = cp['args'].alpha
        args.finetune_standard_fixed_ratio = cp['args'].standard_fixed_ratio
    return diffusion, args.device

def load_model(args, RESCALER):
    diffusion, device = load_ddpm_model(args, RESCALER)
    RESCALER = RESCALER.to(device)

    return diffusion
     

class InferencePipeline(object):
    def __init__(
        self,
        model,
        RESCALER=1,
        results_path=None,
        args_general=None,
    ):
        super().__init__()
        self.model = model
        self.results_path = results_path
        self.args_general = args_general
        self.device = self.args_general.device
        self.RESCALER = RESCALER
        self.Q = 0 # initialize quantile
        if model is not None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args_general.finetune_lr, betas = (0.9, 0.99))
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

    def get_weight(self, state, mode='train'):
        '''
        state: rescaled, [B, 32, 7, 64, 64]
        return: torch.Tensor, [B]
        '''
        if mode == 'train':
            return torch.exp( - self.args_general.standard_fixed_ratio * self.guidance(state))
        else:
            return torch.exp( - self.args_general.finetune_standard_fixed_ratio \
                    * self.guidance(state, Q=self.args_general.finetune_Q))
    
    def normalize_weights(self, weights): 
        '''
        weights: torch.Tensor, [B]
        return: torch.Tensor, [B]
        '''
        # if inf, replace with max
        if torch.isinf(weights).any():
            non_inf_mask = ~torch.isinf(weights)
            max_non_inf = weights[non_inf_mask].max()
            weights[torch.isinf(weights)] = max_non_inf

        if weights.sum() == 0:
            print("weights.sum() is 0 due to too large guidance coefficient")
            normalized_weights = torch.ones_like(weights)
        else:
            normalized_weights = weights.shape[0] * weights / weights.sum()
        normalized_weights[torch.isinf(normalized_weights)] = weights.shape[0] / torch.isinf(normalized_weights).sum()
        return normalized_weights

    def get_weighted_score_set(self, cal_dataloader):
        '''
        cal_dataloader: rescaled, calibration set
        return: conformal_score_set: torch.Tensor, [size of Calibration set], not rescaled
                normalized_weights: torch.Tensor, [size of Calibration set]
                ori_states: list, len = size of Calibration set, rescaled
        '''
        conformal_score_set = []
        ori_states = []
        weights = []
        print("Calculate conformal score...")
        for i in range(self.args_general.N_cal_batch):
            state, sim_id = next(cal_dataloader) # rescaled
            ori_states.append(state)
            state = state.to(self.device) # [B, 32, 7, 64, 64] 
            sampling_model = self.model
            output = sampling_model.sample(
                batch_size = state.shape[0],
                design_fn = None, # no guidance
                init = state[:,0,0],
                control = state[:,:,3:5]
                )
        
            # get weight due to distribution shift
            if self.args_general.finetune_set == 'train':
                weights.append(self.get_weight(state))
            else:
                weights.append(self.get_weight(state) * self.get_weight(state, mode='test'))

            output = output * self.RESCALER
            state = state * self.RESCALER
            conformal_score_set.append((output[:,-1,-1].mean((-1,-2)) - state[:,-1,-1,0,0]).abs()) # not rescaled

        weights = torch.cat(weights)
        normalized_weights = self.normalize_weights(weights) 
        return normalized_weights * torch.cat(conformal_score_set), normalized_weights, torch.cat(ori_states)

    def get_quantile(self, conformal_score_set, normalized_weights, ori_states, alpha): 
        '''
        conformal_score_set: torch.Tensor, [size of Calibration set], not rescaled
        normalized_weights: torch.Tensor, [size of Calibration set]
        ori_states: list, len = size of Calibration set, rescaled
        return: torch.Tensor, []
        '''
        n = conformal_score_set.shape[0]
        # get index of quantile
        sorted_tensor, sorted_indices = torch.sort(conformal_score_set)
        q = int(min(np.ceil((n + 1) * (1 - alpha)), n - 1))
        q_index1 = sorted_indices[q - 1]

        print(f"Calculate quantile, No.{q_index1}")
        Q = conformal_score_set[q_index1]
        return Q

    def conformal_prediction(self, cal_dataloader):
        with torch.no_grad():
            conformal_score_set, normalized_weights, ori_states = self.get_weighted_score_set(cal_dataloader)
        self.Q = self.get_quantile(conformal_score_set, normalized_weights, ori_states, self.args_general.alpha)
        # self.Q = torch.tensor(0., device=self.device)

    def guidance(self, x, Q=None):
        '''
        x: rescaled
        return: [B]
        '''
        if Q is None:
            Q = self.Q
        state = x * self.RESCALER # [B, 32, 7, 64, 64]
        guidance_success = state[:,:,5].mean((-1,-2,-3))
        guidance_energy = state[:,:,3:5].square().mean((1,2,3,4))
        guidance_safe = torch.maximum(state[:,-1,6].mean((-1,-2)) + self.Q - self.args_general.safe_bound, \
                                      torch.zeros_like(state[:,-1,6,0,0]))
        guidance = - (1 - self.args_general.w_safe) * guidance_success + self.args_general.w_safe * guidance_safe
        return guidance
    
    # define design function
    def design_fn(self, x):
        '''
        x: rescaled
        '''
        guidance = self.guidance(x).sum()
        grad_x = grad(guidance, x, grad_outputs=torch.ones_like(guidance))[0]
        return grad_x
        
    def run_model(self, state, backward_finetune=False):
        '''
        state: not rescaled
        '''
        print(f"Start sampling on the test set...")
        state = state.to(self.device) # [B, 32, 7, 64, 64]   
        # get w #
        if self.args_general.use_guidance:
            design_fn = self.design_fn
        else:
            design_fn = None
            
        preds = []
        for _ in range(self.args_general.N_test_batch):
            output = self.model.sample(
                batch_size = state.shape[0],
                design_fn = design_fn, 
                enable_grad = False,
                init = state[:,0,0]/self.RESCALER[:,0,0]
                ) # rescaled

            if backward_finetune:
                output = output.detach().cpu()
                # get c with condition 
                output = self.model.sample(
                    batch_size = state.shape[0],
                    design_fn = None, # no guidance
                    enable_grad = True, # weigheted finetune
                    init = state[:,0,0]/self.RESCALER[:,0,0],
                    control = output[:,:,3:5].to(self.device)
                    )
            
            output = output * self.RESCALER # not rescaled
        
            pred = torch.zeros_like(output)
            pred[:,:,:-2] = output[:,:,:-2]
            pred[:,:,-2] = output[:,:,-2].mean((-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,64,64)
            pred[:,:,-1] = output[:,:,-1].mean((-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,64,64)

            preds.append(pred)
        return preds[0], torch.cat(preds)

    def get_finetune_weight(self, train_dataset): 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 256, shuffle = False, pin_memory = True, num_workers = 16)
        weights = []
        for state, _ in tqdm(train_loader, desc="Getting finetune weights"):
            state = state.to(self.device)
            weights.append(self.get_weight(state))
        weights = torch.cat(weights)
        normalized_weights = self.normalize_weights(weights)
        return normalized_weights
        
    def finetune_model(self, train_state, weight): 
        '''
        train_state: rescaled  
        '''
        train_state, weight = train_state.to(self.device), weight.to(self.device)
        self.model.train()
        loss_fn = torch.nn.MSELoss()
        
        loss_diffusion = self.model(train_state, mean=False)
        loss = (weight * loss_diffusion).mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.eval()

        return loss

    def backward_finetune_model(self, preds): 
        '''
        preds: not rescaled  
        '''
        self.model.train()
        loss_fn = torch.nn.MSELoss()
        
        loss_success = preds[:,:,5].mean((-1,-2,-3))
        loss_safe = loss_fn(torch.maximum(preds[:,-1,-1].mean((-1,-2)) + self.Q - self.args_general.safe_bound, \
                            torch.zeros_like(preds[:,-1,-1].mean((-1,-2)))), torch.zeros_like(preds[:,-1,-1].mean((-1,-2))))
        loss = - (1 - self.args_general.w_safe) * loss_success.mean() + self.args_general.w_safe * loss_safe.mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.eval()

        return loss

    def run(self, dataloader, test_backward_loader, cal_dataloader, train_dataloader, train_dataset):
        start = time.time()
        save_file = f'results_{self.args_general.id}.txt'
        results_path = self.results_path
        model_save_path = os.path.join(results_path, self.args_general.id)
        os.makedirs(model_save_path, exist_ok=True)

        for epoch in range(self.args_general.epochs):
            self.epoch = epoch
            with open(os.path.join(results_path, save_file), 'a') as f:
                if epoch == 0:
                    f.write(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")+'\n')
                f.write(str(self.args_general)+'\n')

            # Finetune
            for i, data in enumerate(test_backward_loader):
                state, sim_id = data 
                if epoch > 0 or self.args_general.backward_finetune or self.args_general.finetune_set == 'train':
                    print('======================== FINETUNE ========================')
                    if self.args_general.backward_finetune:
                        pred, preds = self.run_model(state, backward_finetune=self.args_general.backward_finetune)
                    else:
                        if self.args_general.finetune_set == 'train':
                            finetune_dataset = train_dataset
                            finetune_dataloader = train_dataloader
                        else:
                            finetune_dataset = torch.utils.data.TensorDataset(preds / self.RESCALER.cpu(), torch.arange(preds.shape[0]))
                            finetune_dataloader = torch.utils.data.DataLoader(finetune_dataset, batch_size = self.args_general.finetune_batch_size, \
                                                                            shuffle = False, pin_memory = True, num_workers = 16)
                            finetune_dataloader = cycle(finetune_dataloader)
                        weight = self.get_finetune_weight(finetune_dataset)
                    for finetune_step in range(self.args_general.finetune_steps): 
                        if not self.args_general.backward_finetune:
                            finetune_state, sim_id = next(finetune_dataloader)
                            loss = self.finetune_model(finetune_state, weight[sim_id])
                        else:
                            loss = self.backward_finetune_model(preds)
                        if (finetune_step + 1) % 50 == 0:
                            print(f"Finetune step {finetune_step + 1}, Finetune loss: {loss.item()}")
                            with open(os.path.join(results_path, save_file), 'a') as f:
                                f.write(f"Finetune step {finetune_step + 1}, Finetune loss: {loss.item()}\n")
                    print('==========================================================\n')

                if not self.args_general.backward_finetune:
                    break

            # Conformal prediction
            self.conformal_prediction(cal_dataloader) 
            print(f"Q: {self.Q.item()}")

            # Sampling & Evaluation
            J_targets, safe_targets, J_safe_targets, J_safe_targets_pred, J_safe_targets_time, J_safe_targets_pred_time, mses, n_l2s = [], [], [], [], [], [], [], []
            for i, data in enumerate(dataloader):
                print(f"******************* Test Batch No.{i} *******************")
                state, sim_id = data 
                pred, preds = self.run_model(state)
                J_target, safe_target, J_safe_target, J_safe_target_pred, J_safe_target_time, J_safe_target_pred_time, mse, n_l2 = self.multi_evaluate(pred, state, batch_id=i, plot=False)
                J_targets.append(J_target)
                safe_targets.append(safe_target)
                J_safe_targets.append(J_safe_target)
                J_safe_targets_pred.append(J_safe_target_pred)
                J_safe_targets_time.append(J_safe_target_time)
                J_safe_targets_pred_time.append(J_safe_target_pred_time)
                mses.append(mse)
                n_l2s.append(n_l2)

            print(f"#################### Results of epoch {epoch} begin ####################")
            J_safe_targets_time = np.concatenate(J_safe_targets_time)
            J_safe_targets_pred_time = np.concatenate(J_safe_targets_pred_time)
            total_count_J_safe = np.concatenate(J_safe_targets).size
            total_count_J_safe_time = J_safe_targets_time.shape[0] * J_safe_targets_time.shape[1]
            non_zero_J_safe = np.count_nonzero(np.concatenate(J_safe_targets))
            unsafe_percentage = (non_zero_J_safe / total_count_J_safe) * 100
            non_zero_J_safe_pred = np.count_nonzero(np.concatenate(J_safe_targets_pred))
            unsafe_percentage_pred = (non_zero_J_safe_pred / total_count_J_safe) * 100
            non_zero_J_safe_time = np.count_nonzero(np.concatenate(J_safe_targets_time))
            unsafe_percentage_time = (non_zero_J_safe_time / total_count_J_safe_time) * 100
            non_zero_J_safe_pred_time = np.count_nonzero(np.concatenate(J_safe_targets_pred_time))
            unsafe_percentage_pred_time = (non_zero_J_safe_pred_time / total_count_J_safe_time) * 100
            print(f"J_target: {np.concatenate(J_targets).mean()},\nsafe_target: {np.concatenate(safe_targets).mean()},\nJ_safe_target: {np.concatenate(J_safe_targets).mean()},\
                  \nQ: {self.Q.item()},\nunsafe_percentage: {unsafe_percentage},\nunsafe_percentage_pred: {unsafe_percentage_pred},\nunsafe_percentage_time: {unsafe_percentage_time},\
                  \nunsafe_percentage_pred_time: {unsafe_percentage_pred_time},\nmse: {np.concatenate(mses).mean()},\nn_l2: {np.concatenate(n_l2s).mean()}")
            print(f"##################### Results of epoch {epoch} end #####################\n")

            # Save
            with open(os.path.join(results_path, save_file), 'a') as f:
                f.write(f"#################### Results of epoch {epoch} begin ####################\n")
                f.write("MEAN:\n")
                f.write(f"J_target_mean: {np.concatenate(J_targets).mean()},\nsafe_target_mean: {np.concatenate(safe_targets).mean()},\nJ_safe_target_mean: {np.concatenate(J_safe_targets).mean()},\
                    \nQ: {self.Q.item()},\nunsafe_percentage: {unsafe_percentage},\nunsafe_percentage_pred: {unsafe_percentage_pred},\nunsafe_percentage_time: {unsafe_percentage_time},\
                    \nunsafe_percentage_pred_time: {unsafe_percentage_pred_time},\nmse_mean: {np.concatenate(mses).mean()},\nn_l2_mean: {np.concatenate(n_l2s).mean()}\n")
                f.write(f"J_target: {np.concatenate(J_targets)},\nsafe_target: {np.concatenate(safe_targets)},\nJ_safe_target: {np.concatenate(J_safe_targets)},\nJ_safe_target_pred: {np.concatenate(J_safe_targets_pred)},\
                        \nmse: {np.concatenate(mses)},\nn_l2: {np.concatenate(n_l2s)}\n")
                f.write(f"##################### Results of epoch {epoch} end ######################\n")
                f.write("-----------------------------------------------------------------------------------------\n")
            torch.save({'model': self.model.state_dict(), 'args': self.args_general, 'Q': self.Q.item()}, \
                        os.path.join(model_save_path, f'model-{epoch}.pt'))

        end = time.time()
        print('END!')
        print('Model saved at:', model_save_path)
        print(f"Total time cost: {(end - start) / 60} minutes")

    def per_evaluate(self, sim, eval_no, pred, data, output_queue):
        '''
        eval_no: No of multi-process
        pred: torch.Tensor, [nt, 7, nx, nx]
        '''
        # print(f'Evaluate No.{eval_no}')
        init_velocity = init_velocity_() # 1, 128, 128, 2
        init_density = data[0,0,:,:] # nx, nx
        c1 = pred[:,3,:,:] # nt, nx, nx
        c2 = pred[:,4,:,:] # nt, nx, nx
        per_solver_out = solver(sim, init_velocity, init_density, c1, c2, per_timelength=256)
        # print(f'Evaluate No.{eval_no} down!')
        try:
            output_queue.put({eval_no:per_solver_out})
            # print(f"Queue Put down {eval_no}")
        except Exception as e:
            print(f"Error in process {eval_no}: {e}")

    def multi_evaluate(self, pred, data, batch_id=0, plot=False):
        '''
        pred: torch.Tensor, [B, 32, 7, 64, 64] 
        data: torch.Tensor, [B, 32, 7, 64, 64]
        '''
        pred[:, 0, 0] = data[:, 0, 0] # initial condition
        # control 
        print("------------ Start solving... ------------")
        start = time.time()
        pool_num = pred.shape[0]
        pred_ = pred.detach().cpu().numpy().copy()
        data_ = data.detach().cpu().numpy().copy()
        pred_[:,:,3:5,8:56,8:56] = 0 # indirect control
        solver_out = np.zeros_like(pred_, dtype=float)
        sim = init_sim_128()
        output_queue = mp.Queue()

        processes = []
        args_list = [(sim, i, pred_[i,:,:,:,:].copy(), data_[i,:,:,:,:].copy(),output_queue) for i in range(pool_num)]

        for args in args_list:
            process = mp.Process(target=self.per_evaluate, args=args)
            processes.append(process)
            process.start()

        # print(f"Total processes started: {len(processes)}")

        multi_results_list = []

        for i in range(len(processes)):
            multi_results_list.append(output_queue.get())
        
        multi_results_sorted = dict()
        for eval_no in range(len(processes)): # process no.
            for item in multi_results_list: 
                if list(item.keys())[0] == eval_no:
                    multi_results_sorted[f'{eval_no}']=list(item.values())[0]
                    continue

        for process in processes:
            process.join()

        # print("Process Join Down!")
        for i in range(len(multi_results_sorted)):
            solver_out[i,:,0,:,:] = multi_results_sorted[f'{i}'][0] # density
            solver_out[i,:,1,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,0] # vel_x
            solver_out[i,:,2,:,:] = multi_results_sorted[f'{i}'][2][:,:,:,1] # vel_x
            solver_out[i,:,3,:,:] = multi_results_sorted[f'{i}'][3] # control_x
            solver_out[i,:,4,:,:] = multi_results_sorted[f'{i}'][4] # control_y
            solver_out[i,:,5,:,:] = multi_results_sorted[f'{i}'][5] # smoke_portion
            solver_out[i,:,6,:,:] = multi_results_sorted[f'{i}'][6] # smoke_safe_portion

        if plot:
            print("Start Generating GIFs...")
            """
            Generate GIF
            """
            for i in range(solver_out.shape[0]):
                gif_density_safe(solver_out[i,:,0,:,:],zero=False,name=f'gif_density_{batch_id}_{i}')
        
        data = torch.tensor(solver_out, device=self.device)
        end = time.time()
        print(f"Time cost: {end-start}")

        # calculate metrics
        mask = torch.ones_like(pred, device = self.device)
        mask[:, 0] = False
        pred = pred * mask 
        data = data * mask

        mse = (torch.cat(((pred - data)[:,:,:3], (pred - data)[:,:,-2:]), dim=2).square().mean((1, 2, 3, 4))).detach().cpu().numpy()
        n_l2 = (((pred - data)[:,:,:3].square().sum((1, 2, 3, 4))).sqrt()/data[:,:,:3].square().sum((1, 2, 3, 4)).sqrt()).detach().cpu().numpy()
    
        J_target = - data[:, -1, 5, 0, 0].detach().cpu().numpy()
        safe_target = data[:, -1, 6, 0, 0].detach().cpu().numpy()
        J_safe_target = torch.maximum(data[:, -1, 6, 0, 0] - self.args_general.safe_bound, torch.zeros_like(data[:, -1, 6, 0, 0])).detach().cpu().numpy()
        non_zero_J_safe = np.count_nonzero(J_safe_target)
        total_count_J_safe = J_safe_target.size
        unsafe_percentage = (non_zero_J_safe / total_count_J_safe) * 100
        J_safe_target_pred = torch.maximum(pred[:, -1, 6, 0, 0] + self.Q - self.args_general.safe_bound, torch.zeros_like(data[:, -1, 6, 0, 0])).detach().cpu().numpy()
        non_zero_J_safe_pred = np.count_nonzero(J_safe_target_pred)
        unsafe_percentage_pred = (non_zero_J_safe_pred / total_count_J_safe) * 100

        J_safe_target_time = torch.maximum(data[:, :, 6, 0, 0] - self.args_general.safe_bound, torch.zeros_like(data[:, :, 6, 0, 0])).detach().cpu().numpy()
        non_zero_J_safe_time = np.count_nonzero(J_safe_target_time)
        total_count_J_safe_time = J_safe_target_time.shape[0] * J_safe_target_time.shape[1]
        unsafe_percentage_time = (non_zero_J_safe_time / total_count_J_safe_time) * 100
        J_safe_target_pred_time = torch.maximum(pred[:, :, 6, 0, 0] + self.Q - self.args_general.safe_bound, torch.zeros_like(data[:, :, 6, 0, 0])).detach().cpu().numpy()
        non_zero_J_safe_pred_time = np.count_nonzero(J_safe_target_pred_time)
        unsafe_percentage_pred_time = (non_zero_J_safe_pred_time / total_count_J_safe_time) * 100

        print('J_target=', J_target.mean())
        print('safe_target=', safe_target.mean())
        print('J_safe_target=', J_safe_target.mean())
        print('J_safe_target_pred=', J_safe_target_pred.mean())
        print(f"Unsafe percentage: {unsafe_percentage}%")
        print(f"Unsafe percentage pred: {unsafe_percentage_pred}%")
        print(f"Unsafe time percentage: {unsafe_percentage_time}%")
        print(f"Unsafe time percentage pred: {unsafe_percentage_pred_time}%")

        return J_target, safe_target, J_safe_target, J_safe_target_pred, J_safe_target_time, J_safe_target_pred_time, mse, n_l2


def inference(dataloader, test_backward_loader, cal_dataloader, train_dataloader, train_dataset, diffusion, args, RESCALER):
    model = diffusion # may vary according to different control methods

    inferencePPL = InferencePipeline(
        model, 
        RESCALER,
        results_path = args.inference_result_path,
        args_general=args
    )
    inferencePPL.run(dataloader, test_backward_loader, cal_dataloader, train_dataloader, train_dataset)

def load_data(args): 
    if args.dataset == "Smoke":
        dataset = Smoke(
            dataset_path=args.dataset_path,
            is_train=False,
        ) # not rescaled
        test_loader = torch.utils.data.DataLoader(dataset, batch_size = args.test_batch_size, shuffle = False, pin_memory = True, num_workers = 16)
        print("number of batch in test_loader: ", len(test_loader))

        test_backward_loader = torch.utils.data.DataLoader(dataset, batch_size = args.test_backward_batch_size, shuffle = False, pin_memory = True, num_workers = 16)

        dataset = Smoke(
            dataset_path=args.dataset_path,
            is_train=True,
            is_calibration=True
        ) # rescaled
        cal_loader = torch.utils.data.DataLoader(dataset, batch_size = args.cal_batch_size, shuffle = False, pin_memory = True, num_workers = 16)
        cal_loader = cycle(cal_loader)
        # print("number of batch in calibration_loader: ", len(cal_loader))

        train_dataset = Smoke(
            dataset_path=args.dataset_path,
            is_train=True,
        ) # rescaled
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.finetune_batch_size, \
                                                    shuffle = True, pin_memory = True, num_workers = 16)
        train_loader = cycle(train_loader)

        RESCALER = dataset.RESCALER.unsqueeze(0).to(args.device)
    else:
        assert False
    return test_loader, test_backward_loader, cal_loader, train_loader, train_dataset, RESCALER

def main(args):
    dataloader, test_backward_loader, cal_dataloader, train_dataloader, train_dataset, RESCALER = load_data(args)
    diffusion = load_model(args, RESCALER) # may vary according to different control methods
    inference(dataloader, test_backward_loader, cal_dataloader, train_dataloader, train_dataset, diffusion, args, RESCALER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference 2d inverse design model')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed')
    parser.add_argument('--id', type=str, default='finetune', metavar='S',
                        help='experiment id')
    parser.add_argument('--gpu', type=int, default=0, metavar='S',
                        help='gpu id')
    parser.add_argument('--dataset', default='Smoke', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--dataset_path', default="/data/", type=str,
                        help='path to dataset')
    parser.add_argument('--safe_bound', default=0.1, type=float,
                        help='bound of safety score')
    parser.add_argument('--inference_result_path', default="/results/test/", type=str,
                        help='path to save inference result')

    # DDPM
    parser.add_argument('--diffusion_model_path', default="/results/train/", type=str,
                        help='directory of trained diffusion model (Unet)')
    parser.add_argument('--diffusion_checkpoint', default=20, type=int,
                        help='index of checkpoint of trained diffusion model (Unet)')
    parser.add_argument('--using_ddim', default=True, type=eval,
                        help='If using DDIM')
    parser.add_argument('--ddim_eta', default=1, type=float, help='eta in DDIM')
    parser.add_argument('--ddim_sampling_steps', default=100, type=int, 
                        help='DDIM sampling steps. Should be smaller than 1000 (total timesteps)')
    parser.add_argument('--use_guidance', default=True, type=eval,
                        help='use guidance')
    parser.add_argument('--standard_fixed_ratio_list', nargs='+', default=[0], type=float,
                        help='standard_fixed_ratio for guidance')
    parser.add_argument('--w_safe_list', nargs='+', default=[0], type=float,
                        help='guidance intensity of safety score')

    # SafeDiffCon
    parser.add_argument('--finetune_set', default='train', type=str,
                        help='finetune set: train | test')
    parser.add_argument('--alpha', default=0.05, type=float,
                        help='alpha of quantile')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of epochs')
    parser.add_argument('--finetune_steps', default=10, type=int,
                        help='number of finetuning steps')
    parser.add_argument('--finetune_lr', default=1e-5, type=float,
                        help='learning rate of finetuning')
    parser.add_argument('--finetune_batch_size', default=1, type=int,
                        help='size of test batch of input to use')
    parser.add_argument('--backward_finetune', action='store_true',
                        help='backward finetune')
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='size of test batch of input to use')
    parser.add_argument('--cal_batch_size', default=1, type=int,
                        help='size of calibration batch of input to use')
    parser.add_argument('--test_backward_batch_size', default=1, type=int,
                        help='size of test backward batch of input to use')
    parser.add_argument('--N_cal_batch', default=1, type=int,
                        help='number of calibration batches per score set')
    parser.add_argument('--N_test_batch', default=1, type=int,
                        help='number of test batches per epoch')


    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')

    if args.finetune_set == 'test' and args.use_guidance == False:
        args.standard_fixed_ratio_list = [0]
    standard_fixed_ratio_list = args.standard_fixed_ratio_list
    w_safe_list = args.w_safe_list

    for w_safe in w_safe_list:
        for standard_fixed_ratio in standard_fixed_ratio_list:
            args.standard_fixed_ratio = standard_fixed_ratio
            args.w_safe = w_safe
            print("args: ", args)
            main(args)