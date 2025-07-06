import torch
from tqdm import tqdm
import datetime

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from accelerate import Accelerator
from ema_pytorch import EMA
from multiprocessing import cpu_count, set_start_method

from model.diffusion import GaussianDiffusion
from model.model_utils import has_int_squareroot, cycle, exists

from tensorboardX import SummaryWriter


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        dataset: Dataset,
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
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        # calculate_mmd = True
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        # mmd_dl = DataLoader(dataset, batch_size=num_samples, shuffle=True) 

        dl = self.accelerator.prepare(dl)
        # mmd_dl = self.accelerator.prepare(mmd_dl)
        self.dl = cycle(dl)
        # self.mmd_dl = cycle(mmd_dl)

        # optimizer & scheduler

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = CosineAnnealingLR(self.opt, T_max=10000, eta_min=0)  
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=20000, gamma=0.5)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        # make_dir(results_folder)
        # self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # self.calculate_mmd = calculate_mmd
        # if calculate_mmd:
        #     self.MMD_loss = MMD_loss()  
        

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
            'loss': self.total_loss,
            # 'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        if type(milestone) is int:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        else:
            data = torch.load(str(self.results_folder / milestone), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        print(self.device)
        writer = SummaryWriter(logdir='../tensorboard_runs/{}'.format(datetime.datetime.now().strftime("%m-%d_%H-%M-%S")))
        
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                self.total_loss = total_loss
                pbar.set_description(f'loss: {total_loss:.4f}')
                writer.add_scalar('loss', total_loss, self.step)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                        #     batches = num_to_groups(self.num_samples, self.batch_size)                      
                        #     if self.calculate_mmd:
                        #         all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                        # if self.calculate_mmd:
                        #     all_samples = torch.cat(all_samples_list, dim = 0)
                        #     real_data = next(self.mmd_dl).to(device)
                        #     mmd_loss = self.MMD_loss(all_samples, real_data) 
                            # pbar.set_description(f'mmd loss: {mmd_loss:.4f}')

                        # torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        writer.close()

class Trainer1D(Trainer):
    '''Rename Trainer1D to Trainer, but also allow the original references
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def make_dir(filename):
    """Make directory using filename if the directory does not exist"""
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise