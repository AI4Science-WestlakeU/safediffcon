import torch
from torch.utils.data import Dataset
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

class Smoke(Dataset):
    def __init__(
        self,
        dataset_path,
        time_steps=32,
        steps=32,
        all_size=64,
        size=64,
        is_train=True,
        is_calibration=False,
    ):
        super().__init__()
        self.root = dataset_path
        self.steps = steps
        self.time_steps = time_steps
        self.time_interval = int(time_steps/steps)
        self.all_size = all_size
        self.size = size
        self.space_interval = int(all_size/size)
        self.is_train = is_train
        self.is_calibration = is_calibration
        self.dirname = "train" if self.is_train else "test"
        if self.is_train:
            if is_calibration:
                self.n_simu = 200
            else:
                self.n_simu = 19800
        else:
            self.n_simu = 50
        self.RESCALER = torch.tensor([2, 19, 20, 17, 20, 1, 1]).reshape(1, 7, 1, 1) 

    def __len__(self):
        return self.n_simu

    def __getitem__(self, sim_id):
        if self.is_train:
            if self.is_calibration:
                # print('Calibration')
                sim_id = sim_id + 20000 - self.n_simu
                d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                                dtype=torch.float).permute(2,3,0,1)
                v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                                dtype=torch.float).permute(2,3,0,1)
                c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                                dtype=torch.float).permute(2,3,0,1)
                s_ori = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                                dtype=torch.float) # 33, 8
                s_safe = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke_safe.npy'.format(sim_id))), \
                                dtype=torch.float) # 33, 9
                s = s_ori[:, 1]/s_ori.sum(-1) # 33
                s_safe = s_safe[:, 0]/s_safe.sum(-1) # 33
                s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) # 1, 33, 64, 64
                s_safe = s_safe.reshape(1, s_safe.shape[0], 1, 1).expand(1, s_safe.shape[0], self.size, self.size) # 1, 33, 64, 64
                state = torch.cat((d, v, c, s, s_safe), dim=0)[:, :32] # 7, 32, 64, 64
            
                data = (
                    state.permute(1, 0, 2, 3) / self.RESCALER, # 32, 7, 64, 64
                    sim_id,
                )
            else:
                # print('Training') 
                d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                                dtype=torch.float).permute(2,3,0,1)
                v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                                dtype=torch.float).permute(2,3,0,1)
                c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                                dtype=torch.float).permute(2,3,0,1)
                s_ori = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                                dtype=torch.float) # 33, 8
                s_safe = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke_safe.npy'.format(sim_id))), \
                                dtype=torch.float) # 33, 9
                s = s_ori[:, 1]/s_ori.sum(-1) # 33
                s_safe = s_safe[:, 0]/s_safe.sum(-1) # 33
                s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) # 1, 33, 64, 64
                s_safe = s_safe.reshape(1, s_safe.shape[0], 1, 1).expand(1, s_safe.shape[0], self.size, self.size) # 1, 33, 64, 64
                state = torch.cat((d, v, c, s, s_safe), dim=0)[:, :32] # 7, 32, 64, 64

                data = (
                    state.permute(1, 0, 2, 3) / self.RESCALER, # 32, 7, 64, 64
                    sim_id,
                )
        else:
            # print('Testing')
            sim_id = sim_id + 20000
            d = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                            dtype=torch.float).permute(2,3,0,1)
            v = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                            dtype=torch.float).permute(2,3,0,1)
            c = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                            dtype=torch.float).permute(2,3,0,1)
            s_ori = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke.npy'.format(sim_id))), \
                            dtype=torch.float)
            s_safe = torch.tensor(np.load(os.path.join(self.root, self.dirname, 'sim_{:06d}/Smoke_safe.npy'.format(sim_id))), \
                            dtype=torch.float)
            s = s_ori[:, 1]/s_ori.sum(-1)
            s_safe = s_safe[:, 0]/s_safe.sum(-1) # 33
            s = s.reshape(1, s.shape[0], 1, 1).expand(1, s.shape[0], self.size, self.size) 
            s_safe = s_safe.reshape(1, s_safe.shape[0], 1, 1).expand(1, s_safe.shape[0], self.size, self.size) # 1, 33, 64, 64
            state = torch.cat((d, v, c, s, s_safe), dim=0)[:, :32] # 7, 32, 64, 64
            data = (
                state.permute(1, 0, 2, 3), # 32, 7, 64, 64, not rescaled
                sim_id,
            )
        
        return data

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
if __name__ == "__main__":
    dataset = Smoke(
        dataset_path="/data/",
        is_train=False
    )
    print(len(dataset))
    data = dataset[4]
    print(data[0].shape, data[1], data[2], data[3])