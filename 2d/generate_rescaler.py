from tqdm import tqdm
import torch
import numpy as np  
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root = '/data/'
dirname = 'train/' 
n_sim = 20000 
num_t = 32
num_x = 64
sim_range = range(n_sim)
max_coef = {}
for i in range(5):
    max_coef[i] = 0
for sim_id in tqdm(sim_range):
    d = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Density.npy'.format(sim_id))), \
                        device=device, dtype=torch.float).permute(2,3,0,1)
    v = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Velocity.npy'.format(sim_id))), \
                        device=device, dtype=torch.float).permute(2,3,0,1)
    c = torch.tensor(np.load(os.path.join(root, dirname, 'sim_{:06d}/Control.npy'.format(sim_id))), \
                    device=device, dtype=torch.float).permute(2,3,0,1)            
    X = torch.cat((d, v, c), dim=0) 
    for i in range(5):
        max_coef[i] = max(int(max_coef[i]), int(X[i].abs().max())+1)
        max_coef[i] = max(int(max_coef[i]), int(X[i].abs().max())+1)
print('Max', list(max_coef.values()))   