import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class TokamakDataset(Dataset):
    def __init__(self, split='train', is_normalize=True, is_need_idx=False):
        dataset = load_from_disk(r"/data/tokamak_dataset")
        dataset.set_format('torch')
        
        # Split dataset based on mode
        if split == 'train':
            self.dataset = dataset.select(range(48950))
        elif split == 'cal':
            self.dataset = dataset.select(range(48950, 49950))
        elif split == 'test':
            self.dataset = dataset.select(range(49950, 50000))
        else:
            raise ValueError("split must be one of ['train', 'cal', 'test']")
            
        self.features = dataset.features
        self.pad_size = 128
        self.nt_total = 122

        # CHOICE: use single scaler or multiple scaler
        # self.scaler = 10.0
        self.scaler = torch.tensor([2, 7, 2, 1, 2, 2, 2, 2, 1, 1, 2, 3]).reshape(12,1)

        self.is_normalize = is_normalize
        self.is_need_idx = is_need_idx

    def __len__(self):
        return len(self.dataset)

    def _process_data(self, data):
        """Process data to fit model input"""
        states = data['outputs'][:, [1, 4, 6]].permute(1, 0)  # Convert to [3, 122]
        actions = data['actions'].permute(1, 0)  # Convert to [9, 121]

        nt = states.shape[1]
        states = torch.nn.functional.pad(states, (0, self.pad_size - nt, 0, 0), 'constant', 0)
        actions = torch.nn.functional.pad(actions, (0, self.pad_size - nt + 1, 0, 0), 'constant', 0)  # padding to [9, 128]
        data = torch.cat((states, actions), dim=0)  # Stack to [12, 128]

        if self.is_normalize:
            data = data / self.scaler
        
        return data

    def __getitem__(self, idx):
        """Get a single data item"""
        data = self.dataset[idx]

        if self.is_need_idx:
            return self._process_data(data), idx
        else:
            return self._process_data(data)

if __name__ == "__main__":
    from IPython import embed
    from tqdm import tqdm
    device = 'cuda:0'
    safe_bound = 4.98
    dataset = TokamakDataset(split='train')
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True, num_workers=10)
    
    q95_mean = 0
    below_samples = 0
    below_points = 0
    total_samples = 0
    safe_samples = 0
    for i, data in tqdm(enumerate(dataloader)):
        data = data.to(device)
        data = data * dataset.scaler.unsqueeze(0).to(device)
        q95 = data[:, 1, :dataset.nt_total]  # Get q95 values
        
        # Calculate mean
        q95_mean += q95.mean().item() * len(data)
        total_samples += len(data)
        
        # Calculate below samples and points
        below_mask = q95 < safe_bound
        below_samples += (below_mask.any(dim=-1)).sum().item()
        below_points += below_mask.sum().item()
        
        # Calculate samples that are always above safe_bound
        safe_samples += ((q95 >= safe_bound).all(dim=-1)).sum().item()

    q95_mean = q95_mean / total_samples
    print('train:')
    print(f"Q95 mean: {q95_mean:.4f}")
    print(f"Percentage of samples below {safe_bound}: {below_samples / total_samples:.4f}")
    print(f"Percentage of time points below {safe_bound}: {below_points / (total_samples * dataset.nt_total):.4f}")
    print(f"Percentage of samples always above {safe_bound}: {safe_samples / total_samples:.4f}")

    dataset = TokamakDataset(split='test')
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    q95_mean = 0
    below_samples = 0
    below_init_points = 0
    below_points = 0
    total_samples = 0
    for i, data in tqdm(enumerate(dataloader)):
        data = data.to(device)
        data = data * dataset.scaler.unsqueeze(0).to(device)
        q95 = data[:, 1, :dataset.nt_total]  # Get q95 values
        q95_mean += q95.mean().item() * len(data)
        total_samples += len(data)
        below_mask = q95 < safe_bound
        below_samples += (below_mask.any(dim=-1)).sum().item()
        below_init_points += below_mask[:, 0].sum().item()
        below_points += below_mask.sum().item()
    q95_mean = q95_mean / total_samples
    print('test:')
    print(f"Q95 mean: {q95_mean:.4f}")
    print(f"Percentage of samples below {safe_bound}: {below_samples / total_samples:.4f}")
    print(f"Percentage of time points below {safe_bound}: {below_points / (total_samples * dataset.nt_total):.4f}")
    print(f"Percentage of initial time points below {safe_bound}: {below_init_points / (total_samples):.4f}")