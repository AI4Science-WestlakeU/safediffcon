from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CustomDataset(Dataset):
    def __init__(self, is_normalize=True):
        dataset = load_from_disk(r"/data/tokamak_dataset")
        dataset.set_format('torch')
        self.dataset = dataset
        self.is_normalize = is_normalize
        self.scaler = torch.tensor([2, 7, 2, 1, 2, 2, 2, 2, 1, 1, 2, 3]).reshape(12,1)
        self.features = dataset.features

    def __len__(self):
        return len(self.dataset)

    def _process_data(self, data):
        states = data['outputs'][:, [1, 4, 6]].permute(1, 0)  # [3, 122]
        actions = data['actions'].permute(1, 0)  # [9, 121]
        actions = torch.nn.functional.pad(actions, (0, 1, 0, 0), 'constant', 0)  # [9, 122]
        data = torch.cat((states, actions), dim=0)  # [12, 122]

        if self.is_normalize:
            data = data / self.scaler
        
        return data

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return self._process_data(data)

class DatasetInspector:
    def __init__(self, dataset):
        self.dataset = dataset

    def inspect(self):
        print(f"length: {len(self.dataset)}")
        for feature in self.dataset.features:
            print(f"  - {feature}: {self.dataset.features[feature]}")
        
        sample_data = self.dataset[0] 
        states = sample_data[:3, :]
        actions = sample_data[3:, :]
        
        print(f"states shape: {states.shape}")
        print(f"actions shape: {actions.shape}")
        
        for i in range(states.shape[0]):
            print(f"states {i} dim range: {states[i, :].min().item()} to {states[i, :].max().item()}")
        for i in range(actions.shape[0]):
            print(f"actions {i} dim range: {actions[i, :].min().item()} to {actions[i, :].max().item()}")

dataset = CustomDataset(is_normalize=True)

inspector = DatasetInspector(dataset)
inspector.inspect()

# batch_size = 32
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for batch in dataloader:
#     states = batch['states']
#     actions = batch['actions']
    
#     print(f"Batch states size: {states.size()}, actions size: {actions.size()}")
