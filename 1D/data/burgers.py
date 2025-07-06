import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from typing import Tuple, Callable, Optional, Union
from data.load_hdf5 import HDF5Dataset
import pdb

from configs.train_config import TrainConfig
from configs.eval_config import EvalConfig
from configs.posttrain_config import PostTrainConfig
from configs.inference_config import InferenceConfig

class BurgersDataset(Dataset):
    """Dataset class for 1D Burgers equation."""
    
    def __init__(
        self,
        dataset: str = "free_u_f_1e5",
        split: str = "train",
        root_path: str = None,
        nt_total: int = 11,
        nx: int = 128,
        is_normalize: bool = True,
        stack_u_and_f: bool = True,
        pad_for_2d_conv: bool = True,
        pad_size: int = 16,
        safety_transform: Optional[Callable] = None,
        is_need_idx: bool = False,
        is_subset: bool = False,
        config: Union[TrainConfig, EvalConfig, PostTrainConfig, InferenceConfig] = None,
    ):
        """Initialize Burgers dataset.
        
        Args:
            dataset: Dataset name
            split: Data split ('train', 'val', 'test')
            root_path: Path to data directory
            nt_total: Total number of time steps
            nx: Number of spatial points
            device: Device to store the data
            is_normalize: Whether to normalize the data
            stack_u_and_f: Whether to stack u and f into channels
            pad_for_2d_conv: Whether to pad time dimension for 2D conv
            data_folder: Name of the folder containing dataset files
            pad_size: Size to pad to for 2D convolution
            safety_transform: Callable function to compute safety scores from u values.
                            If None, uses default u²
            use_max_safety: If True, use maximum safety score for the entire sample
            is_need_idx: If True, return data and indices
        """
        self.root = root_path or "./datasets"
        self.split = split
        self.nt_total = nt_total
        self.nx = nx
        self.data_folder = dataset
        self.pad_size = pad_size
        self.use_max_safety = config.use_max_safety if config is not None else True
        
        if is_normalize:
            self.scaler = 10.0
        else:
            self.scaler = None
        self.stack_u_and_f = stack_u_and_f
        self.pad_for_2d_conv = pad_for_2d_conv
        self.safety_transform = safety_transform or (lambda u: u.pow(2))
        
        self.is_need_idx = is_need_idx
        self.is_subset = is_subset
        if isinstance(config, PostTrainConfig):
            self.finetune_subset_size = config.finetune_subset_size if config is not None else False
        else:
            self.finetune_subset_size = False

        self._init_dataset()
        
    def _init_dataset(self):
        """Initialize the underlying dataset."""
        path = os.path.join(
            self.root,
            self.data_folder,
            f'burgers_{self.split}.h5'
        )
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at {path}")
            
        self.dataset_cache = HDF5Dataset(
            path=path,
            mode=self.split
        )
        
        # If a subset is required, create an index list.
        if self.split == "train" and self.is_subset:
            self.indices = list(range(min(self.finetune_subset_size, len(self.dataset_cache))))
        else:
            self.indices = None

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.dataset_cache)
        
    def _process_data(self, data):
        """Process data according to configuration.
        
        Args:
            data: Raw data from HDF5 dataset
            
        Returns:
            Processed tensor ready for model input
        """
        u = data[0].clone().to(torch.float32)   # trajectory data
        f = data[1].clone().to(torch.float32)   # force data

        # safety score
        s = self.safety_transform(u)

        # If use_max_safety is True, replace all safety scores with the maximum value
        if self.use_max_safety:
            # Find max value for this sample across time and space dimensions
            max_s = s.amax(dim=(0, 1))  # Get the maximum value across time and space
            s = max_s.expand_as(s)  # A scalar tensor (no dimensions) can be broadcast to any shape.
            
        # Stack u, f, s
        if self.stack_u_and_f:
            if self.pad_for_2d_conv:
                nt = f.size(0)
                f = nn.functional.pad(f, (0, 0, 0, self.pad_size - nt), 'constant', 0)
                u = nn.functional.pad(u.squeeze(), (0, 0, 0, self.pad_size - 1 - nt), 'constant', 0)
                s = nn.functional.pad(s.squeeze(), (0, 0, 0, self.pad_size - 1 - nt), 'constant', 0)
                data = torch.stack((u, f, s), dim=0)
            else:
                data = torch.cat((u.squeeze(), f, s.squeeze()), dim=0)
        else:
            data = torch.cat((u.squeeze(), f, s.squeeze()), dim=0)
            
        # normalize data into [-1, 1], before input into diffusion model
        if self.scaler is not None:
            data = data / self.scaler
        
        return data
        
    def __getitem__(self, idx):
        """Get a single data item."""
        if self.indices is not None:
            idx = self.indices[idx]
        
        data = self.dataset_cache[idx]

        if self.is_need_idx:
            return self._process_data(data), idx
        else:
            return self._process_data(data) 
