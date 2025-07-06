import h5py
import torch
from typing import Tuple
from data.base import BaseDataset

class HDF5Dataset(BaseDataset):
    """Dataset class for loading HDF5 format PDE data."""

    def __init__(
        self,
        path: str,
        mode: str,
        nt: int = 11,
        nx: int = 128,
    ):
        """Initialize HDF5 dataset.
        
        Args:
            path: Path to HDF5 file
            mode: Dataset mode ('train', 'val', or 'test')
        """
        super().__init__(path)
        self.mode = mode
        
        # Open HDF5 file
        self.file = h5py.File(path, 'r')
        self.data = self.file[mode]
        
        # Get dataset names
        self.dataset_u = f'pde_{nt}-{nx}'  # state trajectory
        self.dataset_f = f'pde_{nt}-{nx}_f'  # control sequence
        
        # Load data into memory
        self.u_data = torch.from_numpy(self.data[self.dataset_u][:])
        self.f_data = torch.from_numpy(self.data[self.dataset_f][:])
            
        # Setup spatial coordinates
        nx = self.u_data.shape[-1]
        self.x = torch.linspace(0, 1, nx)[:, None]
        
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.u_data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single data item.
        
        Args:
            idx: Data index
            
        Returns:
            Tuple containing:
            - u: State trajectory
            - f: Control sequence
            - x: Spatial coordinates
        """
        return self.u_data[idx], self.f_data[idx], self.x