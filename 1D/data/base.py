from torch.utils.data import Dataset
from typing import Optional
import os

class BaseDataset(Dataset):
    """Base dataset class for PDE data loading."""
    
    def __init__(self, root_path: Optional[str] = None):
        """
        Args:
            root_path: Root directory for the dataset
        """
        self.root = root_path or "./datasets/"
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, idx):
        raise NotImplementedError        
    def _validate_paths(self):
        """Validate all required paths exist"""
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Root path {self.root} does not exist") 
