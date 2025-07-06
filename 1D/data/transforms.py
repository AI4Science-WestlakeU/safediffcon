import torch

class SafetyTransform:
    """Safety score transformer for PDE data."""
    
    def __init__(self, method: str = "square"):
        """
        Args:
            method: Transformation method ('square', 'abs', etc.)
        """
        self.method = method
        
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.method == "square":
            return u.pow(2)
        elif self.method == "abs":
            return torch.abs(u)
        else:
            raise ValueError(f"Unknown method: {self.method}") 