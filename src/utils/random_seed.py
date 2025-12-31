import torch
import numpy as np
import random
import os

def setup_reproducible_environment(seed: int = 42):
    """
    Setup reproducible environment for scientific experiments.
    
    Args:
        seed: Random seed for reproducibility
        
    Note:
        This function ensures deterministic behavior across runs
        for scientific reproducibility requirements.
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Ensure deterministic behavior
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # STRICT deterministic
    torch.use_deterministic_algorithms(True)


