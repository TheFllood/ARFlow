import random
from typing import Any, Tuple
import numpy as np
import torch
from torchvision.datasets import DatasetFolder
from functools import lru_cache

class CustomCacheFolder(DatasetFolder):
    """
    Custom NPZFolder class to return n samples from the same randomly chosen class in each batch,
    optimized to reduce memory usage by caching frequently accessed samples.
    """
    
    def __init__(self, root: str, n: int):
        super().__init__(root, loader=self.npz_loader, extensions=(".npz",))
        self.n = n  # Number of samples per batch from the same class
        self.class_sample_cache = {}  # Cache for class-wise sample paths
    
    @staticmethod
    @lru_cache(maxsize=512)  # Cache up to 512 items to reduce disk I/O
    def npz_loader(path: str) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(path, mmap_mode='r')
        moments = data['moments']
        moments_flip = data['moments_flip'] if 'moments_flip' in data else moments
        return moments, moments_flip

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns n random samples from the same randomly chosen class for the given batch.
        """
        # Randomly choose a class, with probability weights for balanced sampling
        class_idx = random.choices(
            list(self.class_to_idx.values()), 
            k=1
        )[0]
        
        # Use or update cache of samples for the chosen class
        if class_idx not in self.class_sample_cache:
            self.class_sample_cache[class_idx] = [sample for sample in self.samples if sample[1] == class_idx]

        class_samples = self.class_sample_cache[class_idx]
        
        # Check if the class has enough samples
        if len(class_samples) < self.n:
            raise ValueError(f"Not enough samples in class '{class_idx}' to select {self.n} samples.")

        # Randomly select n samples from this class
        selected_samples = random.sample(class_samples, self.n)
        
        # Load, possibly flip, and convert samples to tensors
        moments_list = []
        for path, _ in selected_samples:
            moments, moments_flip = self.npz_loader(path)
            selected_moments = moments if torch.rand(1) < 0.5 else moments_flip
            moments_list.append(torch.tensor(selected_moments))
        
        # Stack moments into a single tensor and return with class index
        moments_tensor = torch.stack(moments_list)
        return moments_tensor, torch.tensor(class_idx)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)