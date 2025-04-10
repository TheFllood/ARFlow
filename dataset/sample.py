import random
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import torch

class CustomImageFolder(ImageFolder):
    """
    Custom ImageFolder class to return n images from the same randomly chosen class in each batch,
    optimized to reduce memory usage by avoiding pre-loading all images into memory.
    """

    def __init__(
        self,
        root: str,
        n: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        self.n = n  # Number of images per batch from the same class

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns n random images from the same randomly chosen class for the given batch,
        without loading all class images into memory at once.

        Args:
            index (int): The index for the data loader.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of a tensor of n images and a tensor of the class index.
        """
        # Randomly choose a class
        class_idx = random.choice(list(self.class_to_idx.values()))  # Convert dict_values to a list

        # Find all images belonging to the chosen class (dynamically)
        class_images = [img for img in self.samples if img[1] == class_idx]

        # Ensure there are enough images to sample
        if len(class_images) < self.n:
            raise ValueError(f"Not enough images in class '{class_idx}' to select {self.n} images.")

        # Randomly sample n images from the chosen class
        selected_images = random.sample(class_images, self.n)

        # Load and transform the images
        images = [self.loader(img_path) for img_path, _ in selected_images]
        if self.transform is not None:
            images = [self.transform(img) for img in images]

        # Stack the images into a single tensor
        images = torch.stack(images)  # Stack the images into a tensor of shape [n, 3, H, W]

        # Return the images and the class index as a tensor
        return images, torch.tensor(class_idx)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)