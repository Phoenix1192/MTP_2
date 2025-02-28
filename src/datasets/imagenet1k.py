import os
import random
import time
import numpy as np
from logging import getLogger
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()

class IUXrayNoLabel(Dataset):
    def __init__(self, root, split='train', transform=None,
                 train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0):
        """
        IUXrayNoLabel Dataset

        :param root: Directory containing all IUXray images.
        :param split: One of 'train', 'val', or 'test'.
        :param transform: Transformations to apply to the images.
        :param train_ratio: Fraction of data for training.
        :param val_ratio: Fraction of data for validation.
        :param test_ratio: Fraction of data for testing.
        :param seed: Random seed for reproducibility.
        """
        self.root = root
        self.transform = transform
        self.split = split

        # List all image files (supporting common image formats)
        all_files = [os.path.join(root, f) for f in os.listdir(root)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_files.sort()  # Ensure a fixed order

        # Shuffle the list to ensure randomness in the splits
        random.seed(seed)
        random.shuffle(all_files)

        n = len(all_files)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        if split == 'train':
            self.files = all_files[:train_end]
        elif split == 'val':
            self.files = all_files[train_end:val_end]
        elif split == 'test':
            self.files = all_files[val_end:]
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        logger.info(f'IUXrayNoLabel: {split} split with {len(self.files)} images.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img


def make_iuxray(
    transform,
    batch_size,
    split='train',
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    copy_data=False,
    drop_last=True,
):
    """
    Creates the IUXray dataset and DataLoader for the given split.
    
    :param transform: Transformations to apply to each image.
    :param batch_size: Batch size.
    :param split: 'train', 'val', or 'test'.
    :param root_path: Directory where all IUXray images are stored.
    Other parameters follow the typical ImageNet-style function signature.
    """
    # If a mechanism for local copy is needed, integrate that logic here.
    dataset = IUXrayNoLabel(root=root_path, split=split, transform=transform)
    logger.info(f'IUXray dataset created for {split} split.')

    dist_sampler = DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )

    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('IUXray DataLoader created.')
    return dataset, data_loader, dist_sampler


# Example usage:
if __name__ == "__main__":
    # Define image transformations (modify as needed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Update this path to point to your folder containing all IUXray images.
    root_path = "/home/karthik_r/MTP/ijepa/images/images_normalized"

    # Create the training DataLoader
    dataset, data_loader, sampler = make_iuxray(
        transform=transform,
        batch_size=32,
        split='train',
        root_path=root_path,
        num_workers=4
    )

    # Iterate over one batch and print shape
    for images in data_loader:
        print(f"Batch shape: {images.shape}")
        break
