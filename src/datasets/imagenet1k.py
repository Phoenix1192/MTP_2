import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from PIL import Image
from logging import getLogger

logger = getLogger()

class ISICDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None):
        """
        ISIC Dataset Loader
        :param root_path: Path to the ISIC dataset directory.
        :param split: One of 'train', 'val', or 'test'.
        :param transform: Image transformations.
        """
        self.root_path = root_path
        self.split = split
        self.transform = transform
        
        split_map = {
            'train': 'ISIC-2017_Training_Data/ISIC-2017_Training_Data',
            'val': 'ISIC-2017_Validation_Data/ISIC-2017_Validation_Data',
            'test': 'ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data'
        }
        
        if split not in split_map:
            raise ValueError("Split must be 'train', 'val', or 'test'")
        
        self.image_dir = os.path.join(root_path, split_map[split])
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg'))])
        logger.info(f'Loaded {len(self.image_files)} images for {split} split.')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image


def make_isic_dataloader(
    root_path,
    transform,
    batch_size,
    split='train',
    collator=None,
    pin_mem=True,
    num_workers=4,
    world_size=1,
    rank=0,
    copy_data=False,
    drop_last=True
):
    """
    Creates an ISIC dataset and DataLoader.
    
    :param root_path: Path to the ISIC dataset directory.
    :param transform: Image transformations.
    :param batch_size: Number of images per batch.
    :param split: 'train', 'val', or 'test'.
    :param collator: Function to customize how a batch is created.
    :param pin_mem: Whether to use pinned memory for faster GPU transfers.
    :param num_workers: Number of parallel data loading workers.
    :param world_size: Total number of processes (for distributed training).
    :param rank: Process rank in distributed training.
    :param copy_data: Whether to copy dataset locally for faster access.
    :param drop_last: Whether to drop the last batch if it's smaller than batch_size.
    """
    dataset = ISICDataset(root_path=root_path, split=split, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=collator  # Include collator if provided
    )
    
    return dataset, dataloader, sampler

# Example usage
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    root_path = "/path/to/ISIC"  # Update this path
    dataset, data_loader, sampler = make_isic_dataloader(
        root_path=root_path,
        transform=transform,
        batch_size=32,
        split='train',
        num_workers=4
    )
    
    for images in data_loader:
        print(f"Batch shape: {images.shape}")
        break
