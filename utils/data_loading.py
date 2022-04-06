import logging
from os import listdir
from os.path import splitext
from pathlib import Path

from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.vision import VisionDataset
from torchvision import transforms

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

class BasicDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 mask_suffix: str = '',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super(BasicDataset, self).__init__(root, transform, target_transform)
        self.images_dir = Path(root, 'imgs')
        self.masks_dir = Path(root, 'masks')
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        img = Image.open(img_file[0]).convert('RGB')
        mask = Image.open(mask_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return img, mask


class CarvanaDataset(pl.LightningDataModule):
    def __init__(self, 
                 root_dir: str, 
                 batch_size: int = 1, 
                 val_percent: float = 0.1, 
                 num_workers: int = 0):
        super().__init__()
        # Directory to store Data
        self.root_dir = root_dir
          
        # Defining val percent of our data
        self.val_percent = val_percent

        # Defining transforms to be applied on the data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Defining DataLoade args
        self.loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    def prepare_data(self):
        # 1. Create dataset
        try:
            self.dataset = BasicDataset(root=self.root_dir,  mask_suffix='_mask', 
                                        transform=self.transform, target_transform=self.target_transform)
        except (AssertionError, RuntimeError):
            self.dataset = BasicDataset(root=self.root_dir, 
                                        transform=self.transform, target_transform=self.target_transform)

    def setup(self, stage: Optional[str] = None):
        # 2. Split into train / validation partitions
        if stage == 'fit' or stage is None:
            n_val = int(len(self.dataset) * self.val_percent)
            n_train = len(self.dataset) - n_val
            self.train_set, self.val_set = random_split(self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_args)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, drop_last=True, **self.loader_args)