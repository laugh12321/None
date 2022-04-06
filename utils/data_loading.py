from pathlib import Path
from typing import Optional

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Dataset(pl.LightningDataModule):
    def __init__(self, data_dir, scale=0.5, batch_size=1):
        super().__init__()

        self.val_dir = Path(data_dir, 'val2017')
        self.train_dir = Path(data_dir, 'train2017')
        self.val_annFile = Path(data_dir, 'annotations/instances_val2017.json')
        self.train_annFile = Path(data_dir, 'annotations/instances_train2017.json')

        h, w = 380, 676
        H, W = int(scale * h), int(scale * w)
        self.transform = transforms.Compose([transforms.Resize([H, W]),
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

        self.loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    
    # def prepare_data(self):
    #     # 1. Create dataset
    #     try:
    #         self.dataset = CarvanaDataset(self.images_dir, self.masks_dir, self.scale)
    #     except (AssertionError, RuntimeError):
    #         self.dataset = BasicDataset(self.images_dir, self.masks_dir, self.scale)
    
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.val_set = datasets.CocoDetection(self.val_dir, self.val_annFile, transform=self.transform) 
            self.train_set = datasets.CocoDetection(self.train_dir, self.train_annFile, transform=self.transform)
            
        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_args)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, drop_last=True, **self.loader_args)
