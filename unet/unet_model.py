""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from utils.dice_score import dice_loss
# from utils.data_loading import BasicDataset, CarvanaDataset

from pathlib import Path
# from argparse import ArgumentParser

# 🍦 Vanilla PyTorch
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import RMSprop, lr_scheduler

# ⚡ PyTorch Lightning
import pytorch_lightning as pl

class UNet(pl.LightningModule):
    
    def __init__(self, 
           n_channels: int, 
           n_classes: int, 
           bilinear: bool = False, 
           lr: float = 1e-5):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # log hyperparameters
        self.save_hyperparameters(logger=False)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def common_step(self, batch, batch_idx):
        images, true_masks = batch['image'], batch['mask']
        # print(type(images))
        # images = images.to(device=self.device, dtype=torch.float32)
        # true_masks = true_masks.to(device=self.device, dtype=torch.long)
        masks_pred = self(images)
        loss = F.cross_entropy(masks_pred, true_masks) + dice_loss(F.softmax(masks_pred, dim=1).float(),
                F.one_hot(true_masks, self.hparams["n_classes"]).permute(0, 3, 1, 2).float(), multiclass=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("training_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx) 
        return loss

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        optimizer = RMSprop(self.parameters(), lr=self.hparams["lr"], weight_decay=1e-8, momentum=0.9)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2),
            #     "monitor": "validation_loss",
            #     "frequency": "indicates how often the metric is updated"
            # },
        }

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)

    #     parser.add_argument('--n_channels', type=int, default=3)
    #     parser.add_argument('--n_classes', type=int, default=1)
    #     return parser
