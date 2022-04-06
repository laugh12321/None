# 🏋️‍♀️ Weights & Biases
import wandb
# 🍦 Vanilla PyTorch
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
# ⚡ PyTorch Lightning
import pytorch_lightning as pl
import numpy as np

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils.dice_score import DiceLoss


class TransUnet(pl.LightningModule):
    def __init__(self, 
        config, 
        img_size: int = 224, 
        lr: float = 0.01, 
        amp: bool = True):
        super(TransUnet, self).__init__()
        # self.save_hyperparameters()
        self.base_lr = lr
        self.grad_scaler = GradScaler(enabled=amp)
        self.dice_loss = DiceLoss(config.n_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.ViT_seg = ViT_seg(config=config, img_size=img_size, num_classes=config.n_classes)
        self.ViT_seg.load_from(weights=np.load(config.pretrained_path))
        
    @autocast()
    def forward(self, x):
        return self.ViT_seg(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=0.0001)
        return [optimizer]

    @property
    def automatic_optimization(self) -> bool:
        return False
    
    def _common_step(self, batch):
        images, true_masks = batch
        true_masks = torch.argmax(true_masks, dim=1)
        masks_pred = self(images)
        loss_ce = self.ce_loss(masks_pred, true_masks)
        loss_dice = self.dice_loss(masks_pred, true_masks, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch)
        opt = self.optimizers()
        opt.zero_grad()

        if self.grad_scaler is not None:
            self.manual_backward(self.grad_scaler.scale(loss))
            self.grad_scaler.step(opt)
            self.grad_scaler.update()
        else:
            self.manual_backward(loss)
  
        self.log("train_loss", loss)
        self.logger.experiment.log({
            'learning rate': opt.param_groups[0]['lr'],
        })

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch)    
        self.log("val_loss", loss, on_epoch=True)
        return loss


class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=5):
        super().__init__()
        self.val_imgs, self.val_masks = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_masks = self.val_masks[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        masks_pred = pl_module(val_imgs)

        trainer.logger.experiment.log({
            'images': wandb.Image(self.val_imgs[0].to("cpu")),
            'masks': {
                'true': wandb.Image(self.val_masks[0].float().to("cpu")),
                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().to("cpu")),
            }})