# 🏋️‍♀️ Weights & Biases
import wandb
# 🍦 Vanilla PyTorch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
# ⚡ PyTorch Lightning
import pytorch_lightning as pl
# ⚡ 🤝 🏋️‍♀️
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from self_attention_cv.transunet import TransUnet

from utils.dice_score import dice_loss
from utils.dice_score import multiclass_dice_coeff, dice_coeff


class TransUnet(pl.LightningModule):
    def __init__(self, 
        n_channels: int, 
        n_classes: int, 
        lr: float = 1e-5, 
        amp: bool = False):
        super(TransUnet, self).__init__()
        self.save_hyperparameters()
        self.grad_scaler = GradScaler(enabled=amp)
        self.criterion = nn.CrossEntropyLoss()
        self.model = TransUnet(in_channels=n_channels, classes=n_classes, 
                               img_dim=64, vit_blocks=8, vit_dim_linear_mhsa_block=128)
        
    @autocast()
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We could make the optimizer more fancy by adding a scheduler and specifying which parameters do
        # not require weight_decay but just using AdamW out-of-the-box works fine
        optimizer = optim.RMSprop(self.parameters(), lr=self.hparams.lr, weight_decay=1e-8, momentum=0.9)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    
        return [optimizer], [lr_scheduler]
    
    def _common_step(self, batch):
        images, true_masks = batch
        masks_pred = self(images)
        loss = self.criterion(masks_pred, true_masks) \
                + dice_loss(F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, 2).permute(0, 3, 1, 2).float(),
                            multiclass=True)
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

    def validation_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        avg_loss = torch.stack([x for x in outputs]).mean()
        sch.step(self.trainer.callback_metrics["val_avg_loss"])
        self.log("val_avg_loss", avg_loss, on_epoch=True)

        # dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        # model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        # torch.onnx.export(self, dummy_input, model_filename, opset_version=11)
        # wandb.save(model_filename)

        # flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        # self.logger.experiment.log(
        #     {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
        #     "global_step": self.global_step})

        return avg_loss


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
            },
            "global_step": trainer.global_step
            })