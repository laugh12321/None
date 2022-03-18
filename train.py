import argparse
from unet import UNet
from pathlib import Path
from utils.data_loading import Dataset

# 🏋️‍♀️ Weights & Biases
import wandb
# 🍦 Vanilla PyTorch
import torch
# ⚡ PyTorch Lightning
import pytorch_lightning as pl
# ⚡ 🤝 🏋️‍♀️
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=5):
        super().__init__()
        self.val_imgs, self.val_masks = val_samples['image'], val_samples['mask']
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_masks = self.val_masks[:num_samples]
          
    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        masks_pred = pl_module(val_imgs)

        trainer.logger.experiment.log({
            'images': wandb.Image(self.val_imgs[0]),
            'masks': {
                'true': wandb.Image(self.val_masks[0].float()),
                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float()),
            },
            "global_step": trainer.global_step
            })

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0.5,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

    return parser.parse_args()

if __name__ == '__main__':
    # init
    args = get_args()
    wandb_logger = WandbLogger(project="UNet")
    pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

    # setup data
    data = Dataset(images_dir=dir_img, masks_dir=dir_mask, scale=args.scale,
                    batch_size=args.batch_size, val=args.val)
    data.prepare_data()
    data.setup()
    # grab samples to log predictions on
    samples = next(iter(data.val_dataloader()))

    # setup model
    model = UNet(n_channels=3, n_classes=2, bilinear=args.bilinear, lr=args.lr)

    # checkpoints
    checkpoint = ModelCheckpoint(
        dirpath = dir_checkpoint,
        filename ='UNet-{epoch:02d}-{val_loss:.2f}',
        monitor ='validation_loss',
        save_top_k = 2,
        verbose = True,
    )

    # earlystopping
    earlystopping = EarlyStopping(
        monitor = 'validation_loss',
        mode = 'min',
        patience = 5, 
        verbose = True,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,    # W&B integration
        # log_every_n_steps=50,   # set the logging frequency
        gpus=-1,                # use all GPUs
        max_epochs=args.epochs,           # number of epochs
        # deterministic=True,     # keep it deterministic
        callbacks=[checkpoint, earlystopping, ImagePredictionLogger(samples)] # see Callbacks section
        )
    
    # fit the model
    trainer.fit(model, data)

    wandb.finish()