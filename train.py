import argparse
from pathlib import Path

# 🏋️‍♀️ Weights & Biases
import wandb
# ⚡ PyTorch Lightning
import pytorch_lightning as pl
# ⚡ 🤝 🏋️‍♀️
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.data_loading import CarvanaDataset
from networks.model import TransUnet, ImagePredictionLogger

dir_root = Path('./data/')
dir_checkpoint = Path('./checkpoints/')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    # init
    args = get_args()
    wandb_logger = WandbLogger(project="TransUnet")
    pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

    # setup data
    data = CarvanaDataset(root_dir=dir_root)
    data.prepare_data()
    data.setup()

    # grab samples to log predictions on
    samples = next(iter(data.val_dataloader()))

    # setup model
    model = TransUnet(n_channels=3, n_classes=2, lr=args.lr, amp=args.amp)

    # checkpoints
    checkpoint = ModelCheckpoint(
        dirpath = dir_checkpoint,
        filename ='TransUnet-{epoch:02d}-{val_loss:.2f}',
        monitor ='val_avg_loss',
        save_top_k = 2,
        verbose = True,
    )

    # earlystopping
    earlystopping = EarlyStopping(
        monitor = 'val_avg_loss',
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