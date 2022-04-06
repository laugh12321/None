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
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

dir_root = Path('./data/')
dir_checkpoint = Path('./checkpoints/')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', '-l', metavar='LR', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_classes', '-n', type=int, default=2, help='output channel of network')
    parser.add_argument('--val', '-v', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    # init
    args = get_args()
    wandb_logger = WandbLogger(project="TransUnet")
    pl.seed_everything(hash("setting random seeds") % 2**32 - 1)

    # setup data
    data = CarvanaDataset(root_dir=dir_root, batch_size=args.batch_size, 
                          val_percent=args.val, num_workers=8)
    data.prepare_data()
    data.setup()

    # grab samples to log predictions on
    samples = next(iter(data.val_dataloader()))

    # setup model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = TransUnet(config_vit, img_size=args.img_size, lr=args.lr, amp=args.amp)

    # checkpoints
    checkpoint = ModelCheckpoint(
        dirpath = dir_checkpoint,
        filename ='TransUnet-{epoch:02d}-{val_loss:.2f}',
        monitor ='val_loss',
        save_top_k = 2,
        verbose = True,
    )

    # earlystopping
    earlystopping = EarlyStopping(
        monitor = 'val_loss',
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