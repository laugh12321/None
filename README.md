# Lightning-UNet

```
pip install -r requirements.txt

bash scripts/download_data.sh

wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz &&
mv {MODEL_NAME}.npz ./model/vit_checkpoint/imagenet21k/{MODEL_NAME}.npz

python train.py -d --amp
```