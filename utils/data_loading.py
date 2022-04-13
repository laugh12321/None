from PIL import Image, ImageDraw

from os.path import splitext
from os import listdir
import os

from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import base64
import json

import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, root_dir: str, size: int = 224, mask_suffix: str = ''):
        self.images_dir = Path(root_dir, 'imgs')
        self.masks_dir = Path(root_dir, 'masks')
        self.size = size
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, size, is_mask):
        pil_img = pil_img.resize((size, size), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, root_dir: str, size: int = 224):
        super().__init__(root_dir, size, mask_suffix='_mask')


def base64_to_image(base64_code, save_dir):
    """
    将base64编码写入为图片
    base64_code: base64编码后数据
    save_dir: 图片保存位置
    """
    # base64解码
    img_data = base64.b64decode(base64_code)
    img_file = open(save_dir, 'wb')
    img_file.write(img_data)
    img_file.close()


def json_to_mask(save_dir, json_dir):
    """
    将json中的信息转换为可训练的标准类型
    """
    images_dir = Path(save_dir, 'imgs')
    masks_dir = Path(save_dir, 'masks')

    ids = [splitext(file)[0] for file in listdir(json_dir) if file.endswith('.json')]
    if not ids:
        print(f'No input file found in {json_dir}, make sure you put your json file there')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)

    for id in tqdm(ids):
        mask_data = json.load(open(Path(json_dir, id + '.json'), 'r'))

        # 生成 image
        base64_to_image(base64_code=mask_data["imageData"], save_dir=Path(images_dir, id + '.jpg'))

        # 生成 mask
        polygons = mask_data["shapes"]
        h, w = mask_data["imageHeight"], mask_data["imageWidth"]
        mask_img = Image.new('L', (w, h), 0)   
        for polygon in polygons:
            points = [tuple(l) for l in polygon["points"]]
            ImageDraw.Draw(mask_img).polygon(points, outline=255, fill=255)
        mask_img.save((Path(masks_dir, id + '_mask.gif')))

    print("Transform Finished!")