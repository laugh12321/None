from os.path import splitext, exists
from os import listdir, makedirs
from labelme import utils
from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import base64
import imgviz
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


def json_to_dataset(save_dir, json_dir, noviz=True):
    """
    将json中的信息转换为可训练的标准类型
    
    save_dir: path, 数据集保存位置
    json_dir: path, json存放位置
    noviz: bool, 是否可视化
    """
    ids = [splitext(file)[0] for file in listdir(json_dir) if file.endswith('.json')]
    if not ids:
        print(f'No input file found in {json_dir}, make sure you put your json file there')

    images_dir = Path(save_dir, 'imgs')
    masks_dir = Path(save_dir, 'masks')
    
    if not exists(images_dir):
        makedirs(images_dir)
    if not exists(masks_dir):
        makedirs(masks_dir)
        
    if not noviz:
        viz_dir = Path(save_dir, 'viz')
        if not exists(viz_dir):
            makedirs(viz_dir)

    class_id = 1
    class_names = ["_background_"]
    class_name_to_id = {"_background_": 0}
    for id in tqdm(ids):
        json_data = json.load(open(Path(json_dir, id + '.json'), encoding='utf-8'))
        
        for shape in json_data['shapes']:
            class_name = shape['label']
            if class_name not in class_names:
                class_names.append(class_name)
            if class_name not in class_name_to_id.keys():
                class_name_to_id[class_name] = class_id
                class_id += 1
            
        # 获取图像数据
        if json_data['imageData']:
            image_data = json_data['imageData']
        else:
            image_path = Path(json_dir, json_data['imagePath'])
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        img = utils.img_b64_to_arr(image_data)
        imgviz.io.imsave(Path(images_dir, id + '.jpg'), img)
        
        # 生成 mask
        lbl, _ = utils.shapes_to_label(
            img_shape=img.shape,
            shapes=json_data['shapes'],
            label_name_to_value=class_name_to_id,
        )
        utils.lblsave(Path(masks_dir, id + '_mask.png'), lbl)
        
        # 可视化
        if not noviz:
            viz = imgviz.label2rgb(
                label=lbl,
                image=imgviz.rgb2gray(img),
                # image=img,
                label_names=class_names,
                font_size=15,
                loc="rb",
            )
            imgviz.io.imsave(Path(viz_dir, id + '_viz.jpg'), viz)

    class_names = tuple(class_names)
    out_class_names_file = Path(save_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)
    print("Finished!")