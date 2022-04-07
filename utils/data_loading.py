import logging
from os import listdir
from os.path import splitext
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset

class BasicDataset(VisionDataset):
    def __init__(self,
                 root: str,
                 mask_suffix: str = '',
                 transforms: Optional[Callable] = None) -> None:
        super(BasicDataset, self).__init__(root, transforms)
        self.images_dir = Path(root, 'imgs')
        self.masks_dir = Path(root, 'masks')
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(self.images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {self.images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        img = Image.open(img_file[0]).convert('RGB')
        mask = Image.open(mask_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return img, mask