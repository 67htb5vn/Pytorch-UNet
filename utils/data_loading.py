import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def _normalize_mask_values(mask_values):
    if mask_values is None:
        return None
    if isinstance(mask_values, (list, tuple, np.ndarray)):
        return [int(v) for v in list(mask_values)]
    return [int(mask_values)]


def load_image(filename, is_mask=False):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        image = Image.open(filename)
        if is_mask:
            return image.convert("L")
        return image.convert("RGB")


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file, is_mask=True))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def find_matching_mask(mask_dir, image_name, mask_suffix):
    if not mask_dir.exists():
        return None

    candidates = []
    for path in mask_dir.iterdir():
        if not path.is_file() or path.name.startswith('.'):
            continue
        stem = path.stem
        if stem == image_name or stem == f'{image_name}{mask_suffix}' or stem == f'{image_name}_mask':
            candidates.append(path)
        elif mask_suffix and stem.replace(mask_suffix, '') == image_name:
            candidates.append(path)

    if not candidates:
        return None

    preferred_names = [image_name, f'{image_name}{mask_suffix}', f'{image_name}_mask']
    for preferred_name in preferred_names:
        for path in candidates:
            if path.stem == preferred_name:
                return path

    return sorted(candidates)[0]


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '',
                 mask_values: list = None, scan_limit: int = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        image_files = []
        for file in sorted(self.images_dir.iterdir()):
            if file.is_file() and not file.name.startswith('.'):
                image_files.append(file)

        if not image_files:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        self.image_files_by_id = {}
        self.mask_files_by_id = {}
        self.ids = []
        for image_file in image_files:
            image_id = splitext(image_file.name)[0]
            mask_file = find_matching_mask(self.mask_dir, image_id, self.mask_suffix)
            if mask_file is None:
                logging.warning(f'Skipping {image_file.name}: no matching mask found')
                continue
            self.image_files_by_id[image_id] = image_file
            self.mask_files_by_id[image_id] = mask_file
            self.ids.append(image_id)

        if not self.ids:
            raise RuntimeError(f'No valid image-mask pairs found in {images_dir} and {mask_dir}')

        self.mask_values = _normalize_mask_values(mask_values)

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        if self.mask_values is not None:
            logging.info('Using provided mask values without scanning masks')
        else:
            logging.info('Scanning mask files to determine unique values')
            unique = []
            for idx in tqdm(self.ids[:scan_limit] if scan_limit is not None else self.ids):
                unique.append(unique_mask_values(idx, self.mask_dir, self.mask_suffix))

            if not unique:
                self.mask_values = []
            else:
                self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))

        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask, target_size=None):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        if target_size is not None:
            newW, newH = target_size

        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            values_to_map = [int(v) for v in mask_values]
            if img.ndim == 2:
                unique_values = np.unique(img)
                if len(unique_values) == len(values_to_map) and len(values_to_map) == 2:
                    values_to_map = [int(v) for v in unique_values]
            for i, v in enumerate(values_to_map):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = self.image_files_by_id.get(name)
        mask_file = self.mask_files_by_id.get(name)

        if img_file is None or mask_file is None:
            raise FileNotFoundError(f'No valid image/mask pair found for the ID {name}')

        mask = load_image(mask_file, is_mask=True)
        img = load_image(img_file, is_mask=False)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        target_size = (512, 512)
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False, target_size=target_size)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True, target_size=target_size)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, mask_values=None, scan_limit=None):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', mask_values=mask_values,
                         scan_limit=scan_limit)
