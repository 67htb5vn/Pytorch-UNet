from pathlib import Path

import numpy as np
from PIL import Image

from utils.data_loading import BasicDataset


def test_basic_dataset_uses_provided_mask_values_without_scanning(tmp_path):
    images_dir = tmp_path / 'images'
    masks_dir = tmp_path / 'masks'
    images_dir.mkdir()
    masks_dir.mkdir()

    image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((4, 4), dtype=np.uint8))

    image_path = images_dir / 'sample.png'
    mask_path = masks_dir / 'sample.png'
    image.save(image_path)
    mask.save(mask_path)

    dataset = BasicDataset(str(images_dir), str(masks_dir), scale=1.0, mask_values=[0, 1], scan_limit=1)

    assert dataset.mask_values == [0, 1]
