from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.datasets import Cityscapes as TorchvisionCityscapes
from torchvision.datasets import VOCSegmentation as TorchvisionVOCSegmentation


class VOCSegmentation(TorchvisionVOCSegmentation):
    """VOC wrapper that supports joint image/mask transforms and trainaug labels."""

    cmap = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
        [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
        [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
        [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
        [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
    ], dtype=np.uint8)

    def __init__(self, root, year="2012", image_set="train", download=False, transform=None):
        torchvision_year = "2012" if year == "2012_aug" else year
        torchvision_image_set = "train" if year == "2012_aug" and image_set == "train" else image_set
        super().__init__(
            root=root,
            year=torchvision_year,
            image_set=torchvision_image_set,
            download=download,
            transforms=None,
        )
        self.transform = transform

        if year == "2012_aug" and image_set == "train":
            self._use_augmented_train_set(root)

    def _use_augmented_train_set(self, root):
        voc_root = Path(root) / "VOCdevkit" / "VOC2012"
        split_file = voc_root / "ImageSets" / "Segmentation" / "trainaug.txt"
        mask_dir = voc_root / "SegmentationClassAug"
        if not split_file.exists() or not mask_dir.exists():
            return

        names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.images = [str(voc_root / "JPEGImages" / f"{name}.jpg") for name in names]
        self.masks = [str(mask_dir / f"{name}.png") for name in names]

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    @classmethod
    def decode_target(cls, mask):
        mask = np.asarray(mask).copy()
        mask[mask == 255] = 0
        return cls.cmap[mask]


class Cityscapes(TorchvisionCityscapes):
    """Cityscapes wrapper that returns train IDs and project-style decoded colors."""

    id_to_train_id = np.full(256, 255, dtype=np.uint8)
    for _id, _train_id in {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8,
        22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16,
        32: 17, 33: 18,
    }.items():
        id_to_train_id[_id] = _train_id

    train_id_to_color = np.array([
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0],
    ], dtype=np.uint8)

    def __init__(self, root, split="train", transform=None):
        super().__init__(root=root, split=split, mode="fine", target_type="semantic", transforms=None)
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        target = Image.fromarray(self.encode_target(np.asarray(target)))
        if self.transform is not None:
            image, target = self.transform(image, target)
        return image, target

    @classmethod
    def encode_target(cls, target):
        target = np.asarray(target)
        return cls.id_to_train_id[target]

    @classmethod
    def decode_target(cls, mask):
        mask = np.asarray(mask).copy()
        mask[mask == 255] = 19
        return cls.train_id_to_color[mask]


cityscapes = Cityscapes
