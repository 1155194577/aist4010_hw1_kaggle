from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from config import Config
from torch.utils.data import DataLoader, Dataset


class TestImageDataset(Dataset):
    def __init__(self, root, transform) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_path.name


def make_transform(cfg: Config, train: bool = True) -> transforms.Compose:
    transform_list = []
    if train and cfg.augmentation:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(), ## not for face classification
                transforms.RandomRotation(degrees=15, fill=(128, 128, 128)),
                transforms.RandomResizedCrop(
                    cfg.resolution,
                    scale=cfg.crop_scale,
                    ratio=cfg.crop_ratio,
                ),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
            ]
        )
    else:
        transform_list.append(transforms.Resize((cfg.resolution, cfg.resolution)))

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    if train and cfg.augmentation and cfg.random_erasing > 0:
        # Cutout-style occlusion: cheap robustness for faces and a reliable small gain.
        transform_list.append(
            transforms.RandomErasing(
                p=cfg.random_erasing, scale=(0.02, 0.15), value="random"
            )
        )
    return transforms.Compose(transform_list)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_val_split(cfg):
    train_dataset = ImageFolder(cfg.train_dir, transform=make_transform(cfg, True))
    val_dataset = ImageFolder(cfg.val_dir, transform=make_transform(cfg, False))
    generator = torch.Generator().manual_seed(cfg.seed)

    # prefetch_factor / persistent_workers are only valid with worker processes.
    loader_kwargs = {
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": cfg.num_workers > 0,
        "prefetch_factor": 4 if cfg.num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
        **loader_kwargs,
    )

    return train_loader, val_loader

def get_test_loader(cfg):
    test_dataset = TestImageDataset(cfg.test_dir, transform=make_transform(cfg, False))
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    return test_loader
