from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    paths: Path = Path("./data")
    seeds: int = 42
    resolution: int = 128
    val_frac: float = 0.2
    batch_size: int = 32
    num_workers: int = 4 ## use 4 worker when have gpu 
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    augmentation: bool = True
    dropout: float = 0.5
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_scale = (0.8, 1.0)
    crop_ratio = (0.75, 1)
    train_dir : Path = Path("./data/train")
    val_dir : Path = Path("./data/val")
    test_dir : Path = Path("./data/test")
    optimizer: str = "Adam"
    backbone : str = "simpleCNN" ## TODO : add resnet 
    checkpoint_path: Path = Path("./checkpoints/best_model.pt")

    def create_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        try:
            optimizer_class = getattr(torch.optim, self.optimizer)
        except AttributeError as error:
            raise ValueError(
                f"Unsupported optimizer '{self.optimizer}'. "
                "Choose an optimizer available in torch.optim, such as "
                "'Adam' or 'SGD'."
            ) from error

        if not isinstance(optimizer_class, type) or not issubclass(
            optimizer_class, torch.optim.Optimizer
        ):
            raise ValueError(
                f"'{self.optimizer}' is not a valid torch.optim optimizer class."
            )

        return optimizer_class(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
    
def to_dict(cfg: Config) -> dict:
    return {
        "paths": str(cfg.paths),
        "seeds": cfg.seeds,
        "resolution": cfg.resolution,
        "val_frac": cfg.val_frac,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "backbone": cfg.backbone,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "epochs": cfg.epochs,
        "augmentation": cfg.augmentation,
        "dropout": cfg.dropout,
        "mean": cfg.mean,
        "std": cfg.std,
        "crop_scale": cfg.crop_scale,
        "crop_ratio": cfg.crop_ratio,
        "train_dir": str(cfg.train_dir),
        "val_dir": str(cfg.val_dir),
        "test_dir": str(cfg.test_dir),
        "optimizer": cfg.optimizer,
        "checkpoint_path": str(cfg.checkpoint_path),
    }