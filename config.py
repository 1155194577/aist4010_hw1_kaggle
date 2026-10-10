from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    paths: Path = Path("./data")
    seeds: int = 42
    resolution: int = 224
    val_frac: float = 0.2
    batch_size: int = 32
    num_workers: int = 0
    backbone: str = "resnet18"
    lrs: float = 1e-3
    epochs: int = 10
    augmentation: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    crop_scale: tuple[float, float] = (0.8, 1.0)
    crop_ratio: tuple[float, float] = (0.75, 1)
    train_dir : Path = Path("./data/train")
    val_dir : Path = Path("./data/val")
    test_dir : Path = Path("./data/test")
    
    
def to_dict(cfg: Config) -> dict:
    return {
        "paths": str(cfg.paths),
        "seeds": cfg.seeds,
        "resolution": cfg.resolution,
        "val_frac": cfg.val_frac,
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "backbone": cfg.backbone,
        "lrs": cfg.lrs,
        "epochs": cfg.epochs,
        "augmentation": cfg.augmentation,
        "mean": cfg.mean,
        "std": cfg.std,
        "crop_scale": cfg.crop_scale,
        "crop_ratio": cfg.crop_ratio,
        "train_dir": str(cfg.train_dir),
        "val_dir": str(cfg.val_dir),
    }