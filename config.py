from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
	paths: Path = Path("./data")
	seeds: int = 42
	resolution: int = 224
	backbone: str = "resnet18"
	lrs: float = 1e-3
	epochs: int = 10

def to_dict(cfg: Config) -> dict:
    return {
        "paths": str(cfg.paths),
        "seeds": cfg.seeds,
        "resolution": cfg.resolution,
        "backbone": cfg.backbone,
        "lrs": cfg.lrs,
        "epochs": cfg.epochs
    }