from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
	paths: Path = Path(".")
	seeds: int = 42
	resolution: int = 224
	backbone: str = "resnet18"
	lrs: float = 1e-3
	epochs: int = 10

