import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


## TODO add early stopping
@dataclass
class Config:
    paths: Path = Path("./data")
    seed: int = 42
    resolution: int = 64
    val_frac: float = 0.2
    batch_size: int = 128
    num_workers: int = 8 ## use 4 worker when have gpu 
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 64   
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
    scheduler: str = "CosineAnnealingWarmup"
    warmup_epochs: int = 1
    backbone : str = "resnet18"  # "resnet18" or "resnet34" (both adapted to 64 px)
    # "arcface" / "cosface" use a margin over cosine similarity (needs labels at
    # forward time); "linear" is the old plain softmax head.
    head: str = "arcface"
    margin: float = 0.35
    scale: float = 30.0
    bnneck: bool = True
    margin_warmup_epochs: int = 2
    ema_decay: float = 0.999  # per-step weight average; 0 disables
    random_erasing: float = 0.25  # Cutout-style occlusion probability; 0 disables
    # Only used by the "linear" head; margin losses train better without smoothing.
    label_smoothing: float = 0.1
    checkpoint_path: Path = Path("./checkpoints/best_model.pt")

    def create_model(self, num_classes: int) -> torch.nn.Module:
        from models import get_model

        return get_model(
            self.backbone,
            num_classes=num_classes,
            dropout=self.dropout,
            head=self.head,
            margin=self.margin,
            scale=self.scale,
            bnneck=self.bnneck,
            margin_warmup_epochs=self.margin_warmup_epochs,
        )

    def create_criterion(self) -> torch.nn.Module:
        """Cross-entropy on the head's logits.

        Margin losses already include their own regularisation, so label smoothing is
        disabled for them; it is kept for the plain linear head.
        """
        smoothing = self.label_smoothing if self.head == "linear" else 0.0
        return torch.nn.CrossEntropyLoss(label_smoothing=smoothing)

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

    def create_scheduler(
        self, optimizer, steps_per_epoch):
        if self.scheduler != "CosineAnnealingWarmup":
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler}'. "
                "Currently supported: 'CosineAnnealingWarmup'."
            )

        total_steps = self.epochs * steps_per_epoch
        warmup_steps = max(1, self.warmup_epochs * steps_per_epoch)
        if total_steps <= 0:
            raise ValueError("The training schedule requires at least one step.")
        if warmup_steps >= total_steps:
            raise ValueError("warmup_epochs must be less than the total epochs.")

        def learning_rate_multiplier(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            cosine_step = step - warmup_steps
            cosine_steps = total_steps - warmup_steps
            return 0.5 * (
                1 + math.cos(math.pi * cosine_step / cosine_steps)
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=learning_rate_multiplier
        )
    
def to_dict(cfg: Config) -> dict:
    return {
        "paths": str(cfg.paths),
        "seed": cfg.seed,
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
        "scheduler": cfg.scheduler,
        "warmup_epochs": cfg.warmup_epochs,
        "head": cfg.head,
        "margin": cfg.margin,
        "scale": cfg.scale,
        "bnneck": cfg.bnneck,
        "margin_warmup_epochs": cfg.margin_warmup_epochs,
        "ema_decay": cfg.ema_decay,
        "random_erasing": cfg.random_erasing,
        "label_smoothing": cfg.label_smoothing,
        "checkpoint_path": str(cfg.checkpoint_path),
    }