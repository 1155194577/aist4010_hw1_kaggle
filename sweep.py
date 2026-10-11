import torch
import wandb
from dataclasses import replace
from pathlib import Path

from config import Config
from dataset import get_train_val_split
from engine import training_loop
from models import get_model


TRIALS = [
    {"learning_rate": 1e-3, "batch_size": 32, "epochs": 10, "augmentation": True},
    {"learning_rate": 3e-4, "batch_size": 32, "epochs": 10, "augmentation": True},
    {"learning_rate": 1e-3, "batch_size": 64, "epochs": 10, "augmentation": False},
]


def main() -> None:
    base_cfg = Config()
    for trial_id, params in enumerate(TRIALS, start=1):
        cfg = replace(
            base_cfg,
            lr=params["learning_rate"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            augmentation=params["augmentation"],
            checkpoint_path=Path("checkpoints") / f"trial_{trial_id:03d}_best.pt",
        )
        with wandb.init(
            project="aist4010-hw1",
            name=f"trial-{trial_id:03d}",
            group="manual-hyperparameter-trials",
            config={**params, "trial_id": trial_id},
        ) as run:
            train_loader, val_loader = get_train_val_split(cfg)
            num_classes = len(train_loader.dataset.classes)
            model = get_model(cfg.backbone, num_classes=num_classes)
            wandb.watch(model, log="all", log_freq=100)
            optimizer = cfg.create_optimizer(model)
            criterion = torch.nn.CrossEntropyLoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_val_loss = training_loop(
                model, train_loader, val_loader, optimizer, criterion, device, cfg, run
            )
            run.summary["best_val_loss"] = best_val_loss


if __name__ == "__main__":
    main()
