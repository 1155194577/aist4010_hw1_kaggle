import torch
import wandb
from dataclasses import replace
from pathlib import Path

from config import Config, seed_everything
from dataset import get_train_val_split
from engine import training_loop

## TODO :　investigate possible GPU memory leakage across trials 
TRIALS = [
    {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 10,
        "augmentation": True,
        "optimizer": "Adam",
        "weight_decay": 0.0,
        "dropout": 0.3,
    },
    {
        "learning_rate": 3e-4,
        "batch_size": 32,
        "epochs": 10,
        "augmentation": True,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "dropout": 0.5,
    },
    {
        "learning_rate": 3e-4,
        "batch_size": 64,
        "epochs": 10,
        "augmentation": True,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "dropout": 0.5,
    },
    {
        "learning_rate": 1e-2,
        "batch_size": 32,
        "epochs": 10,
        "augmentation": True,
        "optimizer": "SGD",
        "weight_decay": 1e-4,
        "dropout": 0.5,
    },
    {
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 10,
        "augmentation": False,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "dropout": 0.3,
    },
]


def main() -> None:
    base_cfg = Config()
    for trial_id, params in enumerate(TRIALS, start=1):
        cfg = replace(
            base_cfg,
            lr=params["learning_rate"],
            optimizer=params["optimizer"],
            weight_decay=params["weight_decay"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            augmentation=params["augmentation"],
            dropout=params["dropout"],
            checkpoint_path=Path("checkpoints") / f"trial_{trial_id:03d}_best.pt",
        )
        seed_everything(cfg.seed)
        with wandb.init(
            project="aist4010-hw1",
            name=f"trial-{trial_id:03d}",
            group="manual-hyperparameter-trials",
            config={**params, "trial_id": trial_id},
        ) as run:
            train_loader, val_loader = get_train_val_split(cfg)
            num_classes = len(train_loader.dataset.classes)
            model = cfg.create_model(num_classes)
            wandb.watch(model, log="all", log_freq=100)
            optimizer = cfg.create_optimizer(model)
            scheduler = cfg.create_scheduler(optimizer, len(train_loader))
            criterion = cfg.create_criterion()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_val_loss = training_loop(
                model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                cfg,
                run=run,
                scheduler=scheduler,
            )
            run.summary["best_val_loss"] = best_val_loss


if __name__ == "__main__":
    main()
