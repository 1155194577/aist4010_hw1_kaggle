import argparse
import os
from dataclasses import replace
from pathlib import Path

import torch

from config import Config, seed_everything
from dataset import get_train_val_split
from engine import training_loop


CONFIG_FIELDS = (
    "seed",
    "backbone",
    "head",
    "margin",
    "scale",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "dropout",
    "resolution",
    "random_erasing",
    "ema_decay",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the face-identification model. "
        "Any flag left out keeps the Config default."
    )
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--backbone", choices=("resnet18", "resnet34", "simpleCNN"))
    parser.add_argument("--head", choices=("arcface", "cosface", "linear"))
    parser.add_argument("--margin", type=float, help="ArcFace/CosFace margin")
    parser.add_argument("--scale", type=float, help="cosine scale for margin heads")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--random-erasing", type=float)
    parser.add_argument("--ema-decay", type=float, help="0 disables EMA")
    parser.add_argument("--checkpoint", type=Path, help="output checkpoint path")
    parser.add_argument("--run-name", help="W&B run name")
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from a *_last.pt checkpoint, or the literal 'last'",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="power off the machine when training finishes (AutoDL/cloud instances)",
    )
    return parser.parse_args(argv)


def build_config(args) -> Config:
    overrides = {
        field: getattr(args, field)
        for field in CONFIG_FIELDS
        if getattr(args, field) is not None
    }
    if args.checkpoint is not None:
        overrides["checkpoint_path"] = args.checkpoint
    cfg = replace(Config(), **overrides)

    # Running a second seed must not overwrite the first run's checkpoint, so give
    # seed/backbone/head variants their own file unless --checkpoint was passed.
    if args.checkpoint is None and any(
        getattr(args, field) is not None for field in ("seed", "backbone", "head")
    ):
        cfg = replace(
            cfg,
            checkpoint_path=Path("checkpoints")
            / f"{cfg.backbone}_{cfg.head}_seed{cfg.seed}.pt",
        )
    return cfg


def run(args):
    cfg = build_config(args)
    seed_everything(cfg.seed)

    import wandb  # imported here so `--help` works without the dependency

    run_name = args.run_name or f"{cfg.backbone}-{cfg.head}-s{cfg.seed}-e{cfg.epochs}"
    with wandb.init(
        project="aist4010-hw1", config=cfg.__dict__, name=run_name
    ) as run:
        train_loader, val_loader = get_train_val_split(cfg)
        num_classes = len(train_loader.dataset.classes)
        model = cfg.create_model(num_classes)
        print(
            f"backbone={cfg.backbone} head={cfg.head} res={cfg.resolution} "
            f"epochs={cfg.epochs} batch={cfg.batch_size} lr={cfg.lr} seed={cfg.seed} "
            f"ema={cfg.ema_decay} erasing={cfg.random_erasing} "
            f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
        )
        print(f"checkpoint={cfg.checkpoint_path} run={run_name}")
        wandb.watch(model, log="all", log_freq=100)
        optimizer = cfg.create_optimizer(model)
        scheduler = cfg.create_scheduler(optimizer, len(train_loader))
        criterion = cfg.create_criterion()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        start_epoch = 0
        best_val_loss = float("inf")
        best_val_accuracy = -1.0
        resume_path = args.resume
        if resume_path is not None and str(resume_path) == "last":
            resume_path = cfg.checkpoint_path.with_name(
                f"{cfg.checkpoint_path.stem}_last.pt"
            )
        if resume_path is not None:
            if not Path(resume_path).exists():
                # Lets the same command be used on a fresh spot instance: if the
                # checkpoint is not there yet, start from scratch instead of failing.
                print(f"no checkpoint at {resume_path}; starting from scratch")
                resume_path = None
        if resume_path is not None:
            state = torch.load(resume_path, map_location=device)
            model.load_state_dict(state["model_state"])
            if state.get("optimizer_state") is not None:
                optimizer.load_state_dict(state["optimizer_state"])
            if scheduler is not None and state.get("scheduler_state") is not None:
                scheduler.load_state_dict(state["scheduler_state"])
            start_epoch = int(state.get("epoch", 0))
            best_val_loss = float(state.get("best_val_loss", best_val_loss))
            best_val_accuracy = float(state.get("best_val_accuracy", best_val_accuracy))
            print(
                f"resumed {resume_path}: epoch {start_epoch}/{cfg.epochs}, "
                f"best val acc {best_val_accuracy:.4f}, "
                f"best val loss {best_val_loss:.4f}"
            )
            if start_epoch >= cfg.epochs:
                print("checkpoint already reached cfg.epochs; nothing to train")

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
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
        )
        run.summary["best_val_loss"] = best_val_loss
        run.summary["checkpoint"] = str(cfg.checkpoint_path)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    finally:
        # Runs even if training raised, so a crashed job does not keep billing.
        # Logs must already be redirected to a file: stdout is lost on shutdown.
        if args.shutdown:
            shutdown_bin = "/usr/bin/shutdown"
            if os.path.exists(shutdown_bin):
                print("shutting down the instance")
                os.system(shutdown_bin)
            else:
                print("--shutdown requested but /usr/bin/shutdown does not exist")


if __name__ == "__main__":
    main()
