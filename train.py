from config import Config, seed_everything
from engine import training_loop
import torch
from dataset import get_train_val_split
from models import get_model
import wandb


def main():
    cfg = Config()
    seed_everything(cfg.seed)
    with wandb.init(project="aist4010-hw1", config=cfg.__dict__) as run:
        train_loader, val_loader = get_train_val_split(cfg)
        num_classes = len(train_loader.dataset.classes)
        model = get_model(
            cfg.backbone, num_classes=num_classes, dropout=cfg.dropout
        )
        wandb.watch(model, log="all", log_freq=100)
        optimizer = cfg.create_optimizer(model)
        scheduler = cfg.create_scheduler(optimizer, len(train_loader))
        criterion = torch.nn.CrossEntropyLoss()
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