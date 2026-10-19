import copy

import torch
from tqdm import tqdm

from config import to_dict
from metrics import evaluate_model


class ModelEma:
    """Exponential moving average of the weights, evaluated instead of the raw model."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict()
        for name, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source[name].detach(), alpha=1 - self.decay)
            else:
                value.copy_(source[name])


def training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    cfg,
    run=None,
    scheduler=None,
):
    model.to(device)
    best_val_loss = float('inf')
    best_val_accuracy = -1.0
    global_step = 0
    ema = ModelEma(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in tqdm(range(cfg.epochs), desc='Epochs'):
        if hasattr(model, "set_epoch"):
            # Ramps the ArcFace/CosFace margin in over the first epochs.
            model.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        train_labels = []
        train_predictions = []
        lr_samples = []
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # Margin heads need the labels to subtract the margin from the target logit.
            outputs = model(images, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            current_lr = optimizer.param_groups[0]["lr"]
            lr_samples.append(current_lr)
            # NOTE: per-step run.log() here caused ~46k network round-trips per run.
            # The epoch-level log below already records the mean learning rate.

            running_loss += loss.item() * images.size(0)
            train_labels.extend(labels.detach().cpu().tolist())
            train_predictions.extend(outputs.detach().argmax(dim=1).cpu().tolist())
            train_progress.set_postfix(loss=f'{loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        train_metrics = evaluate_model(train_labels, train_predictions)
        print(f'Epoch {epoch+1}/{cfg.epochs}, Training Loss: {epoch_loss:.4f}')
        print(f'Training Accuracy: {train_metrics["accuracy"]:.4f}')
        print(f'Training F1 Score: {train_metrics["f1_score"]:.4f}')

        # Validation phase. Evaluate the EMA weights when they are enabled; they track
        # the raw model closely and are usually a little more accurate.
        if ema is not None:
            ema.update(model)
            eval_model = ema.module
        else:
            eval_model = model
        eval_model.eval()
        val_running_loss = 0.0
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation', leave=False)
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = eval_model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                val_labels.extend(labels.cpu().tolist())
                val_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
                val_progress.set_postfix(loss=f'{loss.item():.4f}')

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_metrics = evaluate_model(val_labels, val_predictions)
        print(f'Epoch {epoch+1}/{cfg.epochs}, Validation Loss: {val_epoch_loss:.4f}')
        print(f'Validation Accuracy: {val_metrics["accuracy"]:.4f}')
        print(f'Validation F1 Score: {val_metrics["f1_score"]:.4f}')
        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_score": train_metrics["f1_score"],
                    "val_loss": val_epoch_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_score": val_metrics["f1_score"],
                    "learning_rate": sum(lr_samples) / len(lr_samples),
                    "train_classification_report": train_metrics["classification_report"],
                    "val_classification_report": val_metrics["classification_report"],
                }
            )

        # Checkpointing: the Kaggle metric is accuracy, so select on accuracy rather than
        # on the smoothed loss (with label_smoothing=0.1 the loss keeps improving after
        # accuracy plateaus or degrades).
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state": eval_model.state_dict(),
                    "ema": ema is not None,
                    "backbone": cfg.backbone,
                    "dropout": cfg.dropout,
                    "resolution": cfg.resolution,
                    "mean": cfg.mean,
                    "std": cfg.std,
                    "class_names": train_loader.dataset.classes,
                    "head": cfg.head,
                    "margin": cfg.margin,
                    "scale": cfg.scale,
                    "bnneck": cfg.bnneck,
                    "num_classes": int(model.head.weight.shape[0]),
                    "embed_dim": int(model.embed_dim),
                    "val_accuracy": best_val_accuracy,
                    "val_loss": val_epoch_loss,
                    "epoch": epoch + 1,
                },
                cfg.checkpoint_path,
            )
            print(
                f'Checkpoint saved at epoch {epoch+1} '
                f'(val acc {best_val_accuracy:.4f}, val loss {val_epoch_loss:.4f})'
            )
        best_val_loss = min(best_val_loss, val_epoch_loss)

        # Keep a resumable checkpoint so a preempted run does not lose every epoch.
        last_path = cfg.checkpoint_path.with_name(
            f"{cfg.checkpoint_path.stem}_last.pt"
        )
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "epoch": epoch + 1,
                "cfg": to_dict(cfg),
            },
            last_path,
        )

    return best_val_loss