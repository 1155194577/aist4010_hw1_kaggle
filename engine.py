import torch
from tqdm import tqdm

from metrics import evaluate_model


def training_loop(model, train_loader, val_loader, optimizer, criterion, device, cfg, run=None):
    model.to(device)
    best_val_loss = float('inf')
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in tqdm(range(cfg.epochs), desc='Epochs'):
        model.train()
        running_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch + 1} Training', leave=False)
        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_progress.set_postfix(loss=f'{loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{cfg.epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_labels = []
        val_predictions = []
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f'Epoch {epoch + 1} Validation', leave=False)
            for images, labels in val_progress:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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
        print(f'Validation Classification Report:\n{val_metrics["classification_report"]}')
        if run is not None:
            run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_epoch_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_score": val_metrics["f1_score"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Checkpointing
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}')

    return best_val_loss