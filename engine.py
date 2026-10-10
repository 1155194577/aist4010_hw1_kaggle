import torch
def training_loop(model, train_loader, val_loader, optimizer, criterion, device, cfg):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{cfg.epochs}, Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{cfg.epochs}, Validation Loss: {val_epoch_loss:.4f}')

        # Checkpointing
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), cfg.checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}')