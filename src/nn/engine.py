import torch


def train_step(model, fold_loaders, criterion, optimizer, device):
    model.train().to(device)

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for i, data_loader in enumerate(fold_loaders, start=1):
        # print(f"Fold pair number: {i}")
        for X, y in data_loader[0]:
            X, y = X.to(device), y.to(device)

            train_pred = model(X)

            loss = criterion(train_pred, y)
            train_loss += loss.item() * X.size(0)

            train_pred_class = (torch.sigmoid(train_pred) >= 0.5).float()
            train_correct += (train_pred_class == y).sum().item()
            train_total += y.numel()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_loss /= train_total
    train_acc = train_correct / train_total
    return train_loss, train_acc


def val_step(model, fold_loader, criterion, device):
    model.eval().to(device)

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for data_loader in fold_loader:
            for X, y in data_loader[1]:
                X, y = X.to(device), y.to(device)

                val_pred = model(X)

                loss = criterion(val_pred, y)
                val_loss += loss.item() * X.size(0)

                val_pred_class = (torch.sigmoid(val_pred) >= 0.5).float()
                val_correct += (val_pred_class == y).sum().item()
                val_total += y.numel()

    val_loss /= val_total
    val_acc = val_correct / val_total
    return val_loss, val_acc


def fit(model, fold_loaders, criterion, optimizer, device, epochs):
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step(
            model=model,
            fold_loaders=fold_loaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device
            )
        val_loss, val_acc = val_step(
            model=model,
            fold_loader=fold_loaders,
            criterion=criterion,
            device=device
        )
        print(
            f"Epoch {epoch}/{epochs} | ",
            f"Train loss: {train_loss:.04f} | "
            f"Train acc: {train_acc:.04f} | "
            f"Val loss: {val_loss:.04f} | "
            f"Val acc: {val_acc:.04f} | "
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
    return results
