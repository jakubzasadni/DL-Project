"""
Pętla treningowa dla Autoencodera i VAE.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import RESULTS_MODELS_DIR, SEED


def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float = 1e-5,
    model_name: str = "ae",
) -> dict[str, list[float]]:
    """Trenuje Autoencoder (AE lub VAE) i zapisuje najlepszy model.
    
    Returns:
        history: {"train_loss": [...], "val_loss": [...]}
    """
    set_seed()
    device = get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Trening ---
        model.train()
        total_train = 0.0
        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            x = x.to(device)
            optimizer.zero_grad()

            if hasattr(model, "loss"):  # VAE
                x_hat, mu, log_var = model(x)
                losses = model.loss(x, x_hat, mu, log_var)
                loss = losses["total"]
            else:  # AE
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)

            loss.backward()
            optimizer.step()
            total_train += loss.item()

        # --- Walidacja ---
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                if hasattr(model, "loss"):  # VAE
                    x_hat, mu, log_var = model(x)
                    losses = model.loss(x, x_hat, mu, log_var)
                    total_val += losses["total"].item()
                else:
                    x_hat, _ = model(x)
                    total_val += criterion(x_hat, x).item()

        train_loss = total_train / len(train_loader)
        val_loss = total_val / len(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        # Zapis najlepszego modelu
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(RESULTS_MODELS_DIR, f"{model_name}_best.pt")
            torch.save(model.state_dict(), save_path)

    print(f"Training done. Best val_loss: {best_val_loss:.6f}")
    return history
