"""
Podstawowy Autoencoder (Fully-Connected) do detekcji anomalii malware.

Architektura:
    Encoder: 55 → 128 → 64 → 32 → latent_dim
    Decoder: latent_dim → 32 → 64 → 128 → 55
"""
import torch
import torch.nn as nn
from src.utils.config import AEConfig


class Encoder(nn.Module):
    """Enkoder Autoencodera — mapuje przestrzeń wejściową na reprezentację latentną."""

    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Decoder(nn.Module):
    """Dekoder Autoencodera — rekonstruuje wejście z reprezentacji latentnej."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, output_dim), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class Autoencoder(nn.Module):
    """Autoencoder do detekcji anomalii.
    
    Trenowany na próbkach benign. Wysoki MSE rekonstrukcji => anomalia (malware).
    """

    def __init__(self, cfg: AEConfig = None):
        super().__init__()
        cfg = cfg or AEConfig()
        self.encoder = Encoder(cfg.input_dim, cfg.hidden_dims, cfg.latent_dim, cfg.dropout)
        self.decoder = Decoder(cfg.latent_dim, cfg.hidden_dims, cfg.input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (rekonstrukcja, wektor latentny)"""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Oblicza błąd rekonstrukcji per próbka (MSE)."""
        x_hat, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)
