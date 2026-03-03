"""
Variational Autoencoder (VAE) do detekcji anomalii i uczenia rozkładu latentnego.

Strata: L = MSE(x, x_hat) + beta * KL(q(z|x) || p(z))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import VAEConfig


class VAE(nn.Module):
    """Variational Autoencoder.
    
    Przestrzeń latentna modeluje parametry rozkładu N(mu, sigma^2).
    Regularyzacja KL zapewnia lepszą separację klas w przestrzeni latentnej.
    """

    def __init__(self, cfg: VAEConfig = None):
        super().__init__()
        cfg = cfg or VAEConfig()
        self.latent_dim = cfg.latent_dim
        self.beta = cfg.beta

        # Enkoder — wspólne warstwy
        encoder_layers = []
        in_dim = cfg.input_dim
        for h_dim in cfg.hidden_dims:
            encoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ]
            in_dim = h_dim
        self.encoder_shared = nn.Sequential(*encoder_layers)

        # Parametry rozkładu latentnego
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_log_var = nn.Linear(in_dim, cfg.latent_dim)

        # Dekoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for h_dim in reversed(cfg.hidden_dims):
            decoder_layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ]
            in_dim = h_dim
        decoder_layers += [nn.Linear(in_dim, cfg.input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (mu, log_var)"""
        h = self.encoder_shared(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparametryzacja: z = mu + eps * std."""
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Podczas inferencji używamy średniej

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: (rekonstrukcja, mu, log_var)"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> dict[str, torch.Tensor]:
        """Strata VAE = rekonstrukcja + beta * KL."""
        rec_loss = F.mse_loss(x_hat, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total = rec_loss + self.beta * kl_loss
        return {"total": total, "reconstruction": rec_loss, "kl": kl_loss}

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """MSE rekonstrukcji per próbka — do detekcji anomalii."""
        x_hat, _, _ = self.forward(x)
        return torch.mean((x - x_hat) ** 2, dim=1)
