"""
Klasyfikator na bazie zamrożonego enkodera AE.

Pipeline: zamrożony Encoder (AE) → Latent Space → FC head → 4 klasy
Trenowany w dwóch fazach:
  1. Trening AE na danych benign (anomaly detection)
  2. Fine-tuning głowicy klasyfikacyjnej na wszystkich klasach
"""
import torch
import torch.nn as nn
from src.models.autoencoder import Autoencoder
from src.utils.config import ClassifierConfig, AEConfig


class AEClassifier(nn.Module):
    """AE Encoder + głowica klasyfikacyjna.
    
    Pozwala ocenić jakość reprezentacji latentnej AE jako feature extractora.
    """

    def __init__(self, ae: Autoencoder, cfg: ClassifierConfig = None):
        super().__init__()
        cfg = cfg or ClassifierConfig()

        self.encoder = ae.encoder

        # Zamrożenie enkodera (opcjonalne)
        if cfg.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Głowica klasyfikacyjna
        self.head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns: logity (N, num_classes)"""
        z = self.encoder(x)
        return self.head(z)

    def unfreeze_encoder(self) -> None:
        """Odmraża enkoder do fine-tuningu end-to-end."""
        for param in self.encoder.parameters():
            param.requires_grad = True
