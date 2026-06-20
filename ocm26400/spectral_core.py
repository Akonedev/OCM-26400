"""SpectralCoreBlock — l'architecture SPECTRALE de l'utilisateur (FFT) comme noyau unifié.

C'est L'ARCHITECTURE du projet (fruit des recherches/découvertes de l'utilisateur,
spXLM v6 SpectralBlock). On la porte dans ocm26400 pour en faire le NOYAU DU MODÈLE
UNIFIÉ (OmniModel). On NE change PAS d'architecture — on unifie TOUT sous le noyau
spectral de l'utilisateur.

Mélangeur spectral FFT bidirectionnel :
  * rfft sur la séquence (O(L log L), PAS d'attention O(L²))
  * filtre fréquentiel complexe APPRIS (filter_real/imag) par dimension
  * stabilité Parseval : ||x||² = ||FFT(x)||²
  * résiduel + FFN (norm)

Adapté pour servir de noyau à l'OmniModel : accepte (B,d) [un AMV, L=1] ou (B,L,d)
[chaîne compositionnelle]. C'est le noyau UNIFIÉ — raisonnement, classification,
génération passent tous par LUI.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .amv import D_MODEL


class SpectralCoreBlock(nn.Module):
    """Noyau spectral FFT de l'utilisateur (architecture du projet). Noyau unifié."""

    def __init__(self, d_model: int = D_MODEL, seq_len: int = 64, bidirectional: bool = True):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.seq_len = seq_len

        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        # filtre fréquentiel complexe appris (per-dimension)
        scale = 1.0 / math.sqrt(d_model)
        self.filter_real = nn.Parameter(torch.randn(seq_len // 2 + 1, d_model) * scale + 1.0)
        self.filter_imag = nn.Parameter(torch.randn(seq_len // 2 + 1, d_model) * scale)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,d) -> (B,d) ; (B,L,d) -> (B,L,d). Mélange spectral FFT sur L."""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1); squeeze = True           # (B,1,d)
        B, L, D = x.shape

        h = self.norm1(x)
        h = self.in_proj(h)
        X_freq = torch.fft.rfft(h, dim=1)               # (B, F, D), F = L//2+1

        fr = self.filter_real[:X_freq.shape[1], :].unsqueeze(0)
        fi = self.filter_imag[:X_freq.shape[1], :].unsqueeze(0)
        X_real = X_freq.real * fr - X_freq.imag * fi
        X_imag = X_freq.real * fi + X_freq.imag * fr
        X_filtered = torch.complex(X_real, X_imag)

        y = torch.fft.irfft(X_filtered, n=L, dim=1)
        y = self.out_proj(y)

        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x.squeeze(1) if squeeze else x
