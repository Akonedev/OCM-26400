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
        # FFT en fp32 (rocFFT/cuFFT ne supportent pas bf16/fp16 sous AMP)
        in_dtype = h.dtype
        h32 = h.float()
        if self.bidirectional:
            X_freq = torch.fft.rfft(h32, dim=1)            # (B, F, D), F = L//2+1
            # filtre robuste à la longueur d'entrée : pad si L > seq_len, slice si L < seq_len
            F_in = X_freq.shape[1]
            fr = self.filter_real
            fi = self.filter_imag
            if F_in > fr.shape[0]:
                pad = F_in - fr.shape[0]
                fr = torch.cat([fr, fr.new_zeros(pad, fr.shape[1])], dim=0)
                fi = torch.cat([fi, fi.new_zeros(pad, fi.shape[1])], dim=0)
            fr = fr[:F_in].unsqueeze(0)
            fi = fi[:F_in].unsqueeze(0)
            X_real = X_freq.real * fr - X_freq.imag * fi
            X_imag = X_freq.real * fi + X_freq.imag * fr
            X_filtered = torch.complex(X_real, X_imag)
            y = torch.fft.irfft(X_filtered, n=L, dim=1).to(in_dtype)
        else:
            # CAUSAL (rapport 18, L10) : zero-pad 2L (conv linéaire, pas circulaire) + filtre
            # causal temporel (support [0,L) seulement) → pas de fuite du futur.
            Lin = L
            h32p = torch.cat([h32, h32.new_zeros(B, Lin, D)], dim=1)   # (B, 2L, D)
            X_freq = torch.fft.rfft(h32p, dim=1)
            kt = torch.fft.irfft(torch.complex(self.filter_real, self.filter_imag),
                                 n=2*Lin, dim=0)                       # (2L, D) noyau temporel
            cmask = torch.cat([torch.ones(Lin), torch.zeros(Lin)]).unsqueeze(1).to(kt.device)  # support [0,L)
            kt = kt * cmask
            kf = torch.fft.rfft(kt, n=2*Lin, dim=0)[:X_freq.shape[1]]  # (F, D) filtre causal
            fr = kf.real.unsqueeze(0); fi = kf.imag.unsqueeze(0)
            X_real = X_freq.real * fr - X_freq.imag * fi
            X_imag = X_freq.real * fi + X_freq.imag * fr
            y = torch.fft.irfft(torch.complex(X_real, X_imag), n=2*Lin, dim=1)[:, :Lin].to(in_dtype)
        y = self.out_proj(y)

        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x.squeeze(1) if squeeze else x
