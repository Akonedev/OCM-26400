"""Noyau SSM (state-space, Mamba/S6-lite) — capture-1-passe en profondeur (P-A).

Avancée techno (plan expert P-A) : remplacer le MLP résiduel de ReasonerBlock par un
noyau SÉQUENTIEL (state-space model, type Mamba-2 / S6) qui traite une CHAÎNE
compositionnelle (B,L,256) en UN seul forward — au lieu de dérouler lsra_loop étape par
étape. Rend « capture-une-passe » réel pour le raisonnement profond.

SSM (récurrence linéaire diagonal) :
    h_t = A ⊙ h_{t-1} + B(x_t)      # état latent séquentiel
    y_t = C(h_t) + D ⊙ x_t          # sortie + skip
A diagonal (stable, init négatif), B/C projections. Le « scan » accumule l'état sur L.

HONNÊTE : SSM-lite (récurrence linéaire diagonal), pas le Mamba-2 complet (sélection
input-dépendante, parallel scan) — mais un vrai modèle à état séquentiel, noyau drop-in
pour OmniModel.core, entraînable conjointement. La profondeur se traite dans l'état, pas
en empilant des couches.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .amv import D_MODEL


class SSMReasonerBlock(nn.Module):
    """Noyau SSM diagonal : traite (B,L,d) ou (B,d) en 1 forward (capture-1-passe)."""

    def __init__(self, d_model: int = D_MODEL, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_state)       # B(x)
        self.out_proj = nn.Linear(d_state, d_model)      # C(h)
        # A diagonal : stable (valeurs dans (0,1)), paramètre en log-space pour rester >0
        self.log_A = nn.Parameter(-torch.rand(d_state) - 0.5)   # A = exp(log_A) ∈ (0, ~0.6)
        self.D = nn.Parameter(torch.ones(d_model))               # skip (residual gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,d) -> (B,d) ; (B,L,d) -> (B,L,d). Récurrence SSM sur la dimension L."""
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1); squeeze = True            # (B,1,d)
        B, L, d = x.shape
        h0 = self.norm(x)
        z = self.in_proj(h0)                              # (B,L,d_state)
        A = torch.exp(self.log_A)                         # (d_state,) diagonal ∈ (0,1)
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(L):
            h = A * h + z[:, t]                           # état séquentiel (diagonal)
            outs.append(self.out_proj(h))                 # (B,d)
        y = torch.stack(outs, dim=1)                      # (B,L,d)
        y = y + self.D * x                                # skip / residual
        return y.squeeze(1) if squeeze else y


@torch.no_grad()
def ssm_final_state(blk: SSMReasonerBlock, seq: torch.Tensor) -> torch.Tensor:
    """Retourne l'état latent final après absorption de la séquence (preuve capture-1-passe)."""
    squeeze = False
    if seq.dim() == 2:
        seq = seq.unsqueeze(1); squeeze = True
    B, L, d = seq.shape
    z = blk.in_proj(blk.norm(seq))
    A = torch.exp(blk.log_A)
    h = torch.zeros(B, blk.d_state, device=seq.device, dtype=seq.dtype)
    for t in range(L):
        h = A * h + z[:, t]
    return h
