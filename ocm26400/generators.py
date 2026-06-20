"""Décodeur GÉNÉRATIF diffusion-lite (flow matching) — vraie génération de signal (E5).

Remplace la 'génération' MSE-régression-de-features (omni.py audio_dec/image_dec Linear)
par un VRAI décodeur génératif : FLOW MATCHING (Lipman 2023, Esser SD3). Conditionné par
l'AMV (l'espace unifié), il génère un signal (image/audio) en intégrant un champ de
vitesse appris du bruit vers la donnée.

Flow matching : x_t = (1-t)·bruit + t·x_cible ; on apprend v(x_t, t, AMV) ≈ x_cible -
bruit. Échantillonnage : intégration d'Euler du bruit vers la donnée en 'steps' pas.

SCOPE STRICT (DA-5 compute) : tiny MLP, digits 8x8 / features audio 32-d, quelques steps.
PAS ImageNet/AudioNet (compute/data massifs, hors scope 'ne pas saturer'). C'est une
preuve de génération neurale réelle (pas MSE), pas un SOTA HiFi.

* AMVConditionedDecoder(x_dim, cond_dim=256) : VelocityNet (x+t+AMV -> vélocité).
* flow_match_loss(amv, x) : entraîne le champ de vélocité.
* sample(amv, steps=8) : GÉNÈRE un signal depuis le bruit.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .amv import D_MODEL


class AMVConditionedDecoder(nn.Module):
    """Décodeur flow-matching conditionné par l'AMV (génération de signal)."""

    def __init__(self, x_dim: int, cond_dim: int = D_MODEL, hidden: int = 128):
        super().__init__()
        self.x_dim = x_dim
        # VelocityNet : entrée (x_t, t, AMV) -> vélocité (x_dim)
        self.net = nn.Sequential(
            nn.Linear(x_dim + 1 + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def _velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)                                  # (B,1)
        inp = torch.cat([x_t, t, cond], dim=-1)             # (B, x_dim+1+cond_dim)
        return self.net(inp)                                # (B, x_dim)

    def flow_match_loss(self, cond: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Loss flow matching : v(x_t,t,cond) ≈ x_target - bruit (cible linéaire)."""
        B = x_target.shape[0]
        noise = torch.randn_like(x_target)
        t = torch.rand(B, device=x_target.device)
        x_t = (1.0 - t).view(-1, 1) * noise + t.view(-1, 1) * x_target   # interpolation
        target_vel = x_target - noise
        pred_vel = self._velocity(x_t, t, cond)
        return ((pred_vel - target_vel) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """GÉNÈRE un signal : intègre du bruit vers la donnée en 'steps' pas (Euler)."""
        B = cond.shape[0]
        x = torch.randn(B, self.x_dim, device=cond.device)   # départ : bruit
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=cond.device)
            x = x + self._velocity(x, t, cond) * dt          # Euler
        return x                                             # (B, x_dim) signal généré
