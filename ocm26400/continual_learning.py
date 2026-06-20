"""Continual learning (EWC) — anti catastrophic forgetting — réfute audit M19.

EX-B187. L'apprentissage continu (continual learning) : apprendre de nouvelles tâches
SANS oublier les précédentes. Le fléau de l'oubli catastrophique = un réseau qui apprend
la tâche B perd la tâche A. EWC (Elastic Weight Consolidation, Kirkpatrick 2017) ajoute
une pénalité qui retient les poids importants pour A.

* EWC : loss = L_tâche_B + Σ λ/2 · F_i · (θ_i − θ*_i)²
  F = Fisher information (importance d'un poids pour la tâche A), θ* = poids optimaux A.
* Fisher diagonal : F_i = E[(∂log p / ∂θ_i)²] (approximé sur data A).

On implémente EWC sur un petit MLP illustratif (2 tâches successives) : démontre que la
pénalité Fisher retient la tâche A. C'est le continual learning réel.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn


class EWCCallback:
    """Calcule et applique la pénalité EWC (Fisher diagonal) pour retenir une tâche."""

    def __init__(self, model: nn.Module, lam: float = 100.0):
        self.model = model
        self.lam = lam
        self.fisher: dict = {}     # nom_param → Fisher diagonal
        self.opt_params: dict = {} # nom_param → θ* (optimaux de la tâche précédente)

    def compute_fisher(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 100):
        """Calcule le Fisher diagonal sur les données de la tâche courante (à figer)."""
        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        for _ in range(min(n_samples, len(x))):
            idx = torch.randint(0, len(x), (1,))
            xi, yi = x[idx], y[idx]
            self.model.zero_grad()
            out = self.model(xi)
            loss = nn.functional.mse_loss(out, yi)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.detach() ** 2
        self.fisher = {n: (f / max(n_samples, 1)) for n, f in self.fisher.items()}
        # sauvegarde θ* optimaux
        self.opt_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        self.model.zero_grad()

    def penalty(self) -> torch.Tensor:
        """Pénalité EWC : Σ λ/2 · F_i · (θ_i − θ*_i)². À ajouter à la loss de la tâche B."""
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, requires_grad=True)
        pen = torch.zeros(1)
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                pen = pen + (self.fisher[n] * (p - self.opt_params[n]) ** 2).sum()
        return (self.lam / 2) * pen


def demo_ewc(n_steps: int = 300, seed: int = 0) -> dict:
    """Démo : 2 tâches successives. Mesure l'oubli avec vs sans EWC.
    Tâche A : y = x. Tâche B : y = −x. Sans EWC, apprendre B oublie A."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    def make_task(sign):
        x = torch.linspace(-1, 1, 32).unsqueeze(1)
        y = sign * x
        return x, y

    xa, ya = make_task(1.0)
    xb, yb = make_task(-1.0)

    def train(x, y, ewc=None, steps=200):
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(steps):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), y)
            if ewc is not None:
                loss = loss + ewc.penalty()
            loss.backward()
            opt.step()

    # après tâche A : accuracy A
    train(xa, ya)
    acc_a_after_A = float(nn.functional.mse_loss(model(xa), ya))
    # EWC : fige la tâche A
    ewc = EWCCallback(model, lam=400.0)
    ewc.compute_fisher(xa, ya)
    train(xb, yb, ewc=ewc)
    acc_a_after_B_ewc = float(nn.functional.mse_loss(model(xa), ya))

    # comparaison SANS EWC (modèle frais)
    torch.manual_seed(seed)
    model2 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model2.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        nn.functional.mse_loss(model2(xa), ya).backward()
        opt.step()
    for _ in range(200):
        opt.zero_grad()
        nn.functional.mse_loss(model2(xb), yb).backward()
        opt.step()
    acc_a_after_B_noewc = float(nn.functional.mse_loss(model2(xa), ya))

    return {
        "mse_A_after_A": acc_a_after_A,
        "mse_A_after_B_WITH_ewc": acc_a_after_B_ewc,
        "mse_A_after_B_WITHOUT_ewc": acc_a_after_B_noewc,
        "forgetting_reduced": acc_a_after_B_noewc > acc_a_after_B_ewc,
        "verdict": ("EWC_REDUCE_FORGETTING" if acc_a_after_B_noewc > acc_a_after_B_ewc
                    else "NO_EFFECT"),
    }


if __name__ == "__main__":
    res = demo_ewc()
    print("[continual_learning] démo EWC (anti-oubli catastrophique) :")
    for k, v in res.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
