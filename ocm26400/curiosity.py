"""Curiosité / exploration active — réfute audit M21.

EX-B173/185, M21. Récompense INTRINSÈQUE : le modèle est curieux (explore ce qui est
nouveau/incertain) au-delà de la récompense externe. C'est le moteur de l'apprentissage
autonome (« bonheur intrinsèque » du cahier des charges).

* CuriosityDrive : récompense intrinsèque = prédiction d'erreur (ICM-like) ou nouveauté.
  Un état SURPRENANT (prédiction erronée) → haute curiosité → on l'explore.
* NoveltyMemory : mémorise les états vus ; nouveauté = distance à ce qui est connu.
* select_curious(candidates) : choisit l'option la + informative (curiosité).

Vérifiable : un état nouveau attire + de curiosité qu'un état familier.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import torch


@dataclass
class NoveltyMemory:
    """Mémorise les états vus (embedding) → nouveauté = distance au + proche connu."""
    seen: List[torch.Tensor] = field(default_factory=list)

    def observe(self, state: torch.Tensor) -> None:
        self.seen.append(state.detach().flatten())

    def novelty(self, state: torch.Tensor) -> float:
        """Nouveauté d'un état : distance cosinus au + proche état connu (0=familier, 1=tout neuf)."""
        if not self.seen:
            return 1.0
        v = state.detach().flatten()
        nv = v.norm() + 1e-8
        best = 0.0
        for s in self.seen:
            sn = s.norm() + 1e-8
            sim = float((v @ s) / (nv * sn))
            best = max(best, sim)
        return 1.0 - best           # 1 - similarité_max = nouveauté


class CuriosityDrive:
    """Récompense intrinsèque (curiosité) : prédiction d'erreur + nouveauté.

    * predict(state) → predicted_next ; l'erreur de prédiction = surprise = curiosité.
    * intrinsic_reward(state) : haut si l'état est surprenant/nouveau.
    """

    def __init__(self, dim: int = 64, novelty_weight: float = 0.5):
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(dim, dim), torch.nn.ReLU(), torch.nn.Linear(dim, dim))
        self.memory = NoveltyMemory()
        self.novelty_weight = novelty_weight

    def prediction_error(self, state: torch.Tensor, next_state: torch.Tensor) -> float:
        """Erreur du prédicteur forward (ICM) : surprise = curiosité."""
        pred = self.predictor(state.flatten().float())
        return float((pred - next_state.flatten().float()).pow(2).mean())

    def intrinsic_reward(self, state: torch.Tensor, next_state: torch.Tensor = None) -> float:
        """Récompense intrinsèque : nouveauté (+ erreur de prédiction si next fourni)."""
        nov = self.memory.novelty(state)
        err = self.prediction_error(state, next_state) if next_state is not None else 0.0
        return self.novelty_weight * nov + (1 - self.novelty_weight) * min(err, 1.0)

    def observe(self, state: torch.Tensor) -> None:
        self.memory.observe(state)


def select_curious(drive: CuriosityDrive, candidates: List[torch.Tensor]
                   ) -> Tuple[int, float]:
    """Sélectionne l'index du candidat le + curieux (informatif)."""
    rewards = [(i, drive.memory.novelty(c)) for i, c in enumerate(candidates)]
    rewards.sort(key=lambda x: -x[1])
    return rewards[0]


if __name__ == "__main__":
    drive = CuriosityDrive(dim=8)
    # états familiers
    for _ in range(3):
        s = torch.ones(8) * 0.5
        drive.observe(s)
    familiar = torch.ones(8) * 0.5
    novel = torch.randn(8)
    print(f"[curiosity] état familier : nouveauté={drive.memory.novelty(familiar):.3f}")
    print(f"[curiosity] état NOUVEAU  : nouveauté={drive.memory.novelty(novel):.3f}")
    idx, score = select_curious(drive, [familiar, novel])
    print(f"[curiosity] candidat choisi : index={idx} (curiosité={score:.3f}) → le NOUVEAU")
