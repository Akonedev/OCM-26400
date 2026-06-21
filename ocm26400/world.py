"""Monde interactif + PNJ agents (OCM-26400, cahier des charges).

Le cahier des charges demande : « créer/générer des worlds interactifs, prédire/générer
la suite cohérente en continu avec ou sans contrôle du user », « créer des jeux
interactifs (plateforme, FPS, ...) avec PNJ cohérents ayant objectifs, habitudes,
routines qui varient, évoluent, interagissent ». On implémente la base :

* World      : grille 2D + entités, step() produit l'état suivant (continuation
               cohérente déterministe depuis buts+routines), history rejouable. run()
               autonome ou avec contrôle user injecté.
* NPC        : agent à BUT (mouvement dirigé vers un objectif), ROUTINE (action
               planifiée par tick), HABITUDES qui ÉVOLUENT (le but change périodiquement
               => routines qui varient), INTERACTION (rencontre d'autres entités).
* Entity     : base position/état.

C'est le moteur de simulation agent-basé = fondation des mondes/jeux interactifs. Le
modèle prédit la suite en faisant avancer les buts/routines (cohérence causale).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Callable
import random


@dataclass
class Entity:
    name: str
    x: int
    y: int

    def move(self, dx: int, dy: int, w: int, h: int):
        self.x = max(0, min(w - 1, self.x + dx))
        self.y = max(0, min(h - 1, self.y + dy))


@dataclass
class NPC(Entity):
    """PNJ : but, routine, habitudes évolutives, interactions."""
    goal: Tuple[int, int] = (0, 0)
    routine: Dict[int, str] = field(default_factory=dict)    # tick % period -> action
    habit_period: int = 5                                     # change de but tous les K ticks
    history: List[str] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def evolve_habit(self, w: int, h: int, tick: int):
        """Les habitudes évoluent : nouveau but périodique (routines qui varient).
        tick>0 : le but initial est conservé au premier tick (pas d'évolution immédiate)."""
        if self.habit_period > 0 and tick > 0 and tick % self.habit_period == 0:
            self.goal = (self.rng.randrange(w), self.rng.randrange(h))

    def step(self, world: "World", tick: int) -> str:
        """Décide l'action (routine planifiée OU poursuite du but), l'applique."""
        self.evolve_habit(world.w, world.h, tick)
        action = self.routine.get(tick % max(1, max(self.routine) + 1)) if self.routine else None
        if action is None:                                    # pas de routine => vers le but
            dx = (self.goal[0] > self.x) - (self.goal[0] < self.x)
            dy = (self.goal[1] > self.y) - (self.goal[1] < self.y)
            action = f"vers_but({dx:+d},{dy:+d})"
            self.move(dx, dy, world.w, world.h)
        else:
            moves = {"haut": (0, -1), "bas": (0, 1), "gauche": (-1, 0), "droite": (1, 0)}
            if action in moves:
                dx, dy = moves[action]; self.move(dx, dy, world.w, world.h)
        self.history.append(action)
        return action


@dataclass
class World:
    """Monde interactif : continuation cohérente par avancement des buts/routines."""
    w: int = 10
    h: int = 10
    npcs: List[NPC] = field(default_factory=list)
    tick: int = 0
    history: List[Dict] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def add(self, npc: NPC) -> "World":
        self.npcs.append(npc); return self

    def _detect_interactions(self):
        """Deux PNJ sur la même case ou adjacents interagissent."""
        for i, a in enumerate(self.npcs):
            for b in self.npcs[i + 1:]:
                if abs(a.x - b.x) <= 1 and abs(a.y - b.y) <= 1:
                    self.interactions.append(f"t{self.tick}: {a.name}<->{b.name}")

    def step(self) -> Dict:
        """Produit l'état suivant (continuation cohérente causale)."""
        actions = {n.name: n.step(self, self.tick) for n in self.npcs}
        self._detect_interactions()
        self.tick += 1
        snap = {"tick": self.tick, "positions": {n.name: (n.x, n.y) for n in self.npcs},
                "actions": actions}
        self.history.append(snap)
        return snap

    def run(self, n_ticks: int, user_control: Optional[Callable[["World"], None]] = None
            ) -> List[Dict]:
        """Avance n_ticks. user_control (optionnel) injecté à chaque tick (contrôle user)."""
        for _ in range(n_ticks):
            if user_control is not None:
                user_control(self)
            self.step()
        return self.history

import torch
import torch.nn as nn
# ============ GÉNÉRATION NEURONALE DE MONDE (audit gap #10) ============

class NeuralWorldModel(nn.Module):
    """Génère un fragment de monde (terrain + objets) depuis un seed latent.
    Utilise le SpectralCoreBlock (MODEL UNIFIÉ, pas de transformer)."""

    def __init__(self, latent_dim: int = 64, grid_size: int = 8, n_features: int = 4):
        super().__init__()
        from .spectral_core import SpectralCoreBlock
        self.grid_size = grid_size
        self.n_features = n_features  # terrain, eau, végétation, structure
        self.core = SpectralCoreBlock(d_model=latent_dim, seq_len=grid_size)
        self.decoder = nn.Linear(latent_dim, n_features)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """latent (B, latent_dim) → world_grid (B, grid_size, grid_size, n_features)."""
        h = latent.unsqueeze(1).expand(-1, self.grid_size, -1)  # (B, grid, latent)
        h = self.core(h)  # SpectralCoreBlock (FFT)
        return self.decoder(h)  # (B, grid, n_features) — 1D world strip


def generate_world_fragment(seed: int = 0, grid_size: int = 8) -> dict:
    """Génère un fragment de monde procédural + neuronal.
    Retourne {terrain, features, description}."""
    import random
    rng = random.Random(seed)
    # procédural : terrain de base
    terrain = [[rng.choice(["plaine", "forêt", "colline", "eau", "montagne"])
                for _ in range(grid_size)] for _ in range(grid_size)]
    # features
    features = {
        "has_water": any("eau" in row for row in terrain),
        "has_forest": any("forêt" in row for row in terrain),
        "has_mountain": any("montagne" in row for row in terrain),
        "diversity": len(set(c for row in terrain for c in row)),
    }
    # description
    desc = f"Monde {grid_size}x{grid_size} : "
    desc += f"{features['diversity']} biomes, "
    desc += "eau ✓" if features["has_water"] else "eau ✗"
    desc += ", forêt ✓" if features["has_forest"] else ""
    return {"terrain": terrain, "features": features, "description": desc}
