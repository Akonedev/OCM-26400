"""World model neuronal (JEPA) — réfute audit M12.

M12, EX-B205. World model : prédire l'état futur depuis l'état courant + action.
JEPA (Joint Embedding Predictive Architecture, LeCun) : prédit dans l'espace de
représentation (pas l'espace pixel), plus stable.

* WorldModel : encode(state)→représentation, predict(repr, action)→repr_suivante.
  L'erreur de prédiction = mesure de compréhension du monde (faible = bon modèle).
* train(trajectories) : apprend à prédire s(t+1) depuis s(t)+a(t).
* roll-out : prère plusieurs pas pour planifier.

C'est le 'modèle du monde' interne (le modèle simule l'effet de ses actions).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Tuple


class WorldModel(nn.Module):
    """JEPA-lite : encode l'état → prédit la représentation de l'état suivant (conditionné par l'action)."""

    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden: int = 32, repr_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, repr_dim))
        self.predictor = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, repr_dim))
        self.decoder = nn.Sequential(nn.Linear(repr_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, state_dim))

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def predict_next_repr(self, repr_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.predictor(torch.cat([repr_state, action], dim=-1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Prédit l'état suivant (dans l'espace état, via decode)."""
        r = self.encode(state)
        r_next = self.predict_next_repr(r, action)
        return self.decoder(r_next)

    def prediction_error(self, state: torch.Tensor, action: torch.Tensor,
                         next_state: torch.Tensor) -> torch.Tensor:
        """Erreur JEPA : prédit la REPRÉSENTATION de s(t+1), compare à encode(s(t+1))."""
        r_pred = self.predict_next_repr(self.encode(state), action)
        r_target = self.encode(next_state).detach()    # pas de gradient sur la cible (JEPA)
        return (r_pred - r_target).pow(2).mean()


def train_world_model(model: WorldModel, trajectories: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                      n_steps: int = 300, lr: float = 3e-3, seed: int = 0) -> list:
    """Entraîne le world model sur des transitions (state, action, next_state).
    Loss JEPA (prédiction de représentation) + reconstruction."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(n_steps):
        s, a, ns = trajectories[step % len(trajectories)]
        loss_pred = model.prediction_error(s.unsqueeze(0), a.unsqueeze(0), ns.unsqueeze(0))
        # + reconstruction (pour que l'espace repr reste décodable)
        recon = (model(s.unsqueeze(0), a.unsqueeze(0)) - ns.unsqueeze(0)).pow(2).mean()
        loss = loss_pred + 0.5 * recon
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            history.append((step, float(loss.item())))
    return history


def rollout(model: WorldModel, init_state: torch.Tensor, actions: List[torch.Tensor]
            ) -> List[torch.Tensor]:
    """Simule n pas : prédit les états successifs (roll-out pour planification)."""
    states = [init_state]
    s = init_state
    for a in actions:
        s = model(s.unsqueeze(0), a.unsqueeze(0))[0]
        states.append(s)
    return states


def demo() -> dict:
    """Démo : world model apprend une dynamique linéaire (s(t+1) = s(t) + a(t))."""
    torch.manual_seed(0)
    model = WorldModel(state_dim=4, action_dim=2)
    # génère des transitions (s + a = next_s)
    trajs = []
    for _ in range(20):
        s = torch.randn(4)
        a = torch.randn(2)
        # pad a à 4 dims pour l'addition (simule dynamique)
        ns = s + torch.cat([a, a])
        trajs.append((s, a, ns))
    hist = train_world_model(model, trajs, n_steps=300)
    # erreur de prédiction finale
    s, a, ns = trajs[0]
    err = float(model.prediction_error(s.unsqueeze(0), a.unsqueeze(0), ns.unsqueeze(0)))
    return {"initial_loss": round(hist[0][1], 4), "final_loss": round(hist[-1][1], 4),
            "final_pred_error": round(err, 4),
            "rollout_steps": len(rollout(model, torch.randn(4),
                                         [torch.randn(2) for _ in range(3)]))}


if __name__ == "__main__":
    import json
    print("[world_model] JEPA prédictif :", json.dumps(demo(), indent=2))
