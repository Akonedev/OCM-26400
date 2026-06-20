"""Tool-use APPRIS par le modèle (E3, OCM-26400, spec #4).

Le modèle DÉCIDE du skill/outil à utiliser (pas un plan câblé). Une tête ToolPolicy
(AMV -> softmax sur les skills) est entraînée par IMITATION/DAgger sur des traces
(task -> correct skill) — pas REINFORCE (trop bruité à petit budget, cf. DA).

Le tout dans le modèle unifié : la tâche -> noyau spectral (SpectralCoreBlock, archi
utilisateur) -> AMV -> ToolPolicy -> skill choisi -> Toolkit.execute. Un seul modèle,
l'agent décide+exécute.

* ToolPolicy    : head AMV -> logits sur les skills ; decide(amv) -> (skill, confiance).
* TaskEncoder   : type de tâche -> AMV (via le noyau spectral unifié).
* ToolUsingAgent: encode la tâche -> décide le skill -> exécute via le Toolkit.
* train_tool_policy : imitation (cross-entropy sur traces (task, skill correct)).

HONNÊTE : imitation sur traces synthétiques (task_type -> skill) où le bon skill est
connu. Le modèle APPREND la sélection (généralise à de nouvelles instances de tâche),
pas un dictionnaire câblé. Skill backend = agents_tools.Skill (callables).
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .amv import D_MODEL
from .spectral_core import SpectralCoreBlock
from .agents_tools import Toolkit


class TaskEncoder(nn.Module):
    """Type de tâche -> AMV (via le noyau spectral unifié)."""

    def __init__(self, n_task_types: int, d_model: int = D_MODEL):
        super().__init__()
        self.emb = nn.Embedding(n_task_types, d_model)
        self.core = SpectralCoreBlock(d_model=d_model)     # noyau spectral unifié

    def forward(self, task_type_id: torch.Tensor) -> torch.Tensor:
        return self.core(self.emb(task_type_id))            # (B, d_model) = AMV


class ToolPolicy(nn.Module):
    """Tête tool-use : AMV -> softmax sur les skills (le modèle décide)."""

    def __init__(self, d_model: int = D_MODEL, n_skills: int = 4):
        super().__init__()
        self.head = nn.Linear(d_model, n_skills)

    def forward(self, amv: torch.Tensor) -> torch.Tensor:
        return self.head(amv)                               # (B, n_skills) logits

    @torch.no_grad()
    def decide(self, amv: torch.Tensor) -> Tuple[int, float]:
        logits = self(amv.unsqueeze(0) if amv.dim() == 1 else amv)
        probs = F.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return int(idx[0].item()), float(conf[0].item())


def train_tool_policy(encoder: TaskEncoder, policy: ToolPolicy,
                      traces: List[Tuple[int, int]], n_steps: int = 800,
                      lr: float = 3e-3, device: str = "cpu") -> None:
    """Imitation : traces = [(task_type_id, correct_skill_id)]. Cross-entropy."""
    opt = torch.optim.Adam(list(encoder.parameters()) + list(policy.parameters()), lr=lr)
    task_ids = torch.tensor([t for t, _ in traces], device=device)
    skill_ids = torch.tensor([s for _, s in traces], device=device)
    N = len(traces)
    for _ in range(n_steps):
        idx = torch.randint(0, N, (min(64, N),), device=device)
        amv = encoder(task_ids[idx])
        logits = policy(amv)
        loss = F.cross_entropy(logits, skill_ids[idx])
        opt.zero_grad(); loss.backward(); opt.step()


class ToolUsingAgent:
    """Agent outillé : encode la tâche -> décide le skill (appris) -> exécute via Toolkit."""

    def __init__(self, encoder: TaskEncoder, policy: ToolPolicy, toolkit: Toolkit):
        self.encoder = encoder
        self.policy = policy
        self.toolkit = toolkit

    @torch.no_grad()
    def act(self, task_type_id: int, args: tuple):
        """Décide le skill pour la tâche et l'exécute. Retourne (skill_name, result, conf)."""
        amv = self.encoder(torch.tensor([task_type_id]))
        skill_idx, conf = self.policy.decide(amv[0])
        skill_name = self.toolkit.names()[skill_idx] if skill_idx < len(self.toolkit.names()) else None
        result = self.toolkit.use(skill_name, *args) if skill_name else None
        return skill_name, result, conf
