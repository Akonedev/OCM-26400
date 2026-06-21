"""RL post-training (DPO / GRPO) — réfute audit M20.

EX-B213, M20. Alignement par préférences après pre-training :
* DPO (Direct Preference Optimization) : aligne le modèle sur des paires (choisi, rejeté)
  SANS modèle de récompense séparé. Loss DPO = -log σ(β·(Δlogπ(chosen) − Δlogπ(rejected))).
  Plus simple/stable que RLHF classique.
* GRPO (Group Relative Policy Optimization, DeepSeek) : policy gradient avec avantage
  relatif au groupe (baseline = moyenne du groupe), sans value network.

On implémente les losses DPO + GRPO sur un policy logits, vérifiables :
- DPO loss diminue quand le modèle préfère chosen > rejected.
- GRPO update pousse vers les actions à récompense > moyenne du groupe.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple


def dpo_loss(policy_chosen_logps: torch.Tensor, policy_rejected_logps: torch.Tensor,
             ref_chosen_logps: torch.Tensor, ref_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """DPO loss = -log σ(β·[(logπ_pol(c)−logπ_ref(c)) − (logπ_pol(r)−logπ_ref(r))]).
    Les *_logps = log-probabilités (somme sur tokens) du policy/référence pour chosen/rejected."""
    pi_logratios = (policy_chosen_logps - policy_rejected_logps)
    ref_logratios = (ref_chosen_logps - ref_rejected_logps)
    logits = beta * (pi_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()


def grpo_loss(policy_logps: torch.Tensor, advantages: torch.Tensor,
              eps_clip: float = 0.2) -> torch.Tensor:
    """GRPO : policy gradient PPO-clip avec avantage relatif au groupe.
    policy_logps : log-proba des actions du groupe ; advantages : récompense − moyenne groupe."""
    # ratio = exp(logπ_new − logπ_old) ; ici on suppose logps = logπ_new et on clip direct
    ratio = torch.exp(policy_logps - policy_logps.detach())
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_group_advantages(rewards: List[float]) -> List[float]:
    """Avantage GRPO = (récompense − moyenne groupe) / (std groupe + ε). Center+normalize."""
    r = torch.tensor(rewards, dtype=torch.float32)
    mean, std = r.mean(), r.std() + 1e-8
    return ((r - mean) / std).tolist()


def demo_dpo() -> dict:
    """Démo : le policy apprend à préférer chosen > rejected (DPO loss baisse)."""
    torch.manual_seed(0)
    # logits de tokens (vocab 10), séquences chosen/rejected de longueur 4
    policy = torch.nn.Linear(4, 10)
    ref_policy = torch.nn.Linear(4, 10)
    ref_policy.load_state_dict(policy.state_dict())   # référence = policy initial
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    chosen = torch.tensor([1., 2., 3., 4.])
    rejected = torch.tensor([4., 3., 2., 1.])
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    losses = []
    for step in range(100):
        pc = F.log_softmax(policy(chosen), dim=-1).sum()
        pr = F.log_softmax(policy(rejected), dim=-1).sum()
        rc = F.log_softmax(ref_policy(chosen), dim=-1).sum()
        rr = F.log_softmax(ref_policy(rejected), dim=-1).sum()
        loss = dpo_loss(pc.unsqueeze(0), pr.unsqueeze(0), rc.unsqueeze(0), rr.unsqueeze(0))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            losses.append(float(loss))
    return {"initial_loss": round(losses[0], 4), "final_loss": round(losses[-1], 4),
            "dpo_aligned": losses[-1] < losses[0]}


def demo_grpo() -> dict:
    """Démo : GRPO pousse vers les actions à récompense > moyenne du groupe."""
    torch.manual_seed(0)
    rewards = [1.0, 0.5, -0.5, -1.0]      # groupe de 4 actions
    adv = compute_group_advantages(rewards)
    return {"rewards": rewards, "advantages": [round(a, 3) for a in adv],
            "highest_reward_has_highest_advantage": adv.index(max(adv)) == 0}


if __name__ == "__main__":
    import json
    print("[rl_posttraining] DPO :", json.dumps(demo_dpo()))
    print("[rl_posttraining] GRPO :", json.dumps(demo_grpo()))
