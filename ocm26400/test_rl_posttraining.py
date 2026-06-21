"""Tests RL post-training DPO/GRPO (OCM-26400) — audit M20."""
import torch
from ocm26400.rl_posttraining import (
    dpo_loss, grpo_loss, compute_group_advantages, demo_dpo, demo_grpo,
)


def test_dpo_loss_positive():
    pc = torch.tensor([-1.0]); pr = torch.tensor([-2.0])
    rc = torch.tensor([-1.0]); rr = torch.tensor([-2.0])
    loss = dpo_loss(pc, pr, rc, rr)
    assert loss.item() >= 0


def test_dpo_loss_lower_when_chosen_preferred():
    """Loss plus faible quand le policy préfère chosen."""
    pc = torch.tensor([-0.5]); pr = torch.tensor([-3.0])   # chosen >> rejected
    rc = torch.tensor([-0.5]); rr = torch.tensor([-3.0])
    loss_good = dpo_loss(pc, pr, rc, rr)
    pc_bad = torch.tensor([-3.0]); pr_bad = torch.tensor([-0.5])  # inversé
    loss_bad = dpo_loss(pc_bad, pr_bad, rc, rr)
    assert loss_good < loss_bad


def test_dpo_demo_aligns():
    r = demo_dpo()
    assert r["dpo_aligned"] is True


def test_grpo_advantages_centered():
    adv = compute_group_advantages([1.0, 0.0, -1.0])
    assert abs(sum(adv)) < 1e-6      # centré (somme ≈ 0)


def test_grpo_highest_reward_highest_advantage():
    r = demo_grpo()
    assert r["highest_reward_has_highest_advantage"] is True


def test_grpo_loss_finite():
    logps = torch.tensor([-1.0, -2.0, -1.5])
    adv = torch.tensor([1.0, -0.5, 0.0])
    loss = grpo_loss(logps, adv)
    assert torch.isfinite(loss)
