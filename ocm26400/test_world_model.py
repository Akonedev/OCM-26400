"""Tests world model JEPA (OCM-26400) — audit M12."""
import torch
from ocm26400.world_model import WorldModel, train_world_model, rollout, demo


def test_world_model_forward():
    m = WorldModel(state_dim=4, action_dim=2)
    s = torch.randn(4); a = torch.randn(2)
    out = m(s.unsqueeze(0), a.unsqueeze(0))
    assert out.shape[-1] == 4


def test_prediction_error_decreases():
    """L'entraînement réduit l'erreur de prédiction JEPA."""
    torch.manual_seed(0)
    m = WorldModel(state_dim=4, action_dim=2)
    trajs = []
    for _ in range(20):
        s = torch.randn(4); a = torch.randn(2)
        trajs.append((s, a, s + torch.cat([a, a])))
    hist = train_world_model(m, trajs, n_steps=300)
    assert hist[-1][1] < hist[0][1]      # loss baisse


def test_rollout_length():
    m = WorldModel(state_dim=4, action_dim=2)
    states = rollout(m, torch.randn(4), [torch.randn(2) for _ in range(3)])
    assert len(states) == 4               # init + 3 pas


def test_encode_decode_dim():
    m = WorldModel(state_dim=4, action_dim=2, repr_dim=16)
    r = m.encode(torch.randn(1, 4))
    assert r.shape[-1] == 16
    assert m.decoder(r).shape[-1] == 4


def test_demo_runs():
    d = demo()
    assert d["final_loss"] < d["initial_loss"]
    assert d["rollout_steps"] == 4
