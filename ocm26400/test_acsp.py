"""Tests RED — ACSP loss (Amodal Consistency & Step Penalty), spec §2.

L = alpha*L_align + beta*L_step + gamma*L_sparse + delta*L_consist

  L_align  : (1) cosinus -> nearest primitive du dictionnaire. 0 si v EST un primitif.
  L_step   : 0 si étape légale (V=1), P_backtrack (> > 1) si illégale.
  L_sparse : L1 sur v (empeche la mémorisation de bruit dans les 256 dims).
  L_consist: InfoNCE cross-modal (single-modality ici -> 0).

Sanity check spec §4.4: la loss doit pouvoir descendre à ~0 sur un batch "parfait"
(v=primitif canonique + étape légale).
"""
import pytest
import torch
from ocm26400.amv import AMVVector, PART
from ocm26400.verifier import SymbolicDict, Verifier, P_BACKTRACK
from ocm26400.acsp import acsp_loss, l_align, l_step, l_sparse


@pytest.fixture
def dict_ver():
    d = SymbolicDict()
    return d, Verifier(d)


def test_l_align_zero_when_v_is_canonical_primitive(dict_ver):
    d, _ = dict_ver
    v = AMVVector(torch.zeros(256))
    v.ent.copy_(d.canonical(3))  # ent = primitif canonique 3
    assert l_align(v, d).item() == pytest.approx(0.0, abs=1e-5)


def test_l_align_positive_when_v_is_random(dict_ver):
    d, _ = dict_ver
    v = AMVVector(torch.randn(256) * 0.1)
    assert l_align(v, d).item() > 0.0


def test_l_step_zero_for_legal_operation(dict_ver):
    _, v = dict_ver
    assert l_step(v, d_ent=2, d_prop=5, op_id=0).item() == 0.0


def test_l_step_backtrack_penalty_for_illegal(dict_ver):
    _, v = dict_ver
    loss = l_step(v, d_ent=999, d_prop=5, op_id=0)  # ent hors dictionnaire
    assert loss.item() == pytest.approx(P_BACKTRACK)


def test_l_sparse_is_l1_norm_nonneg(dict_ver):
    _, _ = dict_ver
    v = AMVVector(torch.tensor([1.0] * 256))
    assert l_sparse(v).item() == pytest.approx(256.0)


def test_full_acsp_near_zero_on_perfect_trajectory(dict_ver):
    """SANITY CHECK spec: loss ~ 0 sur batch parfait (canonique + légal)."""
    d, ver = dict_ver
    v = AMVVector(torch.zeros(256))
    v.ent.copy_(d.canonical(2))   # ent = primitif valide
    v.prop.copy_(d.canonical(5))  # prop = primitif valide
    loss = acsp_loss(v, d, ver, d_ent=2, d_prop=5, op_id=0)
    # align~0 + step légal~0 + sparse faible (L1 des one-hots) -> loss petite
    assert loss.item() < 0.05


def test_acsp_is_differentiable(dict_ver):
    """Le gradient doit remonter au tenseur (pour l'entraînement)."""
    d, ver = dict_ver
    raw = torch.zeros(256, requires_grad=True)
    v = AMVVector(raw)
    loss = acsp_loss(v, d, ver, d_ent=2, d_prop=5, op_id=0)
    loss.backward()
    assert raw.grad is not None
    assert raw.grad.shape == (256,)


def test_acsp_dominated_by_step_penalty_when_illegal(dict_ver):
    """Une étape illégale doit dominer (P_backtrack >> reste)."""
    d, ver = dict_ver
    v = AMVVector(torch.zeros(256))
    v.ent.copy_(d.canonical(2))
    loss = acsp_loss(v, d, ver, d_ent=999, d_prop=5, op_id=0)
    assert loss.item() >= P_BACKTRACK  # la pénalité d'étape domine
