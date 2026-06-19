"""Tests RED — InfoNCE core math (P1 juge). SANITY math, PAS de crown-jewel N=11 vacuous.

Spec Besoins_Maths.md §2.4: L_consist = InfoNCE cross-modal.
  L = -(1/N) Σ_i log[ exp(v_i·u_i/τ) / Σ_j exp(v_i·u_j/τ) ]
Implémentation stable via F.cross_entropy (logsumexp interne = max-subtraction).
"""
import math
import pytest
import torch
from ocm26400.infonce import info_nce, info_nce_symmetric, multimodal_l_consist, TAU_DEFAULT


def test_info_nce_zero_when_aligned():
    # z_a == z_b orthonormés (eye) + tau petit -> loss ~ 0 (alignement parfait)
    z = torch.eye(8)
    loss = info_nce(z, z, tau=0.05)
    assert loss.item() < 1e-3


def test_info_nce_positive_when_random():
    # z_a, z_b aléatoires -> loss ~ log(N) (softmax quasi-uniforme)
    torch.manual_seed(0)
    z_a = torch.randn(8, 256)
    z_b = torch.randn(8, 256)
    loss = info_nce(z_a, z_b, tau=1.0)
    assert math.log(8) - 1.0 < loss.item() < math.log(8) + 1.0


def test_symmetric_average_of_directions():
    torch.manual_seed(1)
    z_a = torch.randn(6, 32)
    z_b = torch.randn(6, 32)
    sym = info_nce_symmetric(z_a, z_b, tau=0.1)
    expected = 0.5 * (info_nce(z_a, z_b, 0.1) + info_nce(z_b, z_a, 0.1))
    assert torch.allclose(sym, expected, atol=1e-6)


def test_multimodal_three_modalities_aggregates_pairs():
    # 3 modalités -> moyenne symétrique sur les C(3,2)=3 paires
    torch.manual_seed(2)
    mods = [torch.randn(5, 16) for _ in range(3)]
    loss = multimodal_l_consist(mods, tau=0.1)
    expected = (info_nce_symmetric(mods[0], mods[1], 0.1)
                + info_nce_symmetric(mods[0], mods[2], 0.1)
                + info_nce_symmetric(mods[1], mods[2], 0.1)) / 3.0
    assert torch.allclose(loss, expected, atol=1e-6)


def test_differentiable():
    z_a = torch.randn(4, 8, requires_grad=True)
    z_b = torch.randn(4, 8, requires_grad=True)
    loss = info_nce(z_a, z_b, tau=0.07)
    loss.backward()
    assert z_a.grad is not None and z_b.grad is not None


def test_no_nan_on_large_logits():
    # tau minuscule -> grands logits -> pas de NaN (preuve stabilité logsumexp)
    z_a = torch.randn(8, 16) * 10
    z_b = torch.randn(8, 16) * 10
    loss = info_nce(z_a, z_b, tau=1e-6)
    assert not torch.isnan(loss).item()


def test_acsp_loss_accepts_consist_term():
    # acsp_loss(..., consist_term=t) ajoute delta*term ; sans consist_term -> inchangé
    from ocm26400.amv import AMVVector
    from ocm26400.verifier import SymbolicDict, Verifier
    from ocm26400.acsp import acsp_loss, DELTA
    d = SymbolicDict(); ver = Verifier(d)
    v = AMVVector(torch.zeros(256)); v.ent.copy_(d.canonical(2)); v.prop.copy_(d.canonical(5))
    base = acsp_loss(v, d, ver, d_ent=2, d_prop=5, op_id=0)
    with_c = acsp_loss(v, d, ver, d_ent=2, d_prop=5, op_id=0, consist_term=torch.tensor(2.0))
    assert torch.allclose(with_c, base + DELTA * 2.0, atol=1e-6)
