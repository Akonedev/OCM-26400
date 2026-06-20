"""Tests TDD — noyau SSM (Mamba-lite, capture-1-passe, P-A).

Valide : signature 2D/3D, différentiable, capture de l'état séquentiel, stabilité de A.
"""
import torch

from ocm26400.mamba_reasoner import SSMReasonerBlock, ssm_final_state


def test_ssm_2d_and_3d_shapes():
    blk = SSMReasonerBlock(d_model=64, d_state=8)
    x2 = torch.randn(4, 64)
    assert blk(x2).shape == (4, 64)                      # (B,d)->(B,d)
    x3 = torch.randn(4, 5, 64)
    assert blk(x3).shape == (4, 5, 64)                   # (B,L,d)->(B,L,d)


def test_ssm_differentiable():
    blk = SSMReasonerBlock(d_model=64, d_state=8)
    x = torch.randn(3, 7, 64)
    y = blk(x)
    y.sum().backward()
    assert blk.in_proj.weight.grad is not None
    assert blk.log_A.grad is not None and blk.log_A.grad.abs().sum() > 0


def test_ssm_A_is_stable():
    """A diagonal dans (0,1) -> récurrence stable (pas d'explosion)."""
    blk = SSMReasonerBlock(d_model=64, d_state=8)
    A = torch.exp(blk.log_A)
    assert (A > 0).all() and (A < 1).all()


def test_ssm_captures_sequence_in_state():
    """L'état final diffère pour des séquences différentes (capture-1-passe)."""
    blk = SSMReasonerBlock(d_model=64, d_state=8)
    s1 = torch.randn(1, 6, 64); s2 = torch.randn(1, 6, 64)
    h1 = ssm_final_state(blk, s1); h2 = ssm_final_state(blk, s2)
    assert not torch.allclose(h1, h2)


def test_ssm_drop_in_as_omni_core():
    """Le SSM est utilisable comme noyau (même interface que ReasonerBlock.forward)."""
    blk = SSMReasonerBlock(d_model=256)
    x = torch.randn(2, 256)                              # un AMV
    out = blk(x)
    assert out.shape == (2, 256)
