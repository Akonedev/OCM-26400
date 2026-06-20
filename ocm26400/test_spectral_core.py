"""Tests TDD — noyau spectral unifié (l'architecture de l'utilisateur, OCM-26400).

Valide : SpectralCoreBlock (2D/3D, différentiable) ET qu'il est le noyau PAR DÉFAUT de
l'OmniModel unifié (l'architecture du projet, pas un MLP/SSM).
"""
import torch

from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.omni import OmniModel


def test_spectral_core_2d_3d():
    blk = SpectralCoreBlock(d_model=64, seq_len=16)
    assert blk(torch.randn(4, 64)).shape == (4, 64)           # un AMV (L=1)
    assert blk(torch.randn(4, 8, 64)).shape == (4, 8, 64)     # chaîne (L=8)


def test_spectral_core_differentiable():
    blk = SpectralCoreBlock(d_model=64, seq_len=16)
    y = blk(torch.randn(2, 5, 64))
    y.sum().backward()
    assert blk.filter_real.grad is not None and blk.filter_real.grad.abs().sum() > 0
    assert blk.in_proj.weight.grad is not None


def test_omni_uses_spectral_core_by_default():
    """L'OmniModel unifié utilise le NOYAU SPECTRAL de l'utilisateur par défaut."""
    m = OmniModel()
    assert isinstance(m.core, SpectralCoreBlock)
    assert m.core_type == "spectral"


def test_spectral_omni_classify_and_generate():
    """Le modèle unifié spectral classifie et génère (noyau spectral partagé)."""
    m = OmniModel(core_type="spectral")
    assert m.classify("audio", torch.randn(2, 1200)).shape == (2, 5)
    assert m.classify("image", torch.randn(2, 1, 8, 8)).shape == (2, 10)
    assert m.generate("image", torch.tensor([0, 1])).shape == (2, 64)


def test_omni_mlp_variant_still_available():
    """Le variant MLP (historique) reste accessible (back-compat)."""
    m = OmniModel(core_type="mlp")
    assert m.core_type == "mlp"
