"""Tests génération de monde neuronal (OCM-26400)."""
import torch
from ocm26400.world import NeuralWorldModel, generate_world_fragment


def test_neural_world_forward():
    m = NeuralWorldModel(latent_dim=64, grid_size=8, n_features=4)
    latent = torch.randn(2, 64)
    out = m(latent)
    assert out.shape == (2, 8, 4)


def test_neural_world_no_transformer():
    m = NeuralWorldModel()
    from ocm26400.spectral_core import SpectralCoreBlock
    assert isinstance(m.core, SpectralCoreBlock)
    has_tf = any("Transformer" in type(mod).__name__ or "Attention" in type(mod).__name__
                 for mod in m.modules())
    assert not has_tf


def test_generate_world_fragment():
    w = generate_world_fragment(seed=42, grid_size=4)
    assert len(w["terrain"]) == 4
    assert w["features"]["diversity"] > 0
    assert isinstance(w["description"], str)


def test_generate_world_deterministic():
    w1 = generate_world_fragment(seed=42)
    w2 = generate_world_fragment(seed=42)
    assert w1["terrain"] == w2["terrain"]
