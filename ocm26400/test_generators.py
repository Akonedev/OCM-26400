"""Tests TDD — décodeur flow-matching (vraie génération de signal, E5, OCM-26400).

Valide : génération shape, différentiable, distincte par condition, et la RECONSTRUCTION
baisse après entraînement (le décodeur APPREND à générer le signal cible depuis l'AMV).
"""
import torch

from ocm26400.generators import AMVConditionedDecoder


def test_decoder_sample_shape():
    dec = AMVConditionedDecoder(x_dim=16, cond_dim=8)
    cond = torch.randn(3, 8)
    out = dec.sample(cond, steps=8)
    assert out.shape == (3, 16)


def test_flow_match_loss_differentiable():
    dec = AMVConditionedDecoder(x_dim=16, cond_dim=8)
    cond = torch.randn(4, 8)
    x = torch.randn(4, 16)
    loss = dec.flow_match_loss(cond, x)
    loss.backward()
    assert dec.net[0].weight.grad is not None
    assert dec.net[0].weight.grad.abs().sum() > 0


def test_decoder_distinct_per_condition():
    """Conditions AMV différentes -> signaux générés différents."""
    dec = AMVConditionedDecoder(x_dim=16, cond_dim=8)
    cond = torch.randn(2, 8)
    torch.manual_seed(0)
    out = dec.sample(cond, steps=8)
    assert not torch.allclose(out[0], out[1])


def test_decoder_learns_to_generate_targets():
    """CLÉ E5 : après entraînement, le décodeur génère des signaux proches des cibles
    (reconstruction MSE baisse) = vraie génération apprise, pas MSE-régression cosmétique."""
    torch.manual_seed(0)
    dec = AMVConditionedDecoder(x_dim=16, cond_dim=8)
    cond = torch.randn(4, 8)                       # 4 conditions distinctes
    targets = torch.randn(4, 16) * 3 + 1           # 4 signaux cibles distincts
    opt = torch.optim.Adam(dec.parameters(), lr=5e-3)

    def recon_mse():
        return float(((dec.sample(cond, steps=8) - targets) ** 2).mean())

    mse_before = recon_mse()
    for _ in range(400):
        loss = dec.flow_match_loss(cond, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    mse_after = recon_mse()
    assert mse_after < mse_before, f"la génération n'apprend pas: {mse_before:.2f}->{mse_after:.2f}"
