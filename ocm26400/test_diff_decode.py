"""Tests TDD — L_step différentiable + ACSP câblé (P-B, OCM-26400).

Le test clé : acsp_loss_diff produit un gradient non-nul sur le noyau (l_step était une
constante avant — acsp_loss ne contribuait aucun gradient de légalité).
"""
import torch

from ocm26400.verifier import SymbolicDict, Verifier
from ocm26400.amv import AMVVector
from ocm26400.reasoner import ReasonerBlock, encode_input
from ocm26400.diff_decode import (
    decode_gumbel, l_step_diff, acsp_loss_diff, train_with_acsp, eval_binary,
)


def test_decode_gumbel_straight_through():
    """decode_gumbel -> quasi-one-hot (n,) avec gradient (straight-through)."""
    logits = torch.randn(11, requires_grad=True)
    y = decode_gumbel(logits, tau=1.0, hard=True)
    assert y.shape == (11,)
    assert y.sum().backward() is None                       # backward OK
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_acsp_loss_diff_gradients_flow_to_core():
    """CLÉ P-B : acsp_loss_diff produit un gradient non-nul sur fc1 (l_step différentiable).
    Avant, l_step était une constante -> acsp_loss ne contribuait aucun gradient de légalité."""
    d = SymbolicDict(n=11); ver = Verifier(d)
    blk = ReasonerBlock()
    x = encode_input(2, 3, d).unsqueeze(0)
    out = blk(x)
    loss = acsp_loss_diff(AMVVector(out[0]), d, ver, 2, 3)
    loss.backward()
    assert blk.fc1.weight.grad is not None
    assert blk.fc1.weight.grad.abs().sum().item() > 0       # gradient RÉEL de la légalité


def test_train_with_acsp_converges():
    """Le trainer ACSP-wired grok op(a,b) (ACSP vit enfin dans l'entraînement)."""
    d = SymbolicDict(n=11); ver = Verifier(d)
    blk = train_with_acsp(d, ver, n_steps=800)
    acc = eval_binary(blk, d, ver, n_test=121)
    assert acc > 0.9, f"trainer ACSP non convergé: {acc:.2f}"


def test_existing_acsp_loss_unchanged():
    """Non-régression : l'ancienne acsp_loss (constante) fonctionne toujours."""
    from ocm26400.acsp import acsp_loss
    d = SymbolicDict(n=11); ver = Verifier(d)
    blk = ReasonerBlock()
    x = encode_input(1, 2, d).unsqueeze(0)
    out = blk(x)
    loss = acsp_loss(AMVVector(out[0]), d, ver, 1, 2)       # ancienne API, toujours là
    assert loss is not None
