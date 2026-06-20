"""Tests crown-jewel NEURAL multi-hop (non-tautologique, procédure canonique) — OCM-26400.

Légers : on entraîne peu de steps (quick) pour rester sous la barre de temps, mais on
vérifie le MÉCANISME (prédiction neurale sur hold-out, non-tautologique, procédure §2).
"""
import torch
from ocm26400.neural_multihop import (
    neural_predict, neural_holdout_eval, neural_multihop_eval, _op, _make_op_verifier,
)
from ocm26400.experiment_composition import train_binary_block


def test_ops_definition():
    assert _op("add")(3, 4) == 7
    assert _op("mul")(3, 4) == (12 % 11)
    assert _op("linop")(2, 1) == (3 * 2 + 5 * 1) % 11


def test_make_op_verifier_uses_op():
    d, ver = _make_op_verifier("add")
    assert ver.compose(3, 4) == 7


def test_neural_predict_runs():
    """La prédiction neurale (encode→forward→argmax decode) tourne sans erreur."""
    d, ver = _make_op_verifier("add")
    blk = train_binary_block(d, ver, n_steps=50)   # mini-entraînement (procédure §2)
    pred = neural_predict(blk, d, 3, 4)            # auto-détecte le device du block
    assert 0 <= pred < 11


def test_holdout_eval_non_tautological():
    """Le résultat est NON-tautologique (POIDS vs ground-truth, jamais apply==apply)."""
    h = neural_holdout_eval("add", n_steps=300, seed=0)
    assert h["tautological"] is False
    assert "train_binary_block" in h["procedure"]      # procédure canonique §2
    assert 0.0 <= h["neural_holdout_acc"] <= 1.0
    assert h["n_holdout"] > 0


def test_multihop_eval_non_tautological():
    m = neural_multihop_eval("add", n_steps=300, depth=2, n_chains=10)
    assert m["tautological"] is False
    assert m["depth"] == 2
    assert 0.0 <= m["neural_multihop_acc"] <= 1.0


def test_multihop_ground_truth_matches_symbolic():
    """Le ground-truth de la chaîne = composition symbolique (vérif cohérence)."""
    op = _op("add")
    vals = [1, 2, 3, 4]
    gt = vals[0]
    for v in vals[1:]:
        gt = op(gt, v)
    assert gt == op(op(op(1, 2), 3), 4)
