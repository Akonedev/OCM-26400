"""Tests RED — LSRA pleine boucle (spec §3: test-time compute avec gate confidence).

  v(t+1) = ReasonerBlock(v(t))
  T* = min { t | confidence(v(t)) >= tau_grok }
  si T* > MaxIter -> [ANOMALIE_CAUSALE] (jamais confiant)

On valide: (1) la boucle s'arrete dans max_iter, (2) stop anticipé quand confiance
haute, (3) flag anomalie quand jamais confiant, (4) sur block entraîné (avec
supervision de confiance) la boucle résout op(a,b) et s'arrete confiante.
"""
import pytest
import torch
from ocm26400.amv import D_MODEL, AMVVector
from ocm26400.verifier import SymbolicDict, Verifier
from ocm26400.reasoner import (
    ReasonerBlock, encode_input, lsra_loop, train_reasoner_with_confidence, TAU_GROK,
)


@pytest.fixture
def setup():
    d = SymbolicDict()
    return d, Verifier(d)


def test_lsra_loop_returns_within_max_iter(setup):
    d, ver = setup
    blk = ReasonerBlock()
    x0 = encode_input(2, 5, d)
    idx, n_steps, confident = lsra_loop(blk, d, x0, max_iter=4)
    assert 1 <= n_steps <= 4
    assert isinstance(confident, bool)


def test_lsra_loop_flags_anomaly_when_never_confident(setup):
    """Un block NON entraîné (sortie ~aléatoire) ne doit jamais atteindre tau_grok
    en peu d'itérations -> confident=False (anomalie)."""
    d, ver = setup
    torch.manual_seed(123)
    blk = ReasonerBlock()  # non entraîné
    x0 = encode_input(2, 5, d)
    # tau tres haut pour forcer l'anomalie
    idx, n_steps, confident = lsra_loop(blk, d, x0, max_iter=3, tau=0.999)
    assert confident is False
    assert n_steps == 3  # a épuisé le budget


def test_lsra_loop_stops_early_on_confident_block(setup):
    """Block entraîné AVEC supervision de confiance -> la boucle s'arrete tôt et confiante."""
    d, ver = setup
    blk = train_reasoner_with_confidence(d, ver, n_steps=600)
    x0 = encode_input(3, 7, d)
    idx, n_steps, confident = lsra_loop(blk, d, x0, max_iter=8, tau=TAU_GROK)
    assert confident is True
    assert n_steps <= 4  # stop anticipé


def test_confident_block_solves_binary_op_via_loop(setup):
    """End-to-end : la boucle LSRA confidente récupère le bon op(a,b)."""
    d, ver = setup
    blk = train_reasoner_with_confidence(d, ver, n_steps=800)
    correct = 0
    for a in range(d.n):
        for b in range(d.n):
            x0 = encode_input(a, b, d)
            idx, _, confident = lsra_loop(blk, d, x0, max_iter=8, tau=TAU_GROK)
            if confident and idx == ver.compose(a, b):
                correct += 1
    acc = correct / (d.n * d.n)
    assert acc > 0.9  # la boucle LSRA résout op(a,b) > 90%
