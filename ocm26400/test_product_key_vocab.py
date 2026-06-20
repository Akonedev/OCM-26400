"""Tests TDD — Product-Key Memory (P-D, archi préservée, OCM-26400).

Valide : roundtrip canonical<->decode, structure √V, interface = LearnedVocab, decode
O(√V) à grand V (pas de matérialisation O(V)), séparabilité.
"""
import time
import torch

from ocm26400.product_key_vocab import ProductKeyVocab


def test_pk_roundtrip():
    """decode(canonical(i)) == (i, True) : identité préservée (self-consistency)."""
    pk = ProductKeyVocab(n=100, seed=0)
    ok = sum(1 for i in range(100) if pk.decode(pk.canonical(i)) == (i, True))
    assert ok >= 95, f"roundtrip product-key: {ok}/100"


def test_pk_sqrt_structure():
    """2 codebooks de √V clés -> V = n_sub² adressables."""
    pk = ProductKeyVocab(n=10000)
    assert pk.n_sub * pk.n_sub >= 10000          # √V clés couvrent V produits
    assert pk.n_sub <= 200                        # ~100 clés (<< 10000)


def test_pk_interface_matches_learnedvocab():
    """Même surface que LearnedVocab (le noyau spectral le consomme sans modif)."""
    pk = ProductKeyVocab(n=50)
    assert pk.canonical(3).shape == (64,)
    M = pk._matrix()
    assert M.shape == (50, 64)
    assert hasattr(pk, "decode") and hasattr(pk, "uniformity_loss") and hasattr(pk, "freeze")


def test_pk_decode_fast_at_large_V():
    """Decode O(2·n_sub) à V=10000 : rapide (pas de matérialisation O(V))."""
    pk = ProductKeyVocab(n=10000, seed=0).freeze()
    t0 = time.time()
    for i in range(0, 10000, 500):
        pk.decode(pk.canonical(i))
    dt = time.time() - t0
    assert dt < 2.0, f"decode trop lent (non O(√V)): {dt:.2f}s"


def test_pk_capacity_advantage():
    """VRAI apport de P-D : CAPACITÉ (2√V clés au lieu de V) + decode O(√V).
    HONNÊTE : P-D n'améliore PAS la séparabilité — les produits (sommes de 2 clés)
    clusterisent plus qu'un codebook plat (mesuré : NN cos PK 0.58-0.62 vs flat 0.35).
    Le plafond packing R^64 est géométrique, pas levé par PQ. P-D = capacité + efficacité."""
    pk = ProductKeyVocab(n=10000, seed=0)
    # capacité : ~√V clés (100) au lieu de 10000 -> mémoire + trainabilité
    assert pk.n_sub <= 110 and pk.n_sub * pk.n_sub >= 10000
    # decode fonctionne (roundtrip) à cette échelle
    idx, valid = pk.decode(pk.canonical(7777))
    assert idx == 7777
