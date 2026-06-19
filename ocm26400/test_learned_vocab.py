"""Tests TDD — LearnedVocab (OCM-26400, P2).

Honnête : PAS de test MRR sémantique walk~walked (claim RETIRÉ, falsifié par le
DA : aucun terme de la loss n'injecte de signal distributionnel). On teste
l'IDENTITÉ (roundtrip canonical↔decode), l'anti-collapse sérieux (uniformité +
garde-fou rang), le scaling V>64, et la gate de pureté cosinus.
"""
import torch

from ocm26400.learned_vocab import (
    LearnedVocab,
    TAU_PURE,
    DELTA_MARGIN,
)


# --- identité : canonical est unit-norm ---

def test_canonical_is_unit_norm():
    d = LearnedVocab(n=60)
    for i in [0, 1, 30, 59]:
        v = d.canonical(i)
        assert abs(float(v.norm()) - 1.0) < 1e-5


def test_canonical_out_of_range_is_zero():
    d = LearnedVocab(n=10)
    assert torch.allclose(d.canonical(-1), torch.zeros(64))
    assert torch.allclose(d.canonical(99), torch.zeros(64))


# --- identité : roundtrip canonical -> decode (le contrat cœur) ---

def test_decode_roundtrip_all_primitives_ortho():
    """decode(canonical(i)) == (i, True) pour toute primitive. Identité préservée."""
    d = LearnedVocab(n=60, init="ortho")
    for i in range(60):
        idx, valid = d.decode(d.canonical(i))
        assert idx == i, f"roundtrip failed at {i}: got {idx}"
        assert valid is True, f"decode invalide au roundtrip {i}"


def test_decode_roundtrip_random_init():
    """L'init random (quasi-orthogonale) préserve aussi le roundtrip identité."""
    d = LearnedVocab(n=60, init="random")
    ok = sum(1 for i in range(60) if d.decode(d.canonical(i)) == (i, True))
    assert ok == 60


# --- gate de pureté cosinus ---

def test_decode_rejects_garbage():
    """Un vecteur aléatoire (sans rapport avec E) ⇒ valid=False (cos1<TAU_PURE)."""
    torch.manual_seed(1)
    d = LearnedVocab(n=60, init="ortho")
    n_rejected = 0
    for _ in range(20):
        v = torch.randn(64)
        _, valid = d.decode(v)
        n_rejected += (not valid)
    assert n_rejected >= 18, f"trop de garbage acceptés: {20 - n_rejected}/20"


def test_decode_rejects_ambiguous_midpoint():
    """Mi-chemin entre 2 primitives (équidistant) ⇒ marge<DELTA_MARGIN ⇒ invalide."""
    d = LearnedVocab(n=60, init="ortho")
    mid = d.canonical(0) + d.canonical(1)        # équidistant de 0 et 1
    idx, valid = d.decode(mid)
    assert valid is False, f"ambigu accepté (idx={idx}): la marge ne lève pas l'ambiguïté"


def test_decode_purity_thresholds_exposed():
    """Les seuils cosinus sérieux sont ceux fixés par le verdict (pas euclidien)."""
    assert TAU_PURE == 0.85
    assert DELTA_MARGIN == 0.05


# --- _matrix : shape + différentiable (contrat l_align) ---

def test_matrix_shape_and_differentiable():
    d = LearnedVocab(n=40)
    M = d._matrix()
    assert M.shape == (40, 64)
    loss = (M ** 2).sum()
    loss.backward()
    assert d.E.grad is not None
    assert d.E.grad.shape == (40, 64)


# --- anti-collapse (Leçon 5) : uniformité ---

def test_uniformity_loss_near_zero_for_ortho():
    """Init ortho ⇒ cos inter-paires = 0 ⇒ uniformity_loss ≈ 0."""
    d = LearnedVocab(n=60, init="ortho")
    assert float(d.uniformity_loss()) < 1e-6


def test_uniformity_loss_positive_and_reduces_when_trained():
    """Un E collapsé a une uniformity_loss élevée ; l'optimiser la réduit."""
    d = LearnedVocab(n=30, init="random")
    with torch.no_grad():
        d.E.copy_(torch.randn(30, 64))           # reset non normalisé
    # collapse délibéré : tous vers le même vecteur
    with torch.no_grad():
        base = torch.randn(64)
        d.E.copy_(base.unsqueeze(0).expand(30, 64) + 0.01 * torch.randn(30, 64))
    loss_before = float(d.uniformity_loss())
    opt = torch.optim.Adam([d.E], lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        d.uniformity_loss().backward()
        opt.step()
    loss_after = float(d.uniformity_loss())
    assert loss_after < loss_before, "l'uniformité ne réduit pas le collapse"
    assert loss_after < 0.1, f"collapse non résolu: uniformity={loss_after}"


def test_mean_inter_pair_cos_below_threshold():
    """E sain ⇒ cos moyen inter-paires ≤ 0.5 (seuil sérieux du verdict)."""
    d = LearnedVocab(n=60, init="ortho")
    assert d.mean_inter_pair_cos() <= 0.5


# --- garde-fou secondaire : rang ---

def test_rank_guard():
    """matrix_rank(E) ≥ min(V, dim) − 2 (garde-fou collapse, secondaire)."""
    for n in [30, 60]:
        d = LearnedVocab(n=n, init="ortho")
        r = int(torch.linalg.matrix_rank(d.E.detach()).item())
        assert r >= min(n, 64) - 2, f"rang trop bas pour n={n}: {r}"


# --- scaling V > 64 (impossible en one-hot : assert n<=dim) ---

def test_supports_v_greater_than_64():
    """V>64 : construction OK (pas d'assert n<=dim) + roundtrip identité."""
    d = LearnedVocab(n=120, init="random")
    assert d.n == 120
    ok = sum(1 for i in range(120) if d.decode(d.canonical(i)) == (i, True))
    assert ok >= 118, f"roundtrip V=120: {ok}/120 valides"


# --- freeze : codebook fixe = analogue one-hot ---

def test_freeze_disables_grad():
    d = LearnedVocab(n=20)
    assert d.E.requires_grad is True
    d.freeze()
    assert d.E.requires_grad is False
