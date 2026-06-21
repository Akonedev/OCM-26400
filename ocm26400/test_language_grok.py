"""Tests grok des primitives linguistiques (OCM-26400) — model dev clé."""
import pytest
from ocm26400.language_grok import grok_word_number, grok_cue_operation, run_language_grok, _word_to_hash_pos


def test_word_to_hash_pos():
    """Hash stable d'un mot → position [0, 64)."""
    p1 = _word_to_hash_pos("three")
    p2 = _word_to_hash_pos("three")
    assert p1 == p2  # déterministe
    assert 0 <= p1 < 64


def test_grok_word_number():
    """LE test : le SpectralCoreBlock GROK word→number (>50%)."""
    blk, res = grok_word_number(n_steps=800)
    assert res["grok_acc"] >= 0.5, f"word→number should grok >50%, got {res['grok_acc']}"


def test_grok_cue_operation():
    """LE test : le SpectralCoreBlock GROK cue→operation (>50%)."""
    blk, res = grok_cue_operation(n_steps=800)
    assert res["grok_acc"] >= 0.5, f"cue→operation should grok >50%, got {res['grok_acc']}"


def test_run_language_grok():
    rep = run_language_grok()
    assert rep["phase"].startswith("SOLO")
    assert len(rep["primitives"]) == 2
