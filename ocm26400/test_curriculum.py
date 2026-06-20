"""Tests TDD — curriculum progressif (OCM-26400, paradigme d'entraînement complet).

Valide : phases (primitives → paires → chaînes → inter-règles), progressif (avance si
accuracy ≥ seuil), anti-shortcut (gap train/test).
"""
from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import ReasonerBlock
from ocm26400.curriculum import Curriculum, PhaseResult, should_advance, anti_shortcut_check


def test_curriculum_phases_order():
    """Les phases vont des primitives aux inter-règles (macro←micro)."""
    c = Curriculum()
    phases = c.phases()
    assert phases[0] == "primitives"
    assert phases[-1] == "inter-règles"
    assert len(phases) == 4


def test_curriculum_progressive_threshold():
    """Progressif : une phase est 'passée' si accuracy ≥ seuil ET gap acceptable."""
    r_pass = PhaseResult("primitives", accuracy=0.95, train_test_gap=0.05, passed=True, steps=50)
    r_fail = PhaseResult("primitives", accuracy=0.60, train_test_gap=0.05, passed=False, steps=50)
    assert r_pass.passed is True
    assert r_fail.passed is False


def test_anti_shortcut_detects_memorization():
    """Anti-shortcut : un grand gap train/test = mémorisation (pas de généralisation)."""
    r_ok = PhaseResult("primitives", accuracy=0.90, train_test_gap=0.05, passed=True, steps=50)
    r_memo = PhaseResult("primitives", accuracy=0.90, train_test_gap=0.40, passed=False, steps=50)
    assert r_ok.passed is True
    assert r_memo.passed is False       # gap trop grand = mémorisation


def test_should_advance_all_passed():
    """should_advance : True si toutes les phases passées."""
    results_ok = [PhaseResult("a", 0.9, 0.05, True, 50), PhaseResult("b", 0.9, 0.05, True, 50)]
    assert should_advance(results_ok) is True
    results_fail = [PhaseResult("a", 0.9, 0.05, True, 50), PhaseResult("b", 0.5, 0.05, False, 50)]
    assert should_advance(results_fail) is False
