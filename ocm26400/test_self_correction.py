"""Tests TDD — auto-correction / auto-amélioration (OCM-26400).

Valide que le modèle se corrige (re-raisonnement rattrape les erreurs mémoire) et
s'améliore (justesse -> 100%). Block-oracle (parfait) + mémoire seedée avec erreurs.
"""
import torch
import torch.nn as nn

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import encode_input
from ocm26400.cognitive_agent import CognitiveAgent
from ocm26400.self_correction import (
    reason_pair, self_consistency_confidence, self_correct, self_improve,
)


class _OracleBlock(nn.Module):
    """Block parfait : (a,b) -> canonical(op(a,b)). Re-raisonner donne la vérité."""
    def __init__(self, d, ver):
        super().__init__()
        self._dummy = nn.Linear(1, 1)
        self.d, self.ver = d, ver

    def forward(self, x):
        out = x.clone()
        for i in range(x.shape[0]):
            a = int(x[i, 0:64].argmax()); b = int(x[i, 64:128].argmax())
            out[i, 0:64] = self.d.canonical(self.ver.compose(a, b))
        return out


def _agent_with_errors(frac_wrong=0.5):
    """Agent oracle dont la mémoire contient une fraction de faits ERRONÉS."""
    d = SymbolicDict(n=P_MOD); ver = Verifier(d)
    ag = CognitiveAgent(_OracleBlock(d, ver), d, ver)
    import random
    random.seed(0)
    for a in range(P_MOD):
        for b in range(P_MOD):
            true_r = ver.compose(a, b)
            r = (true_r + (1 if random.random() < frac_wrong else 0)) % P_MOD  # parfois faux
            ag.memory[(a, b)] = r
    return ag, ver


def test_self_correct_fixes_wrong_facts():
    """Mémoire avec erreurs -> auto-correction les répare, justesse monte."""
    ag, ver = _agent_with_errors(frac_wrong=0.5)
    stats = self_correct(ag, ver)
    assert stats["corrected"] > 0
    assert stats["acc_after"] > stats["acc_before"]
    assert stats["acc_after"] == 1.0          # toutes les erreurs rattrapées


def test_self_correct_keeps_correct_memory_at_100():
    """Mémoire déjà correcte -> aucune correction, justesse reste 100%."""
    ag, ver = _agent_with_errors(frac_wrong=0.0)
    stats = self_correct(ag, ver)
    assert stats["corrected"] == 0
    assert stats["acc_after"] == 1.0


def test_self_improve_converges_to_100():
    """L'auto-amélioration converge vers 100% de justesse (courbe croissante)."""
    ag, ver = _agent_with_errors(frac_wrong=0.5)
    curve = self_improve(ag, ver, rounds=5)
    assert curve[-1]["acc_after"] == 1.0
    assert curve[-1]["corrected"] == 0         # convergence : plus rien à corriger


def test_self_consistency_high_without_noise():
    """Block oracle déterministe -> self-consistency = 1.0 (confiance maximale)."""
    ag, ver = _agent_with_errors(frac_wrong=0.0)
    conf = self_consistency_confidence(ag, 2, 5, k=5, noise_std=0.0)
    assert abs(conf - 1.0) < 1e-6
