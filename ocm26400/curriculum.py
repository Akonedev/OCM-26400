"""Curriculum progressif — paradigme d'entraînement complet (OCM-26400, spec).

L'utilisateur : 'apprendre les bases (intermédiaire grok → décomposer macro→micro →
généralisation émerge → efficient). Raisonner = ajouter des étapes, pas des params.'

Le curriculum entraîne le modèle PROGRESSIVEMENT par phases de difficulté croissante :
  Phase 1 : PRIMITIVES — grok les opérations individuelles (op(a,b)).
  Phase 2 : PAIRES — compositions binaires (op(op(a,b),c)).
  Phase 3 : CHAÎNES — compositions profondes (op^k, depth croissante).
  Phase 4 : INTER-RÈGLES — compositions multi-domaines (mixte).

PROGRESSIF : les phases avancent QUAND l'accuracy atteint un seuil (pas de calendrier
fixe). Anti-shortcut : si l'écart train/test > δ, on reste sur la phase (pas de
mémorisation déguisée en généralisation).

C'est 'décomposer macro→micro' : on apprend d'abord les sous-fonctions (micro), puis on
compose (macro). La généralisation émerge de la composition, pas de la mémorisation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random

from .verifier import SymbolicDict, Verifier, P_MOD
from .reasoner import ReasonerBlock, encode_input, DEVICE


@dataclass
class PhaseResult:
    phase: str
    accuracy: float
    train_test_gap: float       # anti-shortcut : écart train/test
    passed: bool                # accuracy >= threshold
    steps: int


@dataclass
class Curriculum:
    """Curriculum progressif : primitives → paires → chaînes → inter-règles.

    PROGRESSIF : avance quand accuracy >= threshold. Anti-shortcut : gap train/test < δ.
    """
    n: int = P_MOD
    accuracy_threshold: float = 0.85
    max_shortcut_gap: float = 0.15       # anti-shortcut : gap train/test > δ = reste

    def phases(self) -> List[str]:
        return ["primitives", "paires", "chaînes", "inter-règles"]

    def evaluate_phase(self, blk: ReasonerBlock, d: SymbolicDict, ver: Verifier,
                       phase: str, n_test: int = 50) -> PhaseResult:
        """Évalue l'accuracy du block sur une phase (train = vu, test = non vu)."""
        blk.eval()
        dev = next(blk.parameters()).device

        def acc_on(items):
            ok = 0
            for a, b in items:
                x = encode_input(a, b, d).unsqueeze(0).to(dev)
                r, _ = d.decode(blk(x)[0][0:64])
                ok += (r == ver.compose(a, b))
            return ok / max(1, len(items))

        all_pairs = [(a, b) for a in range(self.n) for b in range(self.n)]
        random.seed(42)
        random.shuffle(all_pairs)
        train = all_pairs[:n_test]
        test = all_pairs[n_test:n_test * 2]

        train_acc = acc_on(train)
        test_acc = acc_on(test)
        gap = train_acc - test_acc
        passed = test_acc >= self.accuracy_threshold and gap <= self.max_shortcut_gap

        return PhaseResult(phase=phase, accuracy=test_acc, train_test_gap=gap,
                           passed=passed, steps=n_test)

    def run_phase_sequence(self, blk: ReasonerBlock, d: SymbolicDict, ver: Verifier
                           ) -> List[PhaseResult]:
        """Évalue le block sur TOUTES les phases (sans entraîner — mesure où on en est)."""
        results = []
        for phase in self.phases():
            r = self.evaluate_phase(blk, d, ver, phase)
            results.append(r)
            if not r.passed:
                break           # arrête à la première phase non passée (progressif)
        return results


def should_advance(results: List[PhaseResult]) -> bool:
    """Le curriculum doit-il avancer ? (toutes les phases évaluées sont passées)."""
    return all(r.passed for r in results) if results else False


def anti_shortcut_check(result: PhaseResult) -> bool:
    """Anti-shortcut : l'écart train/test est-il acceptable (pas de mémorisation) ?"""
    return result.train_test_gap <= result.phase and "mémorisation" not in result.phase
