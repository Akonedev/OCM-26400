"""Tests TDD — CognitiveAgent (cycle retrieve/raisonner/vérifier/apprendre).

Block-oracle mock (renvoie canonical(op(a,b))) pour des tests déterministes et
rapides, sans entraînement GPU. Valide la logique du cycle cognitif.
"""
import torch
import torch.nn as nn

from ocm26400.verifier import SymbolicDict, Verifier, P_MOD
from ocm26400.reasoner import encode_input
from ocm26400.cognitive_agent import CognitiveAgent


class _OracleBlock(nn.Module):
    """Block parfait : extrait (a,b) de l'input, écrit canonical(op(a,b)) dans ent."""
    def __init__(self, d, ver):
        super().__init__()
        self._dummy = nn.Linear(1, 1)
        self.d, self.ver = d, ver

    def forward(self, x):
        out = x.clone()
        for i in range(x.shape[0]):
            a = int(x[i, 0:64].argmax())
            b = int(x[i, 64:128].argmax())
            out[i, 0:64] = self.d.canonical(self.ver.compose(a, b))
        return out


class _GarbageBlock(nn.Module):
    """Block défaillant : ent = zéros => decode invalide => abstention."""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, x):
        out = x.clone()
        out[:, 0:64] = 0.0
        return out


def _agent(block_cls="oracle"):
    d = SymbolicDict(n=P_MOD)
    ver = Verifier(d)
    blk = _OracleBlock(d, ver) if block_cls == "oracle" else _GarbageBlock()
    return CognitiveAgent(blk, d, ver)


def test_first_solve_reasoned_then_retrieved():
    """1re requête -> raisonné+appris ; 2e (même paire) -> retrieved (mémoire)."""
    ag = _agent()
    r1, mode1 = ag.solve(2, 5)
    assert mode1 == "reasoned+learned"
    assert r1 == ag.ver.compose(2, 5)
    r2, mode2 = ag.solve(2, 5)
    assert mode2 == "retrieved"
    assert r2 == r1


def test_memory_grows_with_learning():
    """Chaque nouveau paire raisonner grossit la mémoire."""
    ag = _agent()
    for a in range(P_MOD):
        ag.solve(a, 0)
    assert ag.knowledge_size() == P_MOD
    assert ag.stats["reasoned"] == P_MOD
    assert ag.stats["retrieved"] == 0


def test_abstention_on_invalid_block():
    """Block défaillant (ent invalide) -> abstention, réponse None."""
    ag = _agent("garbage")
    r, mode = ag.solve(3, 4)
    assert mode == "abstained"
    assert r is None
    assert ag.stats["abstained"] == 1
    assert ag.knowledge_size() == 0      # rien d'appris sur abstention


def test_accuracy_one_with_oracle():
    """Tous les faits raisonnés par l'oracle sont corrects (accuracy 1.0)."""
    ag = _agent()
    for _ in range(50):
        ag.solve(torch.randint(0, P_MOD, (1,)).item(), torch.randint(0, P_MOD, (1,)).item())
    assert ag.accuracy() == 1.0


def test_solve_chain_composes_and_learns():
    """Requête compositionnelle [a,b,c] -> op(op(a,b),c). Intermédiaires appris."""
    ag = _agent()
    chain = [2, 5, 7]
    r, modes = ag.solve_chain(chain)
    truth = ag.ver.compose(ag.ver.compose(2, 5), 7)
    assert r == truth
    assert all(m == "reasoned+learned" for m in modes)   # 2 étapes raisonnées+apprises
    # les intermédiaires (2,5) et (op(2,5),7) sont en mémoire
    assert (2, 5) in ag.memory


def test_solve_chain_abstention_propagates():
    """Block défaillant -> la chaîne abstient à la 1re étape, réponse None."""
    ag = _agent("garbage")
    r, modes = ag.solve_chain([3, 4, 2])
    assert r is None
    assert modes[0] == "abstained"
