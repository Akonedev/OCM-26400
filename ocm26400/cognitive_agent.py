"""Agent cognitif auto-apprenant (OCM-26400, cahier des charges).

INTÉGRATION des briques en un cycle cognitif cohérent — le flow exact du cahier
des charges pour la capacité « apprentissage » :

    question (a,b) ->
      1. RETRIEVE : ai-je déjà la réponse en mémoire ?  -> 'je sais' (rapide)
      2. sinon RAISONNER : block grokké -> r_pred
      3. VÉRIFIER : is_valid_intermediate(a,b,r_pred) via le Verifier (gate symbolique)
      4. si valide -> APPRENDRE (stocker en mémoire) -> 'raisonné+appris'
         si invalide -> ABSTENTION ('je ne sais pas', [ANOMALIE_CAUSALE])

Ça combine : composition (ReasonerBlock grokké, crown-jewel) + gate symbolique
(Verifier.is_valid_intermediate) + mémoire/retrieval + apprentissage online +
abstention (P3). L'agent part d'une mémoire VIDE et apprend au fil des requêtes :
les réponses répétées deviennent des 'retrieved' (O(1)), les nouvelles sont
calculées+vérifiées puis stockées.

Honnête : sur op(a,b)=(3a+5b) mod 11 (grok parfait), le raisonnement est exact
donc l'apprentissage stocke des faits corrects et l'abstention est rare. La
démonstration porte sur le MÉCANISME du cycle (courbe d'apprentissage, bascule
raisonné->retrieved), pas sur une tâche ouverte.
"""
from typing import Optional, Tuple, Dict
import torch

from .verifier import SymbolicDict, Verifier
from .reasoner import ReasonerBlock, encode_input


class CognitiveAgent:
    """Agent auto-apprenant : retrieve -> raisonner -> vérifier -> apprendre / abstention."""

    def __init__(self, blk: ReasonerBlock, dictionary: SymbolicDict, verifier: Verifier):
        self.blk = blk
        self.d = dictionary
        self.ver = verifier
        self.memory: Dict[Tuple[int, int], int] = {}   # (a,b) -> r  (faits appris)
        self.stats = {"retrieved": 0, "reasoned": 0, "abstained": 0, "errors": 0}

    @torch.no_grad()
    def solve(self, a: int, b: int) -> Tuple[Optional[int], str]:
        """Cycle cognitif. Retourne (réponse, mode)."""
        # 1. RETRIEVE
        if (a, b) in self.memory:
            self.stats["retrieved"] += 1
            return self.memory[(a, b)], "retrieved"
        # 2. RAISONNER
        dev = next(self.blk.parameters()).device
        x = encode_input(a, b, self.d).unsqueeze(0).to(dev)
        out = self.blk(x)[0]
        r_pred, _ = self.d.decode(out[0:64])
        # 3. VÉRIFIER (gate symbolique)
        if self.ver.is_valid_intermediate(a, b, r_pred):
            # 4a. APPRENDRE
            r_true = self.ver.compose(a, b)
            correct = (r_pred == r_true)
            self.memory[(a, b)] = r_pred
            self.stats["reasoned"] += 1
            self.stats["errors"] += (not correct)
            return r_pred, "reasoned+learned"
        # 4b. ABSTENTION
        self.stats["abstained"] += 1
        return None, "abstained"

    def knowledge_size(self) -> int:
        return len(self.memory)

    def solve_chain(self, chain):
        """Requête COMPOSITIONNELLE : r = op(...op(op(chain[0],chain[1]),chain[2]),...).

        Compose le cycle binaire (solve) sur chaque étape de la chaîne : chaque
        intermédiaire est raisonné+vérifié+appris (récurrence fenêtrée intégrée au
        cycle cognitif). Retourne (réponse, modes_par_étape). Si une étape abstient,
        la chaîne échoue (réponse None) — l'incertitude se propage honnêtement.
        """
        if len(chain) < 2:
            return (chain[0] if chain else None), []
        cur = chain[0]
        modes = []
        for nxt in chain[1:]:
            cur, mode = self.solve(cur, nxt)
            modes.append(mode)
            if cur is None:               # abstention -> chaîne échoue
                return None, modes
        return cur, modes

    def accuracy(self) -> float:
        """Précision des faits raisonnés stockés (vs vérité)."""
        if not self.memory:
            return 1.0
        ok = sum(1 for (a, b), r in self.memory.items() if r == self.ver.compose(a, b))
        return ok / len(self.memory)
