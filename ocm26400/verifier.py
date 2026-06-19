"""Vérifieur symbolique + dictionnaire de primitives (OCM-26400, spec §2.2).

La gate LSRA interroge ce vérifieur DETERMINISTE pendant la boucle récurrente
pour valider qu'une étape compositionnelle est légale. C'est l'opposé d'un LLM
qui devine : ici la règle est codée explicitement.

Tâche compositionnelle de validation (non-commutative + non-associative) :
    op(a, b) = (A_COEF*a + B_COEF*b) mod P_MOD
sur Z_{P_MOD}. Non-associative => calculer l'intermédiaire m=(a o b) est
VRAIMENT nécessaire : on ne peut pas sauter directement à (a,b,c)->r.
"""
import torch

from .amv import PART  # une primitive vit dans UN slot (ent) = 64 dims

P_MOD = 11
A_COEF = 3   # coefficient de l'entité
B_COEF = 5   # coefficient de la propriété
P_BACKTRACK = 1000.0  # spec §2.2: pénalité massive pour étape illégale


class SymbolicDict:
    """Dictionnaire des primitives grokkées. Chaque primitive = vecteur canonique
    one-hot dans les P_MOD premières dims du slot ent (64). Décodage = argmax + pureté.
    """

    def __init__(self, n: int = P_MOD, dim: int = PART):
        assert n <= dim
        self.n = n
        self.dim = dim

    def canonical(self, idx: int) -> torch.Tensor:
        v = torch.zeros(self.dim)
        if 0 <= idx < self.n:
            v[idx] = 1.0
        return v

    def _matrix(self) -> torch.Tensor:
        """Matrice (n, dim) de toutes les primitives canoniques (pour cosinus)."""
        return torch.stack([self.canonical(i) for i in range(self.n)])

    def decode(self, vec: torch.Tensor):
        """Retourne (idx, valid). valid=True ssi vec est un one-hot pur sur [0,n)."""
        head = vec[: self.n]
        idx = int(torch.argmax(head).item())
        # one-hot pur : un seul 1.0, le reste 0.0 (meme device que head)
        expected = torch.zeros(self.n, device=head.device, dtype=head.dtype)
        expected[idx] = 1.0
        valid = bool(torch.allclose(head, expected, atol=1e-3))
        return idx, valid


class Verifier:
    """Vérifieur symbolique déterministe. Connaît la table d'opération.

    Généralisé : compose_fn(a,b)->int est pluggable (défaut = arithmétique Z_P_MOD).
    Permet de réutiliser le même moteur pour arithmétique OU morphologie linguistique.
    """

    def __init__(self, dictionary: SymbolicDict, compose_fn=None, n_ops: int = 1):
        self.dict = dictionary
        self.n_ops = n_ops
        self._compose_fn = compose_fn  # si None -> arithmétique par défaut

    def compose(self, a: int, b: int, op_id: int = 0) -> int:
        """op(a,b,op_id). Par défaut (3a+5b) mod P_MOD ; sinon la table fournie.
        op_id présent pour le dispatch multi-op futur (CONJUGATE/AGREE...).
        Le chemin compose_fn l'ignore (rétrocompatible avec les expériences existantes)."""
        if self._compose_fn is not None:
            return self._compose_fn(a, b)
        return (A_COEF * a + B_COEF * b) % P_MOD

    def V(self, d_ent: int, d_prop: int, op_id: int = 0) -> bool:
        """V(ent, prop, op) -> True ssi l'opération est légale
        (ent & prop dans le dictionnaire, op connu)."""
        if not (0 <= d_ent < self.dict.n):
            return False
        if not (0 <= d_prop < self.dict.n):
            return False
        if not (0 <= op_id < self.n_ops):
            return False
        return True

    def is_valid_intermediate(self, a: int, b: int, m: int, op_id: int = 0) -> bool:
        """m est-il le bon intermédiaire m = op(a,b,op_id) ?"""
        return m == self.compose(a, b, op_id=op_id)
