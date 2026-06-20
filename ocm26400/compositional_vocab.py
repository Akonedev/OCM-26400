"""Vocabulaire compositionnel SCALABLE (OCM-26400, cahier des charges §1).

Réponse paradigmatique au « TOUT le vocabulaire (1M mots) » du cahier des charges :
on NE met PAS 1M symboles dans un slot ent 64-dim (impossible — packing montré en
P2). On les COMPOSE depuis un ensemble fini de morphèmes/primitives, exactement comme
la linguistique réelle (infini de mots depuis un nombre fini de morphèmes/lexèmes) et
comme le mécanisme crown-jewel le prouve (composition de sous-fonctions grokkées).

    espace adressable = P_primitives ^ longueur_mot     (EXPONENTIEL)

Avec P=20 morphèmes et longueur jusqu'à 4 : 20^4 = 160 000 mots adressables. Avec
longueur 5 : 3.2M. L'espace adressable n'est DONC PAS borné par le slot 64-dim — il
l'est par la profondeur de composition, qui est elle-même illimitée (récurrence
fenêtrée, experiment_recursion prouve profondeur 5+).

word_vector(seq) : embedding compositionnel positionnel — somme de projections
positionnelles des embeddings de morphèmes, normalisée. Positions distinctes =>
séquences distinctes (projections aléatoires fixes par position). Côté déterministe
de la composition (le côté APPRENTIS est le ReasonerBlock / crown-jewel).

HONNÊTE (frontière) : l'espace ADRESSABLE est exponentiel, mais la SÉPARABILITÉ du
retrieval reste bornée par la dimension 64 (packing) — à très grand lexique dans
64-dim, les vecteurs compositionnels se rapprochent. Le levier pour séparer un très
grand vocabulaire = dim>64 (casser la partition AMV fixe) OU index hiérarchique.
On documente les deux : l'addressage scale, la séparabilité pure suit le packing.
"""
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from .learned_vocab import LearnedVocab
from .amv import PART


class CompositionalVocabulary(nn.Module):
    """Vocabulaire composé : mots = compositions de morphèmes. Espace P^L."""

    def __init__(self, primitives: LearnedVocab, max_len: int = 4, seed: int = 0):
        super().__init__()
        self.prim = primitives
        self.max_len = max_len
        self.dim = primitives.dim
        g = torch.Generator().manual_seed(seed)
        # projection positionnelle fixe (distincte par position) => séquences distinctes
        self.pos_proj = nn.ParameterList([
            nn.Parameter(torch.randn(self.dim, self.dim, generator=g) * (1.0 / self.dim))
            for _ in range(max_len)
        ])

    def addressable_space(self) -> int:
        """Nombre de mots adressables = primitives^longueur (EXPONENTIEL, >> slot)."""
        return self.prim.n ** self.max_len

    def word_vector(self, morpheme_seq: List[int]) -> torch.Tensor:
        """Embedding compositionnel positionnel du mot (séquence de morphèmes)."""
        v = torch.zeros(self.dim)
        for t, m in enumerate(morpheme_seq[: self.max_len]):
            v = v + self.pos_proj[t] @ self.prim.canonical(m)
        return v / (v.norm() + 1e-8)

    @torch.no_grad()
    def build_index(self, lexicon: List[List[int]]):
        """Indexe un lexicon (liste de séquences de morphèmes). Retourne (matrix, lexicon)."""
        M = torch.stack([self.word_vector(w) for w in lexicon])
        return M, lexicon

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor, index_matrix: torch.Tensor,
                 lexicon: List[List[int]], threshold: float = 0.5
                 ) -> Tuple[Optional[List[int]], float]:
        """Retrouve le mot du lexicon le plus proche de query, ou None (abstention)."""
        q = query[: self.dim].to(torch.float32)
        q = q / (q.norm() + 1e-8)
        cos = q @ index_matrix.to(q.device).T
        cos1, idx = torch.max(cos, dim=0)
        conf = float(cos1.item())
        if conf < threshold:
            return None, conf
        return lexicon[int(idx.item())], conf

    @torch.no_grad()
    def decode_word(self, query: torch.Tensor) -> List[int]:
        """DÉCODEUR (génération, spec §4) : AMV -> séquence de morphèmes.

        Inverse de word_vector. Pour chaque position, le morphème dont la projection
        positionnelle matche le mieux le résidu (peeling successif). Permet de
        GÉNÉRER la forme de surface (séquence de morphèmes) depuis un vecteur concept.
        """
        q = query[: self.dim].to(torch.float32).clone()
        q = q / (q.norm() + 1e-8)
        seq = []
        prim_mat = self.prim._matrix().to(q.device)             # (P, dim)
        for t in range(self.max_len):
            proj = self.pos_proj[t].to(q.device) @ prim_mat.T   # (dim, P) projeté par position
            scores = q @ proj                                   # (P,) match par morphème
            m = int(torch.argmax(scores).item())
            seq.append(m)
            q = q - proj[:, m]
            q = q / (q.norm() + 1e-8)
        return seq
