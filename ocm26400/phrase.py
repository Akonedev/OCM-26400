"""Composition de PHRASES — du mot à la phrase (OCM-26400, cahier des charges §TESTS).

Le cahier des charges exige : 'génération de phrases', 'compréhension du sens de la
phrase', 'régénérer la phrase avec des synonymes', 'régénérer une phrase différente
ayant le même sens'. Le modèle traitait les MOTS — ici on étend au niveau PHRASE.

Une PHRASE = séquence de mots, composée via le NOYAU SPECTRAL (FFT mélange la séquence
en une seule passe → sens de la phrase). C'est l'extension naturelle du compositionnel
du mot à la phrase : la compréhension émerge de la composition.

* PhraseComposer : compose (mots → AMV phrase via spectral) + decode (AMV → mots).
* phrase_similarity : similarité entre phrases (pour synonymes/nuances).
* regenerate_with_synonyms : régénère une phrase avec des mots synonymes.

Le noyau spectral mélange les mots d'une phrase en O(L log L) — c'est sa force native.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

from .amv import D_MODEL, PART
from .spectral_core import SpectralCoreBlock
from .verifier import SymbolicDict


class PhraseComposer(nn.Module):
    """Compose des phrases via le noyau spectral : mots → AMV phrase."""

    def __init__(self, d_model: int = D_MODEL, dict_n: int = 64):
        super().__init__()
        self.d_model = d_model
        self.dict_n = dict_n
        # projection mot (ent 64-d) → AMV complet (256-d)
        self.word_proj = nn.Linear(PART, d_model)
        # noyau spectral unifié (mélange la séquence de mots)
        self.core = SpectralCoreBlock(d_model=d_model)
        # projection inverse AMV → ent (pour décoder chaque mot)
        self.amv_to_ent = nn.Linear(d_model, PART)

    def compose(self, word_ids: List[int], d: SymbolicDict) -> torch.Tensor:
        """Compose une phrase : liste de mots → AMV phrase (le spectral mélange)."""
        if not word_ids:
            return torch.zeros(self.d_model)
        # encode chaque mot → ent (64-d) → projette → AMV (256-d)
        word_ents = torch.stack([d.canonical(w) for w in word_ids])  # (L, 64)
        word_amvs = self.word_proj(word_ents).unsqueeze(0)           # (1, L, 256)
        # le noyau spectral mélange la séquence → sens de la phrase
        phrase_amv = self.core(word_amvs)                            # (1, L, 256)
        return phrase_amv[0, -1]                                     # dernier = sens global

    def decode_words(self, phrase_amv: torch.Tensor, d: SymbolicDict,
                     max_words: int = 5) -> List[int]:
        """Décode un AMV phrase → mots (matching contre le dictionnaire)."""
        amv = phrase_amv.unsqueeze(0).unsqueeze(0)                   # (1, 1, 256)
        ent = self.amv_to_ent(amv[0, 0])                             # (64,)
        # match contre le dictionnaire
        M = d._matrix()                                              # (n, 64)
        cos = ent / (ent.norm() + 1e-8) @ (M / (M.norm(dim=1, keepdim=True) + 1e-8)).T
        top = torch.topk(cos, min(max_words, d.n))
        return [int(i) for i in top.indices]


def phrase_similarity(amv_a: torch.Tensor, amv_b: torch.Tensor) -> float:
    """Similarité cosinus entre deux phrases (pour synonymes/nuances)."""
    a = amv_a / (amv_a.norm() + 1e-8)
    b = amv_b / (amv_b.norm() + 1e-8)
    return float((a @ b).item())


def regenerate_with_synonyms(word_ids: List[int], synonyms: dict,
                              composer: PhraseComposer, d: SymbolicDict) -> Tuple[List[int], float]:
    """Régénère une phrase en remplaçant des mots par leurs synonymes.
    Retourne (nouveaux_mots, similarité_avec_l_original)."""
    original_amv = composer.compose(word_ids, d)
    # remplace chaque mot par son synonyme (si disponible)
    new_words = [synonyms.get(w, w) for w in word_ids]
    new_amv = composer.compose(new_words, d)
    sim = phrase_similarity(original_amv, new_amv)
    return new_words, sim
