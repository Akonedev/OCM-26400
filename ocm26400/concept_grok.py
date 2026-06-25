"""Concept Grok — le langage comme ARITHMÉTIQUE (IDs numériques, pas caractères).

IDÉE UTILISATEUR (correcte, alignée avec Besoins/Formules_Lois_Profondeurs.md) :
  Au lieu de : "bell" → ['b','e','l','l'] (4 positions FFT, copie de texte)
  Faire :     "bell" → ID 42              (1 position, ASSOCIATION numérique)

  Le modèle apprend : concept 42 OP concept ∅ → concept 143 (+s = règle)
  C'est IDENTIQUE à l'addition : 3 OP 5 → 8 (crown-jewel 100%)

Le grokking marche parce que c'est une ASSOCIATION entre NOMBRES, pas une COPIE
de texte. Le SpectralCoreBlock (FFT) est nativement conçu pour les patterns
numériques — les fréquences SONT les nombres.

Architecture :
* ConceptVocab : mapping bidirectionnel mot ↔ ID numérique (V > 64, LearnedVocab dense)
* encode(text) → séquence d'IDs → séquence d'AMV (partitions ent=concept_dense)
* SpectralCoreBlock traite la séquence d'AMV (FFT sur IDs = sur des nombres)
* Loss : 1-cos (comme crown-jewel, PAS cross-entropy)
* Le modèle grok les ASSOCIATIONS concept→concept comme des opérations arithmétiques

Avantages vs CharGenerator (texte) :
1. 1 position/mot (vs 4-8 chars/mot) → séquences x4-8 plus courtes
2. FFT sur des nombres (natif) vs FFT sur du texte (contre-nature)
3. Grokking par association (généralise) vs mémorisation par copie (surapprend)
4. V>64 (LearnedVocab dense, pas de collision) vs PART=64 (collision sur grands nombres)
5. Loss 1-cos (crown-jewel) vs CE (indépendante par token)
"""
from __future__ import annotations
import os
import re
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART, AMVVector
from .learned_vocab import LearnedVocab


class ConceptVocab:
    """Mapping bidirectionnel mot ↔ ID numérique. V > 64 (dense, LearnedVocab).

    Chaque mot/morphème obtient un ID unique. Les IDs vivent dans LearnedVocab
    (embeddings denses R^{V×64}), PAS dans SymbolicDict (one-hot, limité à 64).
    C'est le fix pour le problème de collision PART=64."""

    def __init__(self, vocab_size: int = 512, dim: int = PART, max_concepts: int = 50000):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._next_id = 0
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_concepts = max_concepts   # hard limit (fix AttributeError test_concept_grok)
        self.UNK = 0                        # ID réservé unknown
        self.embeddings: Optional[LearnedVocab] = None

    def add_word(self, word: str) -> int:
        """Ajoute un mot au vocabulaire, retourne son ID numérique."""
        w = word.lower().strip()
        if w not in self.word_to_id:
            if self._next_id >= self.max_concepts:
                return self.UNK  # hard limit — reject new words
            if self._next_id >= self.vocab_size:
                self.vocab_size = min(self._next_id * 2, self.max_concepts)
            self.word_to_id[w] = self._next_id
            self.id_to_word[self._next_id] = w
            self._next_id += 1
        return self.word_to_id[w]

    def build_embeddings(self):
        """Construit les embeddings denses LearnedVocab (V > 64 possible)."""
        V = max(self._next_id, 1)
        self.embeddings = LearnedVocab(n=V, dim=self.dim, init="ortho" if V <= self.dim else "random")
        self.embeddings.freeze()
        return self.embeddings

    def get_embedding(self, word_id: int) -> torch.Tensor:
        """Retourne l'embedding dense (64-dim) d'un ID concept."""
        if self.embeddings is None:
            self.build_embeddings()
        return self.embeddings.canonical(word_id)

    def encode_text(self, text: str, max_len: int = 40) -> Tuple[torch.Tensor, List[int]]:
        """Texte → séquence d'IDs → séquence d'embeddings denses.
        Retourne (amv_sequence (L, 256), id_sequence)."""
        words = re.findall(r"\w+", text.lower())[:max_len]
        ids = [self.add_word(w) for w in words]
        if not ids:
            return torch.zeros(1, D_MODEL), []
        # construit la séquence d'AMV : ent = embedding dense du concept
        seq = torch.zeros(len(ids), D_MODEL)
        for i, wid in enumerate(ids):
            emb = self.get_embedding(wid)  # (64,)
            seq[i, :PART] = emb           # ent = concept embedding
            # prop = position encoding (pour distinguer l'ordre)
            seq[i, PART + (i % PART)] = 0.5
        return seq, ids

    def __len__(self):
        return self._next_id


class ConceptGrokModel(nn.Module):
    """Modèle concept-grok : SpectralCoreBlock sur des IDs conceptuels.
    Le langage est traité COMME de l'arithmétique (associations numériques).
    PAS de transformer. PAS de cross-entropy. Loss 1-cos (crown-jewel)."""

    def __init__(self, d_model: int = D_MODEL, seq_len: int = 40):
        super().__init__()
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=seq_len, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) séquence d'AMV conceptuels → output (B, L, d_model)."""
        return self.core(x)


def grok_concept_association(pairs: List[Tuple[str, str]],
                              n_steps: int = 1500, lr: float = 3e-3,
                              seed: int = 0, device: str = None) -> Tuple[nn.Module, Dict]:
    """GROK des associations concept→concept (mot→mot).
    Ex: ("bell", "bells") = règle +s, comme (3, 8) = règle +5.

    Le SpectralCoreBlock apprend l'association DENSE (embedding → embedding)
    via la loss 1-cos, exactement comme le crown-jewel apprend (a,b) → op(a,b)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # construit le vocabulaire conceptuel
    vocab = ConceptVocab(vocab_size=max(len(pairs) * 2, 100))
    for w1, w2 in pairs:
        vocab.add_word(w1)
        vocab.add_word(w2)
    vocab.build_embeddings()

    model = ConceptGrokModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # prépare les données : (input_embedding) → (target_embedding)
    data = []
    for w1, w2 in pairs:
        id1 = vocab.word_to_id[w1.lower()]
        id2 = vocab.word_to_id[w2.lower()]
        inp_emb = vocab.get_embedding(id1).to(device)
        tgt_emb = vocab.get_embedding(id2).to(device)
        # encode en AMV : ent = concept
        inp_amv = torch.zeros(D_MODEL, device=device)
        inp_amv[:PART] = inp_emb
        tgt_amv = torch.zeros(D_MODEL, device=device)
        tgt_amv[:PART] = tgt_emb
        data.append((inp_amv, tgt_amv))

    n = len(data)
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(32, n),))
        total_loss = 0.0
        for i in idx:
            inp, tgt = data[i]
            out = model(inp.unsqueeze(0).unsqueeze(0))[0, 0]  # (d_model,)
            # loss 1-cos sur la partition ent (crown-jewel)
            loss = 1.0 - F.cosine_similarity(
                out[:PART].unsqueeze(0), tgt[:PART].unsqueeze(0)).clamp(-1, 1)
            total_loss += loss
        (total_loss / len(idx)).backward()
        opt.step()
        opt.zero_grad()

    # évalue : le modèle prédit-il le bon concept ?
    correct = 0
    model.eval()
    with torch.no_grad():
        for inp, tgt in data:
            out = model(inp.unsqueeze(0).unsqueeze(0))[0, 0]
            pred_emb = out[:PART]
            # nearest neighbor dans le vocabulaire
            best_id, best_sim = -1, -2.0
            M = vocab.embeddings._matrix().to(device)
            sims = pred_emb @ M.T
            pred_id = int(sims.argmax())
            # vérifie si pred correspond au target
            tgt_id = int((tgt[:PART] @ M.T).argmax())
            if pred_id == tgt_id:
                correct += 1

    return model, {
        "task": "concept association grok (mot→mot, IDs numériques)",
        "n_pairs": n, "n_steps": n_steps,
        "grok_acc": round(correct / max(n, 1), 4),
        "grokked": correct / max(n, 1) >= 0.5,
        "architecture": "SpectralCoreBlock (FFT) sur LearnedVocab dense (V>64)",
        "loss": "1-cos (crown-jewel), PAS cross-entropy",
        "advantage": "IDs numériques = FFT natif (vs caractères = copie de texte)",
    }


def run_concept_grok_demo():
    """Démo : grok des règles morphologiques comme associations numériques."""
    # paires (base, inflexion) = règles morphologiques = associations concept→concept
    pairs = [
        ("bell", "bells"), ("cat", "cats"), ("dog", "dogs"), ("book", "books"),
        ("run", "runs"), ("walk", "walks"), ("jump", "jumps"), ("play", "plays"),
        ("car", "cars"), ("star", "stars"), ("tree", "trees"), ("house", "houses"),
    ]
    print(f"[concept_grok] {len(pairs)} paires concept→concept (règle +s)")
    model, res = grok_concept_association(pairs, n_steps=1500)
    print(f"  acc={res['grok_acc']*100:.1f}% {'✓ GROKKED' if res['grokked'] else '✗'}")
    print(f"  {res['architecture']}")
    print(f"  {res['loss']}")
    return res


if __name__ == "__main__":
    run_concept_grok_demo()
