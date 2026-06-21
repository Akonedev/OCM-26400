"""ConceptVocab — IDs numériques pour concepts/mots/nombres — l'approche crown-jewel.

L'utilisateur dit : 'tout convertir en IDs numériques (math) plutôt qu'en caractères.
Le CharGenerator (CE sur caractères) est fondamentalement contre-productif pour le
grokking. La bonne implémentation = ConceptVocab (IDs) + SpectralCoreBlock (FFT sur IDs)
+ loss 1-cos (comme crown-jewel).'

ConceptVocab : mappe chaque concept (mot, nombre, opération, ponctuation) à un ID
numérique unique. Le SpectralCoreBlock traite ces IDs via LearnedVocab dense (1-cos),
exactement comme le crown-jewel traite les nombres 0-10.

C'est l'approche prescrite par le Besoins.md et validée par le crown-jewel (100%).
Le dense number-binding à 66% (vs 19% one-hot) confirme que ça marche.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART
from .reasoner import ReasonerBlock
from .learned_vocab import LearnedVocab


class ConceptVocab:
    """Vocabulaire de concepts : mappe mots/nombres/ops → IDs numériques uniques.
    Chaque ID obtient un vecteur dense dans LearnedVocab (pas de collision).
    C'est l'équivalent du SymbolicDict pour Z_11, mais avec V>>64 concepts."""

    def __init__(self, max_concepts: int = 10000):
        self.max_concepts = max_concepts
        self._id2concept: List[str] = []
        self._concept2id: Dict[str, int] = {}

        # IDs réservés
        self.PAD = 0
        self.START = 1
        self.END = 2
        self.UNK = 3
        for c in ["<pad>", "<start>", "<end>", "<unk>"]:
            self._id2concept.append(c)
            self._concept2id[c] = len(self._id2concept) - 1

        # IDs spéciaux : nombres 0-9999 (IDs 4..10003)
        for n in range(10000):
            c = f"NUM_{n}"
            self._id2concept.append(c)
            self._concept2id[c] = len(self._id2concept) - 1

        # IDs spéciaux : opérations (IDs 10004..10007)
        for op in ["OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV"]:
            self._id2concept.append(op)
            self._concept2id[op] = len(self._id2concept) - 1

    @property
    def size(self) -> int:
        return len(self._id2concept)

    def add_concept(self, concept: str) -> int:
        """Ajoute un concept (mot) au vocabulaire s'il n'existe pas."""
        if concept not in self._concept2id:
            if len(self._id2concept) >= self.max_concepts:
                return self.UNK
            self._id2concept.append(concept)
            self._concept2id[concept] = len(self._id2concept) - 1
        return self._concept2id[concept]

    def get_id(self, concept: str) -> int:
        return self._concept2id.get(concept, self.UNK)

    def get_num_id(self, n: int) -> int:
        """ID d'un nombre (0-9999)."""
        if 0 <= n < 10000:
            return 4 + n  # offset des nombres
        # nombres > 9999 : ajouter comme concept
        return self.add_concept(f"NUM_{n}")

    def get_op_id(self, op: str) -> int:
        """ID d'une opération."""
        return self._concept2id.get(f"OP_{op}", self.UNK)

    def text_to_ids(self, text: str) -> List[int]:
        """Convertit un texte en liste d'IDs numériques.
        Les nombres deviennent NUM IDs, les mots deviennent des concept IDs."""
        ids = [self.START]
        for token in re.findall(r"\d+|[a-zA-Z]+|[+\-*/=]", text.lower()):
            if token.isdigit():
                ids.append(self.get_num_id(int(token)))
            elif token in "+-*/":
                op_map = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}
                ids.append(self.get_op_id(op_map[token]))
            elif token == "=":
                ids.append(self.add_concept("EQ"))
            else:
                ids.append(self.add_concept(token))
        ids.append(self.END)
        return ids

    def ids_to_text(self, ids: List[int]) -> str:
        """Reconstruit le texte depuis les IDs."""
        tokens = []
        for i in ids:
            if i < len(self._id2concept):
                c = self._id2concept[i]
                if c.startswith("NUM_"):
                    tokens.append(c[4:])
                elif c.startswith("OP_"):
                    ops = {"ADD": "+", "SUB": "-", "MUL": "*", "DIV": "/"}
                    tokens.append(ops.get(c[3:], "?"))
                elif c.startswith("<"):
                    continue  # skip special tokens
                else:
                    tokens.append(c)
        return " ".join(tokens)


class ConceptModel(nn.Module):
    """Modèle conceptuel : ConceptVocab IDs → SpectralCoreBlock → IDs.
    Utilise LearnedVocab dense (pas de collision) + loss 1-cos (crown-jewel).
    PAS de transformer. PAS de CE loss. C'est l'approche prescrite."""

    def __init__(self, vocab_size: int, d_model: int = D_MODEL, seq_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        # LearnedVocab dense : chaque concept → vecteur unique R^64 (pas de collision)
        self.embeddings = LearnedVocab(n=vocab_size, dim=PART, init="random", seed=0)
        self.embeddings.freeze()  # codebook fixe (comme SymbolicDict mais dense)
        # SpectralCoreBlock (FFT, MODEL UNIFIÉ)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=seq_len, bidirectional=True)
        # tête de décodage : projecte vers le codebook (1-cos loss)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """x_ids: (B, L) IDs numériques → (B, L, d_model) via LearnedVocab dense.
        Chaque ID obtient un vecteur dense unique. Le SpectralCoreBlock mixe."""
        B, L = x_ids.shape
        # encode chaque ID en vecteur dense (lookup dans LearnedVocab)
        M = self.embeddings._matrix()  # (V, 64)
        # one-hot → matmul = lookup
        x_oh = F.one_hot(x_ids, self.vocab_size).float()  # (B, L, V)
        x_dense = x_oh @ M  # (B, L, 64) — chaque ID → vecteur dense unique
        # pad à 256 dims (4 partitions)
        full = torch.zeros(B, L, D_MODEL, device=x_ids.device)
        full[:, :, :PART] = x_dense
        # SpectralCoreBlock
        out = self.core(full)
        return self.norm(out)

    def loss_1cos(self, x_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Loss 1-cos (crown-jewel) : compare output.ent avec l'embedding du target.
        PAS de cross-entropy. C'est l'approche prescrite par le Besoins."""
        out = self.forward(x_ids)  # (B, L, 256)
        M = self.embeddings._matrix()  # (V, 64)
        # target embeddings
        target_oh = F.one_hot(target_ids, self.vocab_size).float()
        target_dense = target_oh @ M  # (B, L, 64)
        # 1 - cos(output[:64], target_dense)
        out_ent = out[:, :, :PART]  # (B, L, 64)
        cos = F.cosine_similarity(out_ent, target_dense, dim=-1).clamp(-1, 1)
        return (1 - cos).mean()


def train_concept_model(model: ConceptModel, train_ids: List[List[int]],
                         target_ids: List[List[int]], n_steps: int = 2000,
                         lr: float = 3e-3, device: str = "cpu") -> List[float]:
    """Entraîne le ConceptModel avec loss 1-cos (crown-jewel)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    n = len(train_ids)
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(32, n),))
        x_batch = torch.tensor([train_ids[i] for i in idx], device=device)
        t_batch = torch.tensor([target_ids[i] for i in idx], device=device)
        loss = model.loss_1cos(x_batch, t_batch)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            history.append(float(loss.item()))
    return history


if __name__ == "__main__":
    # démo : ConceptVocab encode un texte GSM8K en IDs numériques
    cv = ConceptVocab(max_concepts=10000)
    text = "Janet has 16 eggs and eats 3 for breakfast"
    ids = cv.text_to_ids(text)
    print(f"[concept_vocab] '{text}'")
    print(f"  IDs: {ids}")
    print(f"  decoded: '{cv.ids_to_text(ids)}'")
    print(f"  vocab size: {cv.size}")
    print(f"  NUM_16 = ID {cv.get_num_id(16)}, NUM_3 = ID {cv.get_num_id(3)}")
    print(f"\n  ConceptModel: LearnedVocab dense + SpectralCoreBlock + 1-cos (crown-jewel)")
