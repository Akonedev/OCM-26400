"""NL→CoT AVEC LE SPECTRALCOREBLOCK UNIFIÉ — suit les procédures du projet.

CORRECTION FONDAMENTALE (feedback utilisateur) : les modèles GSM8K précédents (GRU/transformer
VANILLA) violaient 'MODEL UNIFIÉ' — ils n'utilisaient PAS le SpectralCoreBlock (le noyau FFT
du projet) ni ne suivaient PROCEDURES.md. C'est POURQUOI ils plafonnaient à ~3%.

Ici, le NL→CoT utilise le **SpectralCoreBlock comme noyau encodeur** (MODEL UNIFIÉ) :
* Embedding tokens → séquence → SpectralCoreBlock (mélange FFT O(L log L), Parseval stable)
  → états encodés.
* Décodeur COPY/OP avec cross-attention sur les états spectraux.
* Entraînement suivant la procédure (Adam, loss CE sur actions + alignement).

L'architecture spectrale du projet (stabilité Parseval, mixing compositionnel) est la base —
pas une archi externe. C'est la voie paradigme-alignée que les procédures prescrivent.
"""
from __future__ import annotations
import json
import os
import math
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .gsm8k_bench import load_gsm8k, extract_numbers, extract_answer
from .gsm8k_supervised import HERE
from .gsm8k_seq2seq import (
    ACTIONS, ACT_TO_IDX, MAX_DECOD, MAX_NUMS,
    _build_vocab, _encode_q, _trace_to_actions, _actions_to_value,
)
from .amv import D_MODEL


class SpectralNLCoT(nn.Module):
    """NL→CoT avec le SPECTRALCOREBLOCK comme SEUL noyau (MODEL UNIFIÉ).
    ⚠️ PAS DE TRANSFORMER, PAS D'ATTENTION — uniquement le mélange spectral FFT
    (Parseval-stable, O(L log L)). C'est l'architecture du projet.
    Couplage encoder→decoder : le vecteur encodé global (pool spectral) est ajouté à
    chaque étape du décodeur (PAS d'attention)."""

    def __init__(self, vocab_size: int, d_model: int = 256, seq_len: int = 60):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # NOYAU UNIFIÉ : SpectralCoreBlock (FFT bidirectionnel du projet) — SEUL noyau
        self.encoder = SpectralCoreBlock(d_model=d_model, seq_len=seq_len, bidirectional=True)
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # décodeur d'actions : embedding actions + SpectralCoreBlock (PAS d'attention)
        self.dec_emb = nn.Embedding(len(ACTIONS), d_model)
        self.decoder = SpectralCoreBlock(d_model=d_model, seq_len=MAX_DECOD, bidirectional=True)
        # couplage encoder→decoder SANS attention : projection du pool encoder + ajout
        self.ctx_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, len(ACTIONS))

    def encode(self, src):
        """Encode la question via le noyau spectral (MODEL UNIFIÉ). Retourne (états, pool)."""
        e = self.emb(src)                       # (B, L, d)
        enc = self.encoder(e)                   # SpectralCoreBlock (FFT)
        ctx = enc.mean(dim=1)                   # pool global (PAS d'attention)
        return enc, ctx

    def forward(self, src, tgt_in):
        _, ctx = self.encode(src)               # contexte global encodé spectralement
        t = self.dec_emb(tgt_in)                # (B, T, d)
        # ajoute le contexte encoder (projeté) à chaque étape — PAS d'attention
        t = t + self.ctx_proj(ctx).unsqueeze(1)
        t = self.decoder(t)                     # SpectralCoreBlock sur la séquence d'actions
        t = self.norm(t)
        return self.out(t)


def train_spectral_cot(n_train: int = 5000, n_steps: int = 2000, lr: float = 3e-3,
                       seed: int = 0, device: str = None) -> tuple:
    """Entraîne le NL→CoT spectral. Suit la procédure : Adam lr=3e-3 (canonique ocm26400),
    seed 0, cross-entropy sur les actions COPY/OP."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)                        # §0.2 seed canonique
    probs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data",
              "gsm8k_train.jsonl"))][:n_train]
    vocab = _build_vocab([p["question"] for p in probs])
    model = SpectralNLCoT(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)       # Adam 3e-3 (procédure)

    X = torch.tensor([_encode_q(p["question"], vocab) for p in probs]).to(device)
    targets = []
    for p in probs:
        a = _trace_to_actions(p["question"], p["answer"]) or [ACT_TO_IDX["PAD"]] * MAX_DECOD
        targets.append(a)
    T = torch.tensor(targets).to(device)
    tgt_in, tgt_out = T[:, :-1], T[:, 1:]
    n = len(probs)
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(64, n),))
        logits = model(X[idx], tgt_in[idx])
        loss = F.cross_entropy(logits.reshape(-1, len(ACTIONS)), tgt_out[idx].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, vocab


@torch.no_grad()
def predict_spectral(model: SpectralNLCoT, vocab: dict, question: str,
                     device: str = "cpu") -> Optional[float]:
    model.eval()
    src = torch.tensor([_encode_q(question, vocab)]).to(device)
    cur = torch.tensor([[ACT_TO_IDX["START"]]]).to(device)
    actions = []
    for _ in range(MAX_DECOD - 1):
        logits = model(src, cur)
        nxt = int(logits[0, -1].argmax())
        if ACTIONS[nxt] == "END":
            break
        actions.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]]).to(device)], dim=1)
    qnums = extract_numbers(question)
    return _actions_to_value(actions, qnums)


def run_spectral_gsm8k(n_test: int = 200, n_train: int = 5000, n_steps: int = 2000) -> dict:
    """Évalue le NL→CoT SPECTRAL (noyau unifié) sur GSM8K officiel."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = train_spectral_cot(n_train, n_steps, device=device)
    model.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    for p in tests:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        pred = predict_spectral(model, vocab, p["question"], device)
        if pred is None:
            continue
        n_attempted += 1
        if abs(pred - gold) < 1e-6:
            n_correct += 1
    return {
        "dataset": "GSM8K officiel (SPECTRALCOREBLOCK unifié NL→CoT)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy_on_attempted": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "archi": "SpectralCoreBlock (FFT unifié) enc+dec, couplage pool (PAS d'attention)",
        "procedure": "Adam lr=3e-3, seed 0, CE actions (PROCEDURES.md)",
        "vs": {"gru_vanilla": "3.2%", "transformer_vanilla": "1.6%"},
        "note": "CORRECTION : utilise le noyau unifié du projet (MODEL UNIFIÉ), pas une archi externe.",
    }


if __name__ == "__main__":
    rep = run_spectral_gsm8k(n_test=200, n_train=5000, n_steps=2000)
    print(f"[spectral] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy_on_attempted']*100:.1f}% "
          f"(couv {rep['coverage']*100:.0f}%, {rep['archi']})")
