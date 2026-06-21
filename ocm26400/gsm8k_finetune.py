"""Fine-tuning GSM8K depuis le modèle PRÉ-ENTRAÎNÉ — Besoins §5 étape finale.

Après pré-training linguistique (70% masked word prediction) + grok des primitives
(83-87%), on FINE-TUNE le SpectralCoreBlock pré-entraîné sur GSM8K.

C'est le pipeline ML complet : PRE-TRAIN → GROK PRIMITIVES → FINE-TUNE.
Le modèle pré-entraîné a déjà une compréhension de la structure du langage →
le fine-tuning devrait converger plus vite qu'un modèle from-scratch.

Procédure :
1. Charger le SpectralCoreBlock pré-entraîné (language_pretrain).
2. Fine-tuner sur GSM8K train (DOSC curriculum : Phase 1 1-step → Phase 2 2-step).
3. Évaluer sur GSM8K test.
"""
from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART
from .reasoner import ReasonerBlock, encode_input
from .verifier import SymbolicDict
from .gsm8k_bench import load_gsm8k, extract_answer
from .gsm8k_supervised import HERE
from .gsm8k_seq2seq import ACTIONS, ACT_TO_IDX, MAX_DECOD, _build_vocab, _encode_q, _trace_to_actions, _actions_to_value
from .language_pretrain import pretrain_masked_word


class FinetuneModel(nn.Module):
    """SpectralCoreBlock pré-entraîné + tête de raisonnement GSM8K."""

    def __init__(self, vocab_size: int, d_model: int = 256, pretrained_blk=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Utilise le block PRÉ-ENTRAÎNÉ si fourni (transfer learning)
        self.core = pretrained_blk if pretrained_blk is not None else SpectralCoreBlock(d_model=d_model, seq_len=60)
        self.dec_emb = nn.Embedding(len(ACTIONS), d_model)
        self.dec_core = SpectralCoreBlock(d_model=d_model, seq_len=MAX_DECOD)
        self.ctx = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, len(ACTIONS))

    def forward(self, src, tgt_in):
        e = self.emb(src)
        enc = self.core(e)
        ctx = enc.mean(dim=1)
        t = self.dec_emb(tgt_in) + self.ctx(ctx).unsqueeze(1)
        t = self.dec_core(t)
        return self.out(self.norm(t))


def run_finetuned_gsm8k(n_test: int = 200, n_train: int = 3000, n_steps: int = 1500,
                         use_pretrain: bool = True, device: str = None) -> Dict:
    """Fine-tune sur GSM8K depuis le modèle pré-entraîné (ou from-scratch)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # 1. PRÉ-TRAIN (optionnel)
    pretrained_blk = None
    if use_pretrain:
        print("[finetune] Chargement du modèle pré-entraîné...")
        pretrained_blk, pt_info = pretrain_masked_word(n_sentences=300, n_steps=1000, device=device)
        # extraire le SpectralCoreBlock du ReasonerBlock
        if hasattr(pretrained_blk, 'block'):
            pretrained_blk = pretrained_blk.block
        print(f"  pré-training : {pt_info.get('masked_word_acc', '?')}")

    # 2. FINE-TUNE sur GSM8K
    probs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data",
              "gsm8k_train.jsonl"))][:n_train]
    vocab = _build_vocab([p["question"] for p in probs])

    model = FinetuneModel(len(vocab), d_model=256, pretrained_blk=pretrained_blk).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    X = torch.tensor([_encode_q(p["question"], vocab) for p in probs]).to(device)
    targets = []
    for p in probs:
        a = _trace_to_actions(p["question"], p["answer"]) or [ACT_TO_IDX["PAD"]] * MAX_DECOD
        targets.append(a)
    T = torch.tensor(targets).to(device)
    tgt_in, tgt_out = T[:, :-1], T[:, 1:]
    n = len(probs)

    print(f"[finetune] Fine-tuning sur {n} problèmes GSM8K ({n_steps} steps)...")
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(64, n),))
        logits = model(X[idx], tgt_in[idx])
        loss = F.cross_entropy(logits.reshape(-1, len(ACTIONS)), tgt_out[idx].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # 3. ÉVALUATION
    model.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    with torch.no_grad():
        for p in tests:
            gold = extract_answer(p["answer"])
            if gold is None:
                continue
            n_total += 1
            src = torch.tensor([_encode_q(p["question"], vocab)]).to(device)
            cur = torch.tensor([[ACT_TO_IDX["START"]]]).to(device)
            actions = []
            for _ in range(MAX_DECOD - 1):
                logits = model(src, cur)
                nxt = int(logits[0, -1].argmax())
                if ACTIONS[nxt] == "END":
                    break
                actions.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]]).to(device)], dim=1)
            from .gsm8k_bench import extract_numbers
            qnums = extract_numbers(p["question"])
            pred = _actions_to_value(actions, qnums)
            if pred is None:
                continue
            n_attempted += 1
            if abs(pred - gold) < 1e-6:
                n_correct += 1

    return {
        "dataset": "GSM8K officiel (FINE-TUNED depuis pré-training)",
        "use_pretrain": use_pretrain,
        "pretrain_acc": pt_info.get("masked_word_acc", 0) if use_pretrain else 0,
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "archi": "SpectralCoreBlock pré-entraîné + fine-tuning GSM8K (PAS de transformer)",
        "procedure": "PRE-TRAIN (langage) → FINE-TUNE (GSM8K) — Besoins §5",
    }


if __name__ == "__main__":
    print("=== FROM SCRATCH (sans pré-training) ===")
    rep_scratch = run_finetuned_gsm8k(n_test=200, n_train=3000, n_steps=1500,
                                       use_pretrain=False)
    print(f"  {rep_scratch['n_correct']}/{rep_scratch['n_attempted']} = "
          f"{rep_scratch['accuracy']*100:.1f}%")

    print("\n=== FINE-TUNED (avec pré-training 70%) ===")
    rep_ft = run_finetuned_gsm8k(n_test=200, n_train=3000, n_steps=1500,
                                  use_pretrain=True)
    print(f"  {rep_ft['n_correct']}/{rep_ft['n_attempted']} = "
          f"{rep_ft['accuracy']*100:.1f}% (pré-training: {rep_ft['pretrain_acc']})")
