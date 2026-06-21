"""Seq2Seq TRANSFORMER (attention) pour GSM8K — upgrade de capacité.

Diagnostic : le GRU seq2seq plateaut à ~3% (capacity bottleneck). On remplace par un
VRAI transformer (self-attention encoder + cross-attention decoder) — l'architecture qui
permet le NL→binding complexe. COPY mechanism conservé (number-binding).

* Encoder transformer : self-attention sur les tokens de la question.
* Decoder transformer : cross-attention sur l'encoder + génération d'actions COPY/OP.
* Entraîné sur GSM8K train (7K), mesuré sur test.

C'est l'investissement architectural demandé. Mesure réelle si l'attention > GRU sur
le NL→CoT de GSM8K.
"""
from __future__ import annotations
import json
import math
import os
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gsm8k_bench import load_gsm8k, extract_numbers, extract_answer
from .gsm8k_supervised import HERE
from .gsm8k_seq2seq import (
    ACTIONS, ACT_TO_IDX, MAX_DECOD, MAX_NUMS,
    _build_vocab, _encode_q, _trace_to_actions, _actions_to_value,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerCoT(nn.Module):
    """Encoder transformer (question) + decoder transformer (actions COPY/OP)."""

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 3, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(len(ACTIONS), d_model)
        self.pos = PositionalEncoding(d_model, max_len=60)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.out = nn.Linear(d_model, len(ACTIONS))

    def forward(self, src, tgt_in, src_key_padding_mask=None):
        src_emb = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        tgt_emb = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        # masque causal decoder (autoregressif)
        T = tgt_in.size(1)
        causal = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        dec_out = self.decoder(tgt_emb, memory, tgt_mask=causal,
                               memory_key_padding_mask=src_key_padding_mask)
        return self.out(dec_out)


def train_transformer(n_train: int = 5000, n_steps: int = 2000, lr: float = 3e-4,
                      seed: int = 0, device: str = None) -> tuple:
    """Entraîne le transformer CoT. lr plus bas (transformer) + warmup implicite."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    probs = [json.loads(l) for l in open(os.path.join(HERE, "..", "data",
              "gsm8k_train.jsonl"))][:n_train]
    vocab = _build_vocab([p["question"] for p in probs])
    model = TransformerCoT(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

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
def predict_cot_transformer(model: TransformerCoT, vocab: dict, question: str,
                            device: str = "cpu") -> Optional[float]:
    """Génération greedy autoregressive → exécute les actions → valeur."""
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


def run_transformer_gsm8k(n_test: int = 200, n_train: int = 5000, n_steps: int = 2000) -> dict:
    """Évalue le transformer CoT sur GSM8K officiel."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = train_transformer(n_train, n_steps, device=device)
    model.eval()
    tests = load_gsm8k(n=n_test)
    n_correct = n_attempted = n_total = 0
    for p in tests:
        gold = extract_answer(p["answer"])
        if gold is None:
            continue
        n_total += 1
        pred = predict_cot_transformer(model, vocab, p["question"], device)
        if pred is None:
            continue
        n_attempted += 1
        if abs(pred - gold) < 1e-6:
            n_correct += 1
    return {
        "dataset": "GSM8K officiel (TRANSFORMER attention NL→CoT)",
        "n_test": n_total, "n_attempted": n_attempted, "n_correct": n_correct,
        "accuracy_on_attempted": round(n_correct / max(n_attempted, 1), 4),
        "coverage": round(n_attempted / max(n_total, 1), 4),
        "archi": "transformer enc+dec, d=128, 4 heads, 3 layers",
        "vs": {"gru_seq2seq": "3.2%", "rule_based": "3.0%"},
    }


if __name__ == "__main__":
    rep = run_transformer_gsm8k(n_test=200, n_train=5000, n_steps=2000)
    print(f"[transformer] {rep['dataset']}")
    print(f"  {rep['n_correct']}/{rep['n_attempted']} = {rep['accuracy_on_attempted']*100:.1f}% "
          f"(couv {rep['coverage']*100:.0f}%, archi={rep['archi']})")
