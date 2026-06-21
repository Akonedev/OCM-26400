"""Number-binding DENSE via LearnedVocab (V>64) — résout la collision PART=64.

Le problème racine : PART=64 provoque des collisions (5 mod 64 = 69 mod 64).
La solution : LearnedVocab (P2) avec V=1000+ positions DENSE (pas de collision).

Chaque nombre GSM8K (0-10000+) obtient un vecteur dense UNIQUE dans R^64.
Le SpectralCoreBlock apprend (a_dense, op) → result_dense via 1-cos.
La densité évite les collisions → le grokking peut avoir lieu.

C'est le passage one-hot → dense que P2 a préparé (learned_vocab.py).
"""
from __future__ import annotations
import json
import os
import re
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART
from .reasoner import ReasonerBlock
from .learned_vocab import LearnedVocab


def extract_bindings_dense(n_problems: int = 5000) -> List[dict]:
    """Extrait les bindings (a, op, b, result) depuis le CoT GSM8K.
    Utilise les valeurs RÉELLES (pas modulo) pour LearnedVocab."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "gsm8k_train.jsonl")
    if not os.path.exists(path):
        return []
    probs = [json.loads(l) for l in open(path)][:n_problems]
    bindings = []
    for p in probs:
        exprs = re.findall(r"<<([\d.\s]+)([+\-*/])([\d.\s]+)=([\d.]+)>>", p["answer"])
        for a_str, op_ch, b_str, result_str in exprs:
            try:
                a, b, result = int(float(a_str)), int(float(b_str)), int(float(result_str))
                if 0 <= a <= 9999 and 0 <= b <= 9999 and 0 <= result <= 99999:
                    bindings.append({"a": a, "op": op_ch, "b": b, "result": result})
            except (ValueError, TypeError):
                continue
    return bindings


# Mappe op → index (4 ops : +, -, *, /)
OP_TO_IDX = {"+": 0, "-": 1, "*": 2, "/": 3}
N_OPS = 4


def grok_dense_number_binding(n_problems: int = 5000, vocab_size: int = 10000,
                                n_steps: int = 3000, lr: float = 3e-3,
                                device: str = None) -> Tuple[nn.Module, Dict]:
    """GROK le number-binding avec LearnedVocab DENSE (V=10000, pas de collision).
    encode(a_dense, b_dense) → result_dense via SpectralCoreBlock (1-cos loss)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    bindings = extract_bindings_dense(n_problems)
    if len(bindings) < 100:
        return ReasonerBlock().to(device), {"error": "pas assez de bindings"}

    # LearnedVocab : V=10000 nombres + 4 ops = 10004 entrées (DENSE, pas de collision)
    # Chaque nombre 0-9999 obtient un vecteur dense unique dans R^64
    # Les 4 ops obtiennent aussi un vecteur dense
    total_vocab = vocab_size + N_OPS
    vocab = LearnedVocab(n=total_vocab, dim=PART, init="random", seed=0).to(device)
    vocab.freeze()  # codebook fixe (analogue du SymbolicDict mais dense)

    # le SpectralCoreBlock
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)

    # split train/test
    n = len(bindings)
    train = bindings[:int(n * 0.8)]
    test = bindings[int(n * 0.8):]

    print(f"  {len(train)} train, {len(test)} test bindings (V={total_vocab}, dense)")

    # encodage : a → ent(64), b → prop(64), op → première dim de op(64)
    def encode(a_val: int, b_val: int, op_idx: int) -> torch.Tensor:
        v = torch.zeros(D_MODEL, device=device)
        v[:PART] = vocab.canonical(a_val).to(device)
        v[PART:2*PART] = vocab.canonical(b_val).to(device)
        v[2*PART + op_idx] = 1.0
        return v

    def encode_result(result_val: int) -> torch.Tensor:
        return vocab.canonical(result_val).to(device)

    for step in range(n_steps):
        idx = torch.randint(0, len(train), (min(64, len(train)),))
        total_loss = 0.0
        for i in idx:
            b = train[i]
            x = encode(b["a"], b["b"], OP_TO_IDX[b["op"]]).unsqueeze(0)
            out = blk(x)[0]
            target = encode_result(b["result"])
            # loss : 1 - cos(output.ent, target_dense)
            loss = 1.0 - F.cosine_similarity(
                out[:PART].unsqueeze(0), target.unsqueeze(0)).clamp(-1, 1)
            total_loss += loss
        (total_loss / len(idx)).backward()
        opt.step()
        opt.zero_grad()

    # évalue : decode = plus proche voisin dans le codebook
    M = vocab._matrix()  # (V, 64)
    correct = 0
    near = 0  # within top-5
    blk.eval()
    with torch.no_grad():
        for b in test[:200]:
            x = encode(b["a"], b["b"], OP_TO_IDX[b["op"]]).unsqueeze(0)
            out = blk(x)[0]
            out_ent = out[:PART]
            # decode : cosinus avec tous les vecteurs du codebook
            sims = M @ out_ent  # (V,)
            # on cherche dans les nombres (pas les ops) : indices 0..vocab_size-1
            top5 = sims[:vocab_size].topk(5).indices.tolist()
            pred = top5[0]
            if pred == b["result"]:
                correct += 1
            if b["result"] in top5:
                near += 1

    acc = correct / min(200, len(test))
    top5_acc = near / min(200, len(test))

    return blk, {
        "primitive": "number-binding DENSE (LearnedVocab V=10000)",
        "n_train": len(train), "n_test": min(200, len(test)),
        "top1_acc": round(acc, 4),
        "top5_acc": round(top5_acc, 4),
        "vocab_size": total_vocab,
        "grokked": acc >= 0.1,  # V=10000, chance = 0.01%
        "collision_free": True,
        "procedure": "LearnedVocab dense + SpectralCoreBlock + 1-cos, données CoT GSM8K RÉELLES",
        "note": "Pas de collision (dense V=10000 vs one-hot PART=64). Chaque nombre a un vecteur unique.",
    }


if __name__ == "__main__":
    print("[dense_binding] Number-binding DENSE (LearnedVocab V=10000)...")
    blk, res = grok_dense_number_binding(n_problems=5000, vocab_size=10000, n_steps=3000)
    if "error" not in res:
        print(f"  top1: {res['top1_acc']*100:.1f}% | top5: {res['top5_acc']*100:.1f}%")
        print(f"  V={res['vocab_size']} | {res['procedure']}")
        print(f"  collision-free: {res['collision_free']}")
    else:
        print(f"  ERROR: {res['error']}")
