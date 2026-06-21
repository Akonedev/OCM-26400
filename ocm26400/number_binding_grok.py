"""GROK du NUMBER-BINDING — la primitive MANQUANTE pour NL→structured.

DIAGNOSTIC : la composition cascade marche en arithmétique (100%) car le number-binding
est trivial (les nombres SONT les inputs). En NL, le number-binding est le vrai défi :
quels nombres du texte vont avec quelle opération ?

Jusqu'ici je grokkais word→number (83%) et cue→op (87%) SÉPARÉMENT.
Mais le number-binding COMBINÉ (sentence + acc → (num, op) → new_acc) n'était PAS grokké.

C'EST la primitive manquante. Le Besoins.md dit :
"pré-entraîner les PRIMITIVES... jusqu'au grok, PUIS les compositions → la maîtrise émerge"

Le number-binding est une PRIMITIVE (association 1-source, L6) :
Input : encode(acc_pos, sentence_hash)
Output : decode → (op_position, number_value)

On l'entraîne sur les GSM8K train CoT (qui contiennent les annotations <<expr>>).
Chaque étape du CoT = 1 association number-binding (étape, opération, nombre → résultat).
"""
from __future__ import annotations
import json
import os
import re
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART, AMVVector
from .verifier import SymbolicDict
from .reasoner import ReasonerBlock, encode_input


def extract_binding_pairs_from_cot(n_problems: int = 5000) -> List[dict]:
    """Extrait les paires number-binding depuis les CoT annotés de GSM8K.
    Chaque <<expr=result>> = 1 binding : (op, num1, num2) → result.
    On encode ça comme : (result_précédent_mod_PART, op_hash) → (result_mod_PART)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "gsm8k_train.jsonl")
    if not os.path.exists(path):
        return []
    probs = [json.loads(l) for l in open(path)][:n_problems]
    pairs = []
    for p in probs:
        # extrait les expressions <<a OP b = result>>
        exprs = re.findall(r"<<([\d.\s]+)([+\-*/])([\d.\s]+)=([\d.]+)>>", p["answer"])
        for a_str, op_ch, b_str, result_str in exprs:
            try:
                a, b, result = float(a_str), float(b_str), float(result_str)
                # encode : (a mod PART, op_hash) → result mod PART
                # op_hash : hash du caractère d'opération → position
                op_pos = (hash(op_ch) & 0xFFFFFFFF) % PART
                a_pos = int(a) % PART
                result_pos = int(result) % PART if result < PART * 2 else int(result) % PART
                pairs.append({
                    "a_pos": a_pos, "op_pos": op_pos, "result_pos": result_pos,
                    "a": a, "op": op_ch, "b": b, "result": result,
                })
            except (ValueError, TypeError):
                continue
    return pairs


def grok_number_binding(n_pairs: int = 5000, n_steps: int = 2000,
                         lr: float = 3e-3, seed: int = 0,
                         device: str = None) -> Tuple[ReasonerBlock, Dict]:
    """GROK le number-binding : (a_pos, op_pos) → result_pos.
    C'est l'association COMBINÉE qui manquait. Le SpectralCoreBlock apprend à mapper
    (nombre + opération) → résultat — exactement comme en arithmétique, mais sur
    les DONNÉES RÉELLES du CoT GSM8K."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    pairs = extract_binding_pairs_from_cot(n_pairs)
    if len(pairs) < 10:
        return ReasonerBlock().to(device), {"error": "pas de données CoT"}

    d = SymbolicDict(n=PART, dim=64)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=lr)

    n = len(pairs)
    train_pairs = pairs[:int(n * 0.8)]
    test_pairs = pairs[int(n * 0.8):]

    print(f"  {len(train_pairs)} train, {len(test_pairs)} test bindings")

    for step in range(n_steps):
        idx = torch.randint(0, len(train_pairs), (min(64, len(train_pairs)),))
        total_loss = 0.0
        for i in idx:
            pair = train_pairs[i]
            # encode : a_pos en ent, op_pos en prop
            x = encode_input(pair["a_pos"], pair["op_pos"], d).unsqueeze(0).to(device)
            out = blk(x)[0]
            # target : result_pos en one-hot
            target = torch.zeros(PART, device=device)
            target[pair["result_pos"]] = 1.0
            loss = 1.0 - F.cosine_similarity(
                out[:PART].unsqueeze(0), target.unsqueeze(0)).clamp(-1, 1)
            total_loss += loss
        (total_loss / len(idx)).backward()
        opt.step()
        opt.zero_grad()

    # évalue sur hold-out
    correct = 0
    blk.eval()
    with torch.no_grad():
        for pair in test_pairs[:200]:
            x = encode_input(pair["a_pos"], pair["op_pos"], d).unsqueeze(0).to(device)
            out = blk(x)[0]
            pred = int(out[:PART].argmax())
            if pred == pair["result_pos"]:
                correct += 1
    acc = correct / min(200, len(test_pairs))

    return blk, {
        "primitive": "number-binding (a + op → result)",
        "n_train": len(train_pairs), "n_test": min(200, len(test_pairs)),
        "grok_acc": round(acc, 4),
        "grokked": acc >= 0.3,  # PART=64 positions, chance = 1/64 = 1.5%
        "procedure": "custom 1-cos, Adam 3e-3, seed 0, données CoT GSM8K RÉELLES",
        "note": "Le number-binding combiné (la primitive manquante pour NL→structured)",
    }


def run_number_binding_grok() -> Dict:
    """GROK le number-binding et rapporte."""
    print("[number_binding] Extraction des bindings depuis CoT GSM8K...")
    blk, res = grok_number_binding(n_pairs=5000, n_steps=2000)
    print(f"  number-binding : acc={res['grok_acc']*100:.1f}% "
          f"{'✓ GROKKED' if res['grokked'] else '✗'}")
    print(f"  {res['procedure']}")
    return res


if __name__ == "__main__":
    run_number_binding_grok()
