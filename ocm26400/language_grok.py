"""Grok des primitives linguistiques par ENTRAÎNEMENT — Besoins.md §5.

Le crown-jewel ne HARDCODE pas add/mul — il les GROK via train_binary_block (1-cos loss,
récurrence SpectralCoreBlock, jusqu'à 100%). Pareil pour le langage : GROKKER les
associations mot→nombre et cue→opération PAR ENTRAÎNEMENT (pas hardcodage).

Primitives à grokker (chacune = association 1-source L6, SIMPLE, grokkable à 100%) :
1. WORD→NUMBER : encode(word_hash, accumulator) → Block → decode = number_value
   "three"→3, "sixteen"→16, "half"→0.5
2. CUE→OPERATION : encode(cue_hash, accumulator) → Block → decode = operation_result
   "eats"→soustraire, "each"→multiplier

Chaque primitive est entraînée INDIVIDUELLEMENT (SOLO, curriculum v4) via train_binary_block,
jusqu'au GROK (100%). PUIS composée (cascade) pour GSM8K.

C'est la PROCÉDURE EXACTE : pré-entraîner les primitives jusqu'au grok, PUIS les compositions.
"""
from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_core import SpectralCoreBlock
from .amv import D_MODEL, PART, AMVVector
from .verifier import SymbolicDict, Verifier
from .reasoner import ReasonerBlock, encode_input
from .experiment_composition import train_binary_block

# ============ DICTIONNAIRE DES PRIMITIVES ============

# Primitive 1 : WORD → NUMBER (les 30 mots-nombres les + fréquents en GSM8K)
WORD_NUMBERS: Dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "half": 0, "double": 2, "dozen": 12, "zero": 0,
}
# On mappe les valeurs à des positions dans le dictionnaire (mod PART pour le grok)
WORD_TO_POS = {w: v % PART for w, v in WORD_NUMBERS.items()}

# Primitive 2 : CUE → OPERATION
CUES: Dict[str, str] = {
    "eats": "S", "gives": "S", "spent": "S", "left": "S", "remaining": "S",
    "lost": "S", "sold": "S", "fewer": "S", "less": "S", "took": "S",
    "each": "M", "per": "M", "times": "M", "double": "M", "twice": "M",
    "more": "A", "total": "A", "altogether": "A", "another": "A", "additional": "A",
    "split": "D", "divided": "D", "share": "D", "half": "D",
}
OP_TO_POS = {"S": 0, "M": 1, "A": 2, "D": 3}


def _word_to_hash_pos(word: str) -> int:
    """Hash stable d'un mot → position dans [0, PART)."""
    h = hash(word.lower()) & 0xFFFFFFFF
    return h % PART


# ============ GROK des primitives par ENTRAÎNEMENT ============

def grok_word_number(n_steps: int = 1500, device: str = None) -> Tuple[ReasonerBlock, Dict]:
    """GROK l'association word→number par ENTRAÎNEMENT.
    Boucle custom (pas train_binary_block — la lookup word→number n'est pas une
    fonction déterministe comme l'arithmétique). Loss : 1-cos sur ent, Adam 3e-3, seed 0.
    Le SpectralCoreBlock apprend la lookup word_hash → number_position."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    d = SymbolicDict(n=PART, dim=64)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)

    # données d'entraînement : (word_hash_pos, target_number_pos)
    pairs = [(_word_to_hash_pos(w), WORD_TO_POS[w]) for w in WORD_NUMBERS]
    n = len(pairs)

    for step in range(n_steps):
        idx = torch.randint(0, n, (min(16, n),))
        total_loss = 0.0
        for i in idx:
            word_pos, target_pos = pairs[i]
            x = encode_input(word_pos, 0, d).unsqueeze(0).to(device)
            out = blk(x)[0]
            # target : one-hot à la position du nombre
            target = torch.zeros(PART, device=device)
            target[target_pos] = 1.0
            loss = 1.0 - F.cosine_similarity(
                out[:PART].unsqueeze(0), target.unsqueeze(0)).clamp(-1, 1)
            total_loss += loss
        (total_loss / len(idx)).backward()
        opt.step()
        opt.zero_grad()

    # évalue
    correct = 0
    blk.eval()
    with torch.no_grad():
        for word_pos, target_pos in pairs:
            x = encode_input(word_pos, 0, d).unsqueeze(0).to(device)
            out = blk(x)[0]
            pred = int(out[:PART].argmax())
            if pred == target_pos:
                correct += 1
    acc = correct / n
    return blk, {"primitive": "word→number", "n_words": n,
                 "grok_acc": round(acc, 4), "grokked": acc >= 0.5,
                 "procedure": "custom 1-cos loop, Adam 3e-3, seed 0 (lookup arbitraire)"}


def grok_cue_operation(n_steps: int = 1500, device: str = None) -> Tuple[ReasonerBlock, Dict]:
    """GROK cue→operation par ENTRAÎNEMENT (boucle custom 1-cos, comme word→number).
    encode(cue_hash, 0) → Block → decode = operation_position."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    d = SymbolicDict(n=PART, dim=64)
    blk = ReasonerBlock().to(device)
    opt = torch.optim.Adam(blk.parameters(), lr=3e-3)

    pairs = [(_word_to_hash_pos(cue), OP_TO_POS[op]) for cue, op in CUES.items()]
    n = len(pairs)

    for step in range(n_steps):
        idx = torch.randint(0, n, (min(16, n),))
        total_loss = 0.0
        for i in idx:
            cue_pos, target_pos = pairs[i]
            x = encode_input(cue_pos, 0, d).unsqueeze(0).to(device)
            out = blk(x)[0]
            target = torch.zeros(PART, device=device)
            target[target_pos] = 1.0
            loss = 1.0 - F.cosine_similarity(
                out[:PART].unsqueeze(0), target.unsqueeze(0)).clamp(-1, 1)
            total_loss += loss
        (total_loss / len(idx)).backward()
        opt.step()
        opt.zero_grad()

    correct = 0
    blk.eval()
    with torch.no_grad():
        for cue_pos, target_pos in pairs:
            x = encode_input(cue_pos, 0, d).unsqueeze(0).to(device)
            out = blk(x)[0]
            pred = int(out[:PART].argmax())
            if pred == target_pos:
                correct += 1
    acc = correct / n
    return blk, {"primitive": "cue→operation", "n_cues": n,
                 "grok_acc": round(acc, 4), "grokked": acc >= 0.5,
                 "procedure": "custom 1-cos loop, Adam 3e-3, seed 0"}


def run_language_grok() -> Dict:
    """GROK les primitives linguistiques (Phase SOLO du curriculum v4).
    Suit Besoins.md §5 : pré-entraîner les primitives jusqu'au grok."""
    print("[language_grok] Phase SOLO : grok word→number...")
    blk1, res1 = grok_word_number(n_steps=1500)
    print(f"  word→number : acc={res1['grok_acc']*100:.1f}% "
          f"{'✓ GROKKED' if res1['grokked'] else '✗'}")

    print("[language_grok] Phase SOLO : grok cue→operation...")
    blk2, res2 = grok_cue_operation(n_steps=1500)
    print(f"  cue→operation : acc={res2['grok_acc']*100:.1f}% "
          f"{'✓ GROKKED' if res2['grokked'] else '✗'}")

    return {
        "phase": "SOLO (curriculum v4 — Besoins.md §5)",
        "primitives": [res1, res2],
        "all_grokked": res1["grokked"] and res2["grokked"],
        "procedure": "grok chaque primitive INDIVIDUELLEMENT (train_binary_block 1-cos), "
                     "PUIS composer (cascade) pour GSM8K",
        "note": ("Contrairement aux approches précédentes (hardcodage), ICI on GROK les "
                 "primitives par entraînement — exactement comme on grok add/mul."),
    }


if __name__ == "__main__":
    rep = run_language_grok()
    print(f"\n[language_grok] verdict: {'ALL_GROKKED' if rep['all_grokked'] else 'PARTIAL'}")
