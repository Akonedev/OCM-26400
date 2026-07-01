#!/usr/bin/env python3
"""CHAÎNAGE END-TO-END — génération CONCEPT-CONDITIONNÉE (stem → grok → AMV fléchi → inverse → fléchi).

Chaîne les phases validées : Phase 1 (grok morpho) produit l'AMV du concept FLÉCHI (zone A
post-SCB, depuis le raisonnement sur le stem, A masqué en entrée) → décodeur inverse Phase 4'
(ré-entraîné sur round-trip FLÉCHI, Phase 1 gelé) → chars fléchis.

C'est la vraie génération concept-conditionnée : l'AMV vient du GROK (pas re-lu).
Honnêteté : Phase 1 gelé, décodeur frais + archi différente (ARCausalDecoder vs head linéaire P1),
split par lemme, + controls (AMV random = ~0%, AMV permuté = ~0%).
Cible exact-match fléchi end-to-end ≥0.50 held-out.
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.phase1_morphology_v4 import (MorphCharModel, build_words, train_rule,
                                           encode_chars, inflect, W_Q, W_A, DM, L, CHARS, VOC, VOC_SIZE, DEVICE)
from ocm26400.phase4_generation_v4_joint import ARCausalDecoder


def extract_inflected_amv(p1, stems_idx):
    """Zone A post-SCB = AMV per-position du concept fléchi (grokké depuis stem, A masqué en entrée)."""
    B = stems_idx.shape[0]; seq = torch.full((B, L), VOC["_"], dtype=torch.long, device=DEVICE)
    seq[:, :W_Q] = stems_idx
    with torch.no_grad(): h = p1.scb(p1.embed(seq))
    return h[:, W_Q:W_Q + W_A, :]  # (B,14,48) = AMV fléchi (W_A==W_Q)


def train_inverse_flechi(p1, dec, stems, rule, steps=12000, bs=64, lr=3e-3, wd=1e-3):
    p1.eval()
    for p in p1.parameters(): p.requires_grad_(False)  # Phase 1 GELÉ (honnêteté)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=wd)
    idx = torch.tensor([encode_chars(w, W_Q) for w in stems], device=DEVICE)
    tgt = torch.tensor([encode_chars(inflect(w, rule), W_A) for w in stems], device=DEVICE)
    n = len(stems)
    for _ in range(steps):
        i = torch.randint(0, n, (min(bs, n),))
        z = extract_inflected_amv(p1, idx[i])           # AMV fléchi (grok path)
        logits = dec(z, tgt[i])                          # ARCausalDecoder teacher-forced
        loss = F.cross_entropy(logits.reshape(-1, VOC_SIZE), tgt[i].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step()


@torch.no_grad()
def e2e_eval(p1, dec, stems, rule):
    p1.eval(); dec.eval()
    idx = torch.tensor([encode_chars(w, W_Q) for w in stems], device=DEVICE)
    z = extract_inflected_amv(p1, idx)                   # grok → AMV fléchi
    pred = dec.generate(z)                               # AMV fléchi → chars fléchis
    def w(t): return "".join(CHARS[c] for c in t.tolist() if c != VOC["_"])
    ok = sum(1 for i, s in enumerate(stems) if w(pred[i]) == inflect(s, rule))
    # CONTROL 1 : AMV random (doit donner ~0%)
    z_rand = torch.randn(len(stems), W_Q, DM, device=DEVICE)
    pred_rand = dec.generate(z_rand)
    ok_rand = sum(1 for i, s in enumerate(stems) if w(pred_rand[i]) == inflect(s, rule))
    # CONTROL 2 : AMV permuté entre items (doit donner ~0%)
    perm = torch.randperm(len(stems))
    pred_perm = dec.generate(z[perm])
    ok_perm = sum(1 for i, s in enumerate(stems) if w(pred_perm[i]) == inflect(s, rule))
    return ok / len(stems), ok_rand / len(stems), ok_perm / len(stems)


def run(rule="PAST", seeds=(0, 1, 2)):
    import numpy as np
    words = build_words(rule, cap=400)
    print(f"E2E concept-generation : {len(words)} mots — stem→grok→AMV fléchi→inverse→fléchi ({rule})", flush=True)
    if len(words) < 50: print("⚠️ trop peu", flush=True); return
    accs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in words if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in words if zlib.crc32(w.encode()) % 100 < 30]
        p1 = MorphCharModel().to(DEVICE); train_rule(p1, tr, rule)      # Phase 1 grok (98%+)
        dec = ARCausalDecoder().to(DEVICE)                               # décodeur FRAIS
        train_inverse_flechi(p1, dec, tr, rule)                         # round-trip fléchi (P1 gelé)
        te_acc, c_rand, c_perm = e2e_eval(p1, dec, te, rule)
        print(f"  [{rule} seed {s}] HELD-OUT E2E {te_acc*100:5.1f}% | control random {c_rand*100:.0f}% | permuté {c_perm*100:.0f}%", flush=True)
        accs.append(te_acc)
    m = float(np.mean(accs))
    tag = "CONCEPT-GEN ✓ (≥0.50)" if m >= 0.50 else ("partiel" if m > 0.1 else "échec")
    print(f"\n=> E2E {rule} held-out moyen : {m*100:.1f}% → {tag}", flush=True)
    json.dump({"e2e_mean": m, "per_seed": accs, "rule": rule, "n": len(words)},
              open("ocm26400/e2e_concept_generation_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    print("="*64); print("CHAÎNAGE E2E — génération concept-conditionnée (Phase 1 grok → Phase 4 inverse)"); print("="*64)
    run("PAST")
