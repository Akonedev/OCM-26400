#!/usr/bin/env python3
"""PHASE 4 — lobe INVERSE (réciproque Phase 1) : AMV(concept stem) → stem chars.

Design expert : 100% spectral, NAR (SCB bidir, 1 passe, sans masque), CE sur chars.
- Encodeur Phase-1 v4 FROZEN (98%+, généralise) → stem_code.
- AMVHead LINEAIRE (anti-hash) : stem_code → AMV-256 [ent|...].
- InverseLobe : AMV-256 → SCB(d=48, bidir) → chars (CE).
Round-trip : stem → encodeur → AMV → lobe inverse → stem chars. Held-out par lemme = honnête
(l'encodeur frozen produit un AMV structuré pour les stems non-vus ; le Dec doit le décoder).
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.phase1_morphology_v4 import (MorphCharModel, build_words, encode_chars,
                                           train_rule, W_Q, W_A, DM, L, CHARS, VOC, VOC_SIZE, DEVICE)
ENT_DIM = 64; AMV_DIM = 256


def stem_seq(stems_idx):
    """seq [stem(W_Q) | A masqué(W_A)] pour réutiliser la représentation Q du SCB encodeur."""
    seq = torch.full((stems_idx.shape[0], L), VOC["_"], dtype=torch.long, device=DEVICE)
    seq[:, :W_Q] = stems_idx
    return seq


def stem_code(enc, stems_idx):
    """encodeur frozen → code spectral du stem (mean sur positions Q). (B,W_Q) → (B,DM)."""
    with torch.no_grad():
        h = enc.scb(enc.embed(stem_seq(stems_idx)))[:, :W_Q, :]
    return h.mean(dim=1)


class AMVHead(nn.Module):  # LINEAIRE (anti-hash) : code(48) → ent(64) → AMV-256
    def __init__(self):
        super().__init__(); self.proj = nn.Linear(DM, ENT_DIM)
    def forward(self, code):
        amv = torch.zeros(code.shape[0], AMV_DIM, device=DEVICE)
        amv[:, :ENT_DIM] = self.proj(code)
        return amv


class InverseLobe(nn.Module):  # AMV-256 → chars (miroir spectral, pas de transformer/GRU)
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(AMV_DIM, DM)
        self.pos_embed = nn.Embedding(W_Q, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=W_Q, bidirectional=True)  # NAR bidir, sans masque
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, amv):
        B = amv.shape[0]
        c = self.proj_in(amv).unsqueeze(1).expand(-1, W_Q, -1)
        pos = self.pos_embed(torch.arange(W_Q, device=DEVICE)).unsqueeze(0)
        return self.head(self.scb(c + pos))


def load_frozen_encoder(rule, train_words):
    enc = MorphCharModel().to(DEVICE)
    train_rule(enc, train_words, rule, steps=12000)  # Phase-1 v4 (98%+)
    for p in enc.parameters(): p.requires_grad_(False)
    enc.eval()
    return enc


def train_inverse(enc, amv_head, dec, words, steps=8000, bs=64, lr=3e-3, wd=1e-3):
    params = list(amv_head.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    stems_idx = torch.tensor([encode_chars(w, W_Q) for w in words], device=DEVICE)
    targets = stems_idx.clone(); n = len(words)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        code = stem_code(enc, stems_idx[idx]); amv = amv_head(code)
        logits = dec(amv)
        loss = F.cross_entropy(logits.reshape(-1, VOC_SIZE), targets[idx].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()


@torch.no_grad()
def eval_inverse(enc, amv_head, dec, words):
    amv_head.eval(); dec.eval()
    stems_idx = torch.tensor([encode_chars(w, W_Q) for w in words], device=DEVICE)
    amv = amv_head(stem_code(enc, stems_idx))
    pred = dec(amv).argmax(-1)
    ok = sum(1 for i, w in enumerate(words)
             if "".join(CHARS[c] for c in pred[i].tolist() if c != VOC["_"]) == w)
    amv_head.train(); dec.train()
    return ok / len(words)


def run(rule="PLURAL", seeds=(0, 1, 2)):
    import numpy as np
    words = build_words(rule, cap=400)
    print(f"Phase 4 : {len(words)} mots — lobe inverse AMV→stem (réciproque Phase 1)", flush=True)
    if len(words) < 50: print("⚠️ trop peu", flush=True); return
    accs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in words if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in words if zlib.crc32(w.encode()) % 100 < 30]
        enc = load_frozen_encoder(rule, tr)
        amv_head = AMVHead().to(DEVICE); dec = InverseLobe().to(DEVICE)
        train_inverse(enc, amv_head, dec, tr)
        tr_acc = eval_inverse(enc, amv_head, dec, tr)
        te_acc = eval_inverse(enc, amv_head, dec, te)
        print(f"  [seed {s}] train {tr_acc*100:5.1f}% | HELD-OUT {te_acc*100:5.1f}% (tr {len(tr)}/te {len(te)})", flush=True)
        accs.append(te_acc)
    m = float(np.mean(accs))
    tag = "GROK ✓ (≥0.90)" if m >= 0.90 else ("partiel" if m > 0.5 else "échec")
    print(f"\n=> Phase 4 {rule} held-out moyen : {m*100:.1f}% → {tag}", flush=True)
    json.dump({"held_out_mean": m, "per_seed": accs, "rule": rule, "n": len(words)},
              open("ocm26400/phase4_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    print("="*64); print("PHASE 4 — lobe INVERSE (AMV→signal, génération, réciproque Phase 1)"); print("="*64)
    run("PLURAL")
