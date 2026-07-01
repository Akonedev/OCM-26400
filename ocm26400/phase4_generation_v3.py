#!/usr/bin/env python3
"""PHASE 4 v3 — décodeur AR CAUSAL SCB (Option A canonique, rapport 18).

Fix définitif : SCB CAUSAL (patch spectral_core.py : zero-pad 2L + masque temporel, leak ~2e-7).
Décodeur AR teacher-forcing : input [BOS, c1..c13] (varie par position) → SCB causal → next-char.
L'état + le char précédent créent la variation positionnelle → train>0 ET pas de fuite futur.
Greedy AR à l'éval (génération gauche→droite, vrai causal).
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.phase1_morphology_v4 import (MorphCharModel, build_words, encode_chars,
                                           train_rule, W_Q, DM, L, CHARS, VOC, VOC_SIZE, DEVICE)
AMV_DIM = 256; ENT_DIM = 64


def stem_seq(idx):
    seq = torch.full((idx.shape[0], L), VOC["_"], dtype=torch.long, device=DEVICE); seq[:, :W_Q] = idx; return seq

def stem_code(enc, idx):
    with torch.no_grad(): h = enc.scb(enc.embed(stem_seq(idx)))[:, :W_Q, :]
    return h.mean(dim=1)

class AMVHead(nn.Module):
    def __init__(self): super().__init__(); self.proj = nn.Linear(DM, ENT_DIM)
    def forward(self, code):
        amv = torch.zeros(code.shape[0], AMV_DIM, device=DEVICE); amv[:, :ENT_DIM] = self.proj(code); return amv

class ARSpectralDecoder(nn.Module):  # SCB CAUSAL (rapport 18, bat MiniGPT)
    def __init__(self):
        super().__init__()
        self.amv_proj = nn.Linear(AMV_DIM, DM)
        self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=W_Q, bidirectional=False)  # CAUSAL
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, amv, target):  # teacher forcing : [BOS, c1..c13] → predict [c1..c14]
        B = amv.shape[0]; a = self.amv_proj(amv).unsqueeze(1)  # (B,1,DM)
        bos = torch.full((B, 1), VOC["_"], dtype=torch.long, device=amv.device)
        inp = torch.cat([bos, target[:, :-1]], dim=1)  # (B, W_Q) shifted
        h = self.embed(inp) + a  # (B, W_Q, DM) char + AMV par position (varie par position)
        return self.head(self.scb(h))  # (B, W_Q, VOC) — causal, pas de fuite futur
    @torch.no_grad()
    def generate(self, amv):  # greedy AR gauche→droite
        B = amv.shape[0]; a = self.amv_proj(amv).unsqueeze(1)
        inp = torch.full((B, W_Q), VOC["_"], dtype=torch.long, device=amv.device)  # BOS + pads
        for t in range(W_Q):
            logits = self.head(self.scb(self.embed(inp) + a))  # causal, pos t voit inp[0..t]
            c = logits[:, t].argmax(-1)
            if t + 1 < W_Q: inp[:, t + 1] = c
        return logits.argmax(-1)  # (B, W_Q) derniers logits argmax


def load_frozen_encoder(rule, tr):
    enc = MorphCharModel().to(DEVICE); train_rule(enc, tr, rule, steps=12000)
    for p in enc.parameters(): p.requires_grad_(False)
    enc.eval(); return enc

def train_inverse(enc, amv_head, dec, words, steps=8000, bs=64, lr=3e-3, wd=1e-3):
    params = list(amv_head.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    idx_all = torch.tensor([encode_chars(w, W_Q) for w in words], device=DEVICE); n = len(words)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(bs, n),))
        code = stem_code(enc, idx_all[idx]); amv = amv_head(code)
        logits = dec(amv, idx_all[idx])  # teacher forcing
        loss = F.cross_entropy(logits.reshape(-1, VOC_SIZE), idx_all[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()

@torch.no_grad()
def eval_inverse(enc, amv_head, dec, words):
    amv_head.eval(); dec.eval()
    idx_all = torch.tensor([encode_chars(w, W_Q) for w in words], device=DEVICE)
    amv = amv_head(stem_code(enc, idx_all)); pred = dec.generate(amv)
    amv_head.train(); dec.train()
    return sum(1 for i, w in enumerate(words)
               if "".join(CHARS[c] for c in pred[i].tolist() if c != VOC["_"]) == w) / len(words)


def run(rule="PLURAL", seeds=(0, 1, 2)):
    import numpy as np
    words = build_words(rule, cap=400)
    print(f"Phase 4 v3 : {len(words)} mots — décodeur AR CAUSAL SCB (rapport 18)", flush=True)
    if len(words) < 50: print("⚠️ trop peu", flush=True); return
    accs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in words if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in words if zlib.crc32(w.encode()) % 100 < 30]
        enc = load_frozen_encoder(rule, tr)
        amv_head = AMVHead().to(DEVICE); dec = ARSpectralDecoder().to(DEVICE)
        train_inverse(enc, amv_head, dec, tr)
        tr_acc = eval_inverse(enc, amv_head, dec, tr)
        te_acc = eval_inverse(enc, amv_head, dec, te)
        print(f"  [seed {s}] train {tr_acc*100:5.1f}% | HELD-OUT {te_acc*100:5.1f}% (tr {len(tr)}/te {len(te)})", flush=True)
        accs.append(te_acc)
    m = float(np.mean(accs))
    tag = "GROK ✓ (≥0.90)" if m >= 0.90 else ("partiel" if m > 0.3 else "échec")
    print(f"\n=> Phase 4 v3 {rule} held-out moyen : {m*100:.1f}% → {tag}", flush=True)
    json.dump({"held_out_mean": m, "per_seed": accs, "rule": rule, "n": len(words)},
              open("ocm26400/phase4_v3_results.json", "w"), indent=2)
    print("[sauvé]")

if __name__ == "__main__":
    print("="*64); print("PHASE 4 v3 — décodeur AR CAUSAL SCB (AMV→signal, rapport 18)"); print("="*64)
    run("PLURAL")
