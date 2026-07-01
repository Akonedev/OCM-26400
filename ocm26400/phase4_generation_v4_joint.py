#!/usr/bin/env python3
"""PHASE 4 v4 — AMV GÉNÉRATIF per-position + round-trip JOINT (recette expert Option 2+4).

2 causes racines du échec Phase 4 (3 décodeurs → 0%) :
  A. encodeur FROZEN + recognition-only (PLURAL="+s" stem-indépendant → hidden states ne
     retiennent pas l'identité per-stem).
  B. Q-MEAN pooling collapse l'info per-position → AMV dégénéré.

Fix (Option 2+4) :
  - AMV per-position (B,W_Q,DM), PAS de mean (bande-passante pleine, slot-à-slot).
  - Encodeur DÉGELÉ, entraîné ROUND-TRIP joint (chars→z→chars + reconnaissance auxiliaire λ).
  - Injection slot-à-slot : h[t] = embed(char_{t-1}) + z[t] (z[:,t] par position, pas broadcast).
  - Décodeur AR causal SCB (validé v3, copie MSE 0.0005).
Warm-start encodeur depuis Phase-1 grok (protège la reconnaissance), puis round-trip joint.
Cible held-out ≥0.50 (preuve de généralisation générative).
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.phase1_morphology_v4 import (MorphCharModel, build_words, encode_chars,
                                           train_rule, W_Q, W_A, DM, L, CHARS, VOC, VOC_SIZE, DEVICE)
LAMBDA_REC = 0.5; LATENT_DROP = 0.1


def stem_seq(idx):
    s = torch.full((idx.shape[0], L), VOC["_"], dtype=torch.long, device=DEVICE); s[:, :W_Q] = idx; return s


class JointEncoder(nn.Module):  # MorphCharModel dégelé, expose latent per-position
    def __init__(self): super().__init__(); self.m = MorphCharModel().to(DEVICE)
    def latent(self, stem_idx):  # (B,W_Q) → (B,W_Q,DM) latent per-position (l'AMV génératif)
        return self.m.scb(self.m.embed(stem_seq(stem_idx)))[:, :W_Q, :]
    def recog(self, stem_idx):   # reconnaissance Q→A (Phase-1 head)
        return self.m(stem_seq(stem_idx))[:, W_Q:W_Q + W_A, :]


class ARCausalDecoder(nn.Module):  # SCB CAUSAL (validé) + injection slot-à-slot
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=W_Q, bidirectional=False)  # CAUSAL
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, z, target):  # teacher forcing ; z:(B,W_Q,DM) slot par position
        B = z.shape[0]
        bos = torch.full((B, 1), VOC["_"], dtype=torch.long, device=z.device)
        inp = torch.cat([bos, target[:, :-1]], dim=1)  # [BOS,c1..c13]
        z = F.dropout(z, p=LATENT_DROP, training=self.training)
        h = self.embed(inp) + z  # slot-à-slot (z[:,t] par position, PAS broadcast)
        return self.head(self.scb(h))  # (B,W_Q,VOC) causal
    @torch.no_grad()
    def generate(self, z):  # greedy AR gauche→droite
        B = z.shape[0]; inp = torch.full((B, W_Q), VOC["_"], dtype=torch.long, device=z.device)
        preds = []
        for t in range(W_Q):
            h = self.embed(inp) + z; logits = self.head(self.scb(h))
            c = logits[:, t].argmax(-1); preds.append(c)
            if t + 1 < W_Q: inp[:, t + 1] = c
        return torch.stack(preds, dim=1)  # (B,W_Q)


def make_recog_batch(words, rule):  # batch reconnaissance (Phase-1) : seq [stem|A masked], tgt A
    from ocm26400.phase1_morphology_v4 import inflect
    B = len(words); seq = torch.full((B, L), VOC["_"], dtype=torch.long, device=DEVICE)
    tgt = torch.zeros(B, W_A, dtype=torch.long, device=DEVICE)
    for i, w in enumerate(words):
        seq[i, :W_Q] = torch.tensor(encode_chars(w, W_Q), device=DEVICE)
        tgt[i] = torch.tensor(encode_chars(inflect(w, rule), W_A), device=DEVICE)
    return seq, tgt


def train_joint(enc, dec, words, rule, steps=12000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=wd)
    stems = torch.tensor([encode_chars(w, W_Q) for w in words], device=DEVICE); n = len(words)
    for _ in range(steps):
        i = torch.randint(0, n, (min(bs, n),))
        z = enc.latent(stems[i])                                  # AMV per-position
        logits = dec(z, stems[i])                                 # AR chars (round-trip)
        L_recon = F.cross_entropy(logits.reshape(-1, VOC_SIZE), stems[i].reshape(-1))
        seq, tgt = make_recog_batch([words[k] for k in i.tolist()], rule)
        pred = F.softmax(enc.m(seq)[:, W_Q:W_Q + W_A, :], dim=-1)
        L_recog = (1 - F.cosine_similarity(pred, F.one_hot(tgt, VOC_SIZE).float(), dim=-1)).mean()
        loss = L_recon + LAMBDA_REC * L_recog                     # round-trip + reconnaissance
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0); opt.step()


@torch.no_grad()
def eval_heldout(enc, dec, words):
    enc.eval(); dec.eval()
    stems = torch.tensor([encode_chars(w, W_Q) for w in words], device=DEVICE)
    pred = dec.generate(enc.latent(stems))
    enc.train(); dec.train()
    return sum("".join(CHARS[c] for c in pred[i].tolist() if c != VOC["_"]) == w
               for i, w in enumerate(words)) / len(words)


def run(rule="PLURAL", seeds=(0, 1, 2)):
    import numpy as np
    words = build_words(rule, cap=400)
    print(f"Phase 4 v4 : {len(words)} mots — AMV per-position + round-trip JOINT", flush=True)
    if len(words) < 50: print("⚠️ trop peu", flush=True); return
    accs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in words if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in words if zlib.crc32(w.encode()) % 100 < 30]
        enc = JointEncoder().to(DEVICE); dec = ARCausalDecoder().to(DEVICE)
        train_rule(enc.m, tr, rule, steps=12000)  # warm-start grok reconnaissance (98%+)
        train_joint(enc, dec, tr, rule)           # round-trip joint (dégèle l'encodeur)
        tr_acc = eval_heldout(enc, dec, tr); te_acc = eval_heldout(enc, dec, te)
        print(f"  [seed {s}] train {tr_acc*100:5.1f}% | HELD-OUT {te_acc*100:5.1f}% (tr {len(tr)}/te {len(te)})", flush=True)
        accs.append(te_acc)
    m = float(np.mean(accs))
    tag = "GROK génératif ✓ (≥0.50)" if m >= 0.50 else ("partiel" if m > 0.1 else "échec")
    print(f"\n=> Phase 4 v4 {rule} held-out moyen : {m*100:.1f}% → {tag}", flush=True)
    json.dump({"held_out_mean": m, "per_seed": accs, "rule": rule, "n": len(words)},
              open("ocm26400/phase4_v4_joint_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    print("="*64); print("PHASE 4 v4 — AMV génératif per-position + round-trip JOINT"); print("="*64)
    run("PLURAL")
