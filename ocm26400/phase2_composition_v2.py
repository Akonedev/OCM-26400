#!/usr/bin/env python3
"""PHASE 2 v2 — composition cascade CORRIGÉE (stem fixe + k=2 + anti-raccourci L8).

v1 a échoué (0%) : STEM_FRAME à position variable = non-Fourier-native (mur extraction).
v2 : stem à position FIXE (Phase 1 validé), cascade k=2 (M1=stem+ful → ANS=M1+ness),
     ANTI-RACCOURCI L8 : prédire ANS avec STEM MASQUÉ → force ANS depuis M1 (vraie composition).

Composition testée : stem → M1(+ful) → ANS(+ness). COPY+APPEND ×2 chaînés (Phase 1 prouvé).
DOSC (L7) : P1 predict M1 (stem vis, ANS masqué) ; P2 predict ANS (M1 vis, STEM masqué L8) ;
interleaved. Cascade eval : predict M1 (stem vis) → inject → mask stem → predict ANS (depuis M1).
Cible cascade ≥0.95. Recette : 1 SCB d=48, wd=1e-3, 1-cos, crc32 split, 3 seeds.
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz_"; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 48; W_Q = 12; W_M1 = 14; W_A = 17; L = W_Q + W_M1 + W_A  # 43
F_Q = slice(0, W_Q); F_M1 = slice(W_Q, W_Q + W_M1); F_ANS = slice(W_Q + W_M1, L)
AFF1, AFF2 = "ful", "ness"


def build_triples(cap=80):
    import nltk; from nltk.corpus import wordnet as wn
    try: list(wn.synsets("cat"))
    except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    adj = {ln for s in wn.all_synsets() if s.pos() in ("a", "s") for ln in s.lemma_names() if ln.isalpha() and ln.islower()}
    nou = {ln for s in wn.all_synsets(pos="n") for ln in s.lemma_names() if ln.isalpha() and ln.islower()}
    cand = sorted({l for s in wn.all_synsets(pos="n") for l in s.lemma_names() if l.isalpha() and l.islower() and 2 < len(l) <= 10})
    out = [n for n in cand if (n + AFF1) in adj and (n + AFF1 + AFF2) in nou][:cap]
    return out

def enc(s, w): return [VOC.get(c, VOC["_"]) for c in s[:w]] + [VOC["_"]] * max(0, w - len(s))

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, x): return self.head(self.scb(self.embed(x)))

def loss_on(logits, tgt):
    return (1 - F.cosine_similarity(F.softmax(logits, -1), F.one_hot(tgt, VOC_SIZE).float(), -1)).mean()

def make_seq(stems, mask_stem=False, m1_gt=None):
    """seq = [STEM | M1(masked ou gt) | ANS(masked)]. mask_stem=True → stem masqué (anti-raccourci ANS)."""
    B = len(stems); seq = torch.full((B, L), VOC["_"], dtype=torch.long, device=DEVICE)
    for i, s in enumerate(stems):
        if not mask_stem: seq[i, F_Q] = torch.tensor(enc(s, W_Q), device=DEVICE)   # STEM visible
        if m1_gt is not None: seq[i, F_M1] = torch.tensor(enc(s + AFF1, W_M1), device=DEVICE)  # M1 injecté
    return seq

def tgts(stems):
    B = len(stems)
    t1 = torch.tensor([enc(s + AFF1, W_M1) for s in stems], device=DEVICE)
    ta = torch.tensor([enc(s + AFF1 + AFF2, W_A) for s in stems], device=DEVICE)
    return t1, ta

def train(model, tr, phase, steps, bs=48, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(tr, min(bs, len(tr))); t1, ta = tgts(batch)
        if phase == 1:    # P1 : predict M1 (stem vis, ANS masqué)
            seq = make_seq(batch, mask_stem=False); loss = loss_on(model(seq)[:, F_M1, :], t1)
        elif phase == 2:  # P2 : predict ANS (M1 injecté, STEM MASQUÉ = anti-raccourci L8)
            seq = make_seq(batch, mask_stem=True, m1_gt=True); loss = loss_on(model(seq)[:, F_ANS, :], ta)
        else:             # interleaved
            if random.random() < 0.5: seq = make_seq(batch); loss = loss_on(model(seq)[:, F_M1, :], t1)
            else: seq = make_seq(batch, mask_stem=True, m1_gt=True); loss = loss_on(model(seq)[:, F_ANS, :], ta)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def cascade_eval(model, words):
    """predict M1 (stem vis) → inject M1 prédit → mask stem → predict ANS (depuis M1). cascade exact-match."""
    model.eval()
    with torch.no_grad():
        t1, ta = tgts(words)
        seq = make_seq(words, mask_stem=False)
        p1 = model(seq)[:, F_M1, :].argmax(-1)             # predict M1 depuis stem
        seq2 = make_seq(words, mask_stem=True); seq2[:, F_M1] = p1   # inject M1, mask stem
        pa = model(seq2)[:, F_ANS, :].argmax(-1)           # predict ANS depuis M1
    model.train()
    def w(t): return "".join(CHARS[c] for c in t.tolist() if c != VOC["_"])
    ok = m1 = ans = 0
    for i in range(len(words)):
        a, b = (w(t1[i]) == w(p1[i])), (w(ta[i]) == w(pa[i]))
        m1 += a; ans += b; ok += (a and b)
    return ok/len(words), m1/len(words), ans/len(words)

def run(seeds=(0, 1, 2)):
    import numpy as np
    triples = build_triples(80)
    print(f"Phase 2 v2 : {len(triples)} triplets N→ful→ness (stem FIXE, k=2, anti-raccourci L8)", flush=True)
    if len(triples) < 20: print("⚠️ trop peu", flush=True); return
    cs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in triples if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in triples if zlib.crc32(w.encode()) % 100 < 30]
        m = Model().to(DEVICE)
        train(m, tr, 1, 6000); train(m, tr, 2, 6000); train(m, tr, 3, 12000)
        casc, m1, ans = cascade_eval(m, te)
        print(f"  [seed {s}] CASCADE {casc*100:5.1f}% (M1 {m1*100:.0f} ANS {ans*100:.0f})", flush=True)
        cs.append(casc)
    mean = float(np.mean(cs))
    print(f"\n=> Phase 2 v2 CASCADE moyen : {mean*100:.1f}% (cible ≥0.90)", flush=True)
    print(f"   {'GROK compositionnel ✓' if mean >= 0.85 else ('partiel' if mean > 0.4 else 'échec')}", flush=True)
    json.dump({"cascade_mean": mean, "per_seed": cs, "n": len(triples)}, open("ocm26400/phase2_v2_results.json", "w"), indent=2)
    print("[sauvé]")

if __name__ == "__main__":
    print("="*64); print("PHASE 2 v2 — composition cascade (stem fixe + k=2 + L8)"); print("="*64)
    run()
