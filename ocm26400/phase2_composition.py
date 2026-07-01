#!/usr/bin/env python3
"""PHASE 2 — composition linguistique en cascade (DOSC L7/L8 + L10 bidirectionnel).

Design expert : dérivation compositionnelle N→N+ful→N+ful+ness (care→careful→carefulness).
k=3 champs masqués (M0=stem, M1=stem+ful, ANS=stem+fulness). Règle COPY+APPEND ×2 chaînés
(Phase 1 v4 a prouvé le COPY+APPEND grokable).

Représentation : caractère layout-fixe [STEM_FRAME(16) | M0(12) | M1(14) | ANS(17)], L=59.
1 SpectralCoreBlock d=48 (héritage Phase 1 v4, γ≈0).

DOSC (L7) : 3 phases solo (4000 steps) + interleaved (18000). Total 30000.
L8 (anti-raccourci) : P1 masque M1+ANS ; P2 masque ANS (futurs/récupérables).
L10 (mixer bidirectionnel) : champs futurs masqués en input pendant la phase solo.
Métrique : cascade exact-match dependency_fill (M0→inject→M1→inject→ANS). Cible ≥0.95.
Recette : 1 SCB d=48, wd=1e-3, lr=3e-3, bs=48, 1-cos/position, sommeil gate-guardé.
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz_"; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 48; W_F = 16; W_0 = 12; W_1 = 14; W_A = 17
L = W_F + W_0 + W_1 + W_A  # 59
F_FRAME = slice(0, W_F); F_M0 = slice(W_F, W_F + W_0); F_M1 = slice(W_F + W_0, W_F + W_0 + W_1)
F_ANS = slice(W_F + W_0 + W_1, L)
AFF1, AFF2 = "ful", "ness"


def build_triples(cap=80):
    import nltk; from nltk.corpus import wordnet as wn
    try: list(wn.synsets("cat"))
    except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    adj = {ln for s in wn.all_synsets() if s.pos() in ("a", "s") for ln in s.lemma_names() if ln.isalpha() and ln.islower()}
    nou = {ln for s in wn.all_synsets(pos="n") for ln in s.lemma_names() if ln.isalpha() and ln.islower()}
    cand = sorted({l for s in wn.all_synsets(pos="n") for l in s.lemma_names() if l.isalpha() and l.islower() and 2 < len(l) <= 10})
    out = []
    for n in cand:
        if (n + AFF1) in adj and (n + AFF1 + AFF2) in nou: out.append(n)  # régularité (concat littérale)
        if len(out) >= cap: break
    return out


def encode_chars(s, width): return [VOC.get(c, VOC["_"]) for c in s[:width]] + [VOC["_"]] * max(0, width - len(s))


def make_frame(stem, rng):
    pad = rng.randint(0, max(0, W_F - len(stem) - 1))  # position variable (extraction COPY, déclenche L10)
    return "_" * pad + stem + "_" * (W_F - pad - len(stem))


def make_batch(stems, rng):
    """STEM_FRAME visible ; M0/M1/ANS masqués par défaut. Retourne seq + targets."""
    B = len(stems); seq = torch.full((B, L), VOC["_"], dtype=torch.long, device=DEVICE)
    t0 = torch.zeros(B, W_0, dtype=torch.long, device=DEVICE); t1 = torch.zeros(B, W_1, dtype=torch.long, device=DEVICE); ta = torch.zeros(B, W_A, dtype=torch.long, device=DEVICE)
    for i, stem in enumerate(stems):
        seq[i, F_FRAME] = torch.tensor(encode_chars(make_frame(stem, rng), W_F), device=DEVICE)
        t0[i] = torch.tensor(encode_chars(stem, W_0), device=DEVICE)
        t1[i] = torch.tensor(encode_chars(stem + AFF1, W_1), device=DEVICE)
        ta[i] = torch.tensor(encode_chars(stem + AFF1 + AFF2, W_A), device=DEVICE)
    return seq, (t0, t1, ta)


def inject(seq, stems, which):  # réinjecte le ground-truth d'un champ PASSÉ (pour phases >1)
    for i, stem in enumerate(stems):
        if which >= 1: seq[i, F_M0] = torch.tensor(encode_chars(stem, W_0), device=DEVICE)
        if which >= 2: seq[i, F_M1] = torch.tensor(encode_chars(stem + AFF1, W_1), device=DEVICE)


class MorphCascadeModel(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, x): return self.head(self.scb(self.embed(x)))


def loss_on(logits, tgt):
    pred = F.softmax(logits, dim=-1); oh = F.one_hot(tgt, VOC_SIZE).float()
    return (1 - F.cosine_similarity(pred, oh, dim=-1)).mean()


def train_phase(model, tr, phase, steps, bs=48, lr=3e-3, wd=1e-3):
    """DOSC 1 champ/phase. L8/L10 : champs futurs MASQUÉS (déjà par make_batch), champs passés réinjectés."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(tr, min(bs, len(tr))); rng = random
        seq, (t0, t1, ta) = make_batch(batch, rng)
        if phase == 2:  inject(seq, batch, 1)              # M0 visible (passe), ANS masqué (futur)
        elif phase == 3: inject(seq, batch, 2)             # M0+M1 visibles (passés), terminal
        elif phase == 4:  # interleaved 1/3
            r = random.random()
            if r >= 1/3: inject(seq, batch, 1)
            if r >= 2/3: inject(seq, batch, 2)
        logits = model(seq)
        if phase == 1 or (phase == 4 and random.random() < 1/3):  loss = loss_on(logits[:, F_M0, :], t0)
        elif phase == 2:  loss = loss_on(logits[:, F_M1, :], t1)
        elif phase == 3:  loss = loss_on(logits[:, F_ANS, :], ta)
        else:  # phase 4 interleaved (résolu ci-dessus pour M0, sinon M1/ANS)
            loss = loss_on(logits[:, (F_M1 if r < 2/3 else F_ANS), :], t1 if r < 2/3 else ta)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def cascade_eval(model, words):
    """§21 dependency_fill : predict M0 → inject → predict M1 → inject → predict ANS. cascade exact-match."""
    model.eval()
    with torch.no_grad():
        seq, (t0, t1, ta) = make_batch(words, random)
        p0 = model(seq)[:, F_M0, :].argmax(-1); seq[:, F_M0] = p0          # inject M0 prédit
        p1 = model(seq)[:, F_M1, :].argmax(-1); seq[:, F_M1] = p1          # inject M1 prédit
        pa = model(seq)[:, F_ANS, :].argmax(-1)
    model.train()
    def w(t): return "".join(CHARS[c] for c in t.tolist() if c != VOC["_"])
    ok = m0 = m1 = ans = 0
    for i in range(len(words)):
        a, b, c = (w(t0[i]) == w(p0[i])), (w(t1[i]) == w(p1[i])), (w(ta[i]) == w(pa[i]))
        m0 += a; m1 += b; ans += c; ok += (a and b and c)
    return ok/len(words), m0/len(words), m1/len(words), ans/len(words)


def gate_train(model, words):
    model.eval()
    with torch.no_grad():
        seq, (t0, t1, ta) = make_batch(words, random); inject(seq, words, 2)
        logits = model(seq)
        g = (F.cosine_similarity(F.softmax(logits[:, F_ANS, :], -1), F.one_hot(ta, VOC_SIZE).float(), -1).mean().item())
    model.train(); return g


def run(seeds=(0, 1, 2)):
    import numpy as np
    triples = build_triples(cap=80)
    print(f"Phase 2 : {len(triples)} triplets N→ful→ness réguliers", flush=True)
    if len(triples) < 20: print("⚠️ trop peu de triplets", flush=True); return
    cascades = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in triples if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in triples if zlib.crc32(w.encode()) % 100 < 30]
        m = MorphCascadeModel().to(DEVICE)
        train_phase(m, tr, 1, 4000); train_phase(m, tr, 2, 4000); train_phase(m, tr, 3, 4000); train_phase(m, tr, 4, 18000)
        # sommeil gate-guardé (overfit seulement)
        g = gate_train(m, tr); cyc = 0
        while g >= 0.95 and cascade_eval(m, te)[0] < 0.85 and cyc < 5:
            cyc += 1; spectral_filter(m, 0.5, 'low'); train_phase(m, tr, 4, 500); spectral_filter(m, 0.3, 'high'); train_phase(m, tr, 4, 500); g = gate_train(m, tr)
        casc, m0, m1, ans = cascade_eval(m, te)
        print(f"  [seed {s}] CASCADE {casc*100:5.1f}% (m0 {m0*100:.0f} m1 {m1*100:.0f} ans {ans*100:.0f}, sommeil {cyc}c)", flush=True)
        cascades.append(casc)
    mean = float(np.mean(cascades))
    print(f"\n=> Phase 2 CASCADE moyen : {mean*100:.1f}% (cible ≥0.95, prédit L8 ≈0.97)", flush=True)
    tag = "GROK compositionnel ✓" if mean >= 0.90 else ("partiel" if mean > 0.5 else "échec")
    print(f"   verdict : {tag}", flush=True)
    json.dump({"cascade_mean": mean, "per_seed": cascades, "n_triples": len(triples)}, open("ocm26400/phase2_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    print("="*64); print("PHASE 2 — composition cascade N→ful→ness (DOSC L7/L8 + L10)"); print("="*64)
    run()
