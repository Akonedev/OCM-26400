#!/usr/bin/env python3
"""PHASE 2 v3 — composition cascade = 2 modèles Phase-1 CHAÎNÉS (recette expert option c).

Découverte expert : COPY cascade (morphologie) ≠ COMPUTE cascade (arithmétique).
- COMPUTE : shortcut = réponse fausse → L8 single-SCB validé (rapport 58 v3, 0.97).
- COPY : shortcut (stem+fulness) = chemin composé ((stem+ful)+ness) algébriquement équivalent
  → aucun masquage ne peut forcer une composition distincte + garder un SCB stable
  (filtre FFT global per-fréquence ne supporte pas le flip STEM).
=> Composition morphologique = CHAÎNER 2 modèles Phase-1 à l'inférence.

Modèle_A (Phase 1 v4 canonique) : [Q=stem(14) | A=stem+ful(14)] → M1 à ~0.98.
Modèle_B (Phase 1 v4 canonique) : [Q=X(14) | A=X+ness(14)], X=stem+ful → ANS à ~0.98.
Cascade (inférence) : stem → A → m1_pred → B → ans_pred. cascade ≈ 0.98² ≈ 0.96.
Teste VRAIMENT la composition (chaînage réel, erreur se propage) ET préserve le grok Phase 1.
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz_"; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 48; W_Q = 14; W_A = 14; L = W_Q + W_A   # layout Phase 1 v4
F_Q = slice(0, W_Q); F_A = slice(W_Q, L)
AFF1, AFF2 = "ful", "ness"


def build_triples(cap=80):
    import nltk; from nltk.corpus import wordnet as wn
    try: list(wn.synsets("cat"))
    except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    adj = {ln for s in wn.all_synsets() if s.pos() in ("a", "s") for ln in s.lemma_names() if ln.isalpha() and ln.islower()}
    nou = {ln for s in wn.all_synsets(pos="n") for ln in s.lemma_names() if ln.isalpha() and ln.islower()}
    cand = sorted({l for s in wn.all_synsets(pos="n") for l in s.lemma_names() if l.isalpha() and l.islower() and 2 < len(l) <= 10})
    return [n for n in cand if (n + AFF1) in adj and (n + AFF1 + AFF2) in nou][:cap]


def enc(s, w): return [VOC.get(c, VOC["_"]) for c in s[:w]] + [VOC["_"]] * max(0, w - len(s))

class Model(nn.Module):  # Phase 1 v4 MorphCharModel
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, x): return self.head(self.scb(self.embed(x)))

def loss_on(logits, tgt):
    return (1 - F.cosine_similarity(F.softmax(logits, -1), F.one_hot(tgt, VOC_SIZE).float(), -1)).mean()

def make_batch(inputs, targets):
    B = len(inputs); seq = torch.full((B, L), VOC["_"], dtype=torch.long, device=DEVICE)
    tgt = torch.zeros(B, W_A, dtype=torch.long, device=DEVICE)
    for i in range(B):
        seq[i, F_Q] = torch.tensor(enc(inputs[i], W_Q), device=DEVICE)
        tgt[i] = torch.tensor(enc(targets[i], W_A), device=DEVICE)
    return seq, tgt

def train_model(model, inputs, targets, steps=12000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); n = len(inputs)
    for _ in range(steps):
        idx = random.sample(range(n), min(bs, n)); seq, tgt = make_batch([inputs[i] for i in idx], [targets[i] for i in idx])
        loss = loss_on(model(seq)[:, F_A, :], tgt)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def predict(model, inputs):
    model.eval()
    with torch.no_grad():
        seq, _ = make_batch(inputs, inputs); pred = model(seq)[:, F_A, :].argmax(-1)
    model.train()
    return ["".join(CHARS[c] for c in pred[i].tolist() if c != VOC["_"]) for i in range(len(inputs))]


def run(seeds=(0, 1, 2)):
    import numpy as np
    triples = build_triples(80)
    print(f"Phase 2 v3 : {len(triples)} triplets — 2 modèles Phase-1 chaînés (A: stem→+ful, B: +ful→+ness)", flush=True)
    if len(triples) < 20: print("⚠️ trop peu", flush=True); return
    cs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in triples if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in triples if zlib.crc32(w.encode()) % 100 < 30]
        # Modèle A : stem → stem+ful
        mA = Model().to(DEVICE); train_model(mA, tr, [w + AFF1 for w in tr])
        # Modèle B : stem+ful → stem+fulness  (entrée X=stem+ful, cible=X+ness)
        mB = Model().to(DEVICE); train_model(mB, [w + AFF1 for w in tr], [w + AFF1 + AFF2 for w in tr])
        # Cascade eval (held-out) : stem → mA → m1_pred → mB → ans_pred
        m1_pred = predict(mA, te)                                    # étape 1
        ans_pred = predict(mB, m1_pred)                              # étape 2 (depuis m1_pred, pas GT)
        # métriques
        m1_ok = sum(1 for i, w in enumerate(te) if m1_pred[i] == w + AFF1) / len(te)
        ans_direct = predict(mB, [w + AFF1 for w in te])             # ANS depuis GT m1 (référence)
        ans_ok_gt = sum(1 for i, w in enumerate(te) if ans_direct[i] == w + AFF1 + AFF2) / len(te)
        casc = sum(1 for i, w in enumerate(te) if ans_pred[i] == w + AFF1 + AFF2) / len(te)  # cascade (m1_pred→ans)
        print(f"  [seed {s}] M1 {m1_ok*100:.0f}% | ANS(entrée GT m1) {ans_ok_gt*100:.0f}% | CASCADE(m1_pred→ans) {casc*100:.1f}%", flush=True)
        cs.append(casc)
    mean = float(np.mean(cs))
    print(f"\n=> Phase 2 v3 CASCADE moyen : {mean*100:.1f}% (cible ≥0.90, prédit ≈0.96)", flush=True)
    print(f"   {'GROK compositionnel (chaînage) ✓' if mean >= 0.85 else ('partiel' if mean > 0.5 else 'échec')}", flush=True)
    json.dump({"cascade_mean": mean, "per_seed": cs, "n": len(triples)}, open("ocm26400/phase2_v3_results.json", "w"), indent=2)
    print("[sauvé]")

if __name__ == "__main__":
    print("="*64); print("PHASE 2 v3 — composition = 2 modèles Phase-1 chaînés (COPY cascade)"); print("="*64)
    run()
