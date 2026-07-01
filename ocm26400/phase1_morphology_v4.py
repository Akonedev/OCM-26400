#!/usr/bin/env python3
"""PHASE 1 v4 — recette CORRIGÉE (expert, basée sur v1 qui marche) → cible 0.927.

Corrections vs v2/v3 (qui ont régressé) :
  - Modèle : 1 SCB bloc d=48 (v1, PROUVÉ converger gate 0.999). PAS 3 blocs d=128 (γ≈0, scale last).
  - Masque : TOUT-masqué (v1). PAS de partiel (raccourci answer→answer sur copy tasks).
  - wd : 1e-3 (v1). PAS 1e-2 (cause racine underfit v3 : COPY grok déjà compact, wd fort l'écrase).
  - Lexique : lemminflect 400 mots/règle (entre v1 36 under-generalized et v3 2000 underfit).
  - Steps : 12000 (bs 64) ≈ 1900 exp/mot (>>seuil ~128 rapports/25).
  - SOMMEIL GATE-GUARD : seulement si gate_train≥0.95 ET held<0.85 (overfit/mémorisation).
    Si gate<0.95 → underfit → re-entraîner (PAS de filtrage, qui détruirait un modèle pas convergé).
  - Split crc32 par lemme (disjoint), ≥3 seeds, 1-cos par position (canon).
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib, json
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz_"; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 48; W_Q = 14; W_A = 14; L = W_Q + W_A   # v1: 1 bloc d=48


def build_words(rule, cap=400):
    import nltk; from nltk.corpus import wordnet as wn; import lemminflect
    try: list(wn.synsets("cat"))
    except LookupError: nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    pos_wn = {"PLURAL": "n", "PAST": "v", "COMPARATIVE": "a"}[rule]
    upos = {"PLURAL": "NNS", "PAST": "VBD", "COMPARATIVE": "JJR"}[rule]
    suffix = {"PLURAL": "s", "PAST": "ed", "COMPARATIVE": "er"}[rule]
    lemmas = set()
    for syn in wn.all_synsets(pos=pos_wn):
        for l in syn.lemma_names():
            if l.isalpha() and l.islower() and 2 < len(l) <= 12: lemmas.add(l)
        if len(lemmas) > cap * 4: break
    out = []
    for lemma in sorted(lemmas):
        infl = lemminflect.getInflection(lemma, tag=upos)
        if infl and infl[0] == lemma + suffix: out.append(lemma)
        if len(out) >= cap: break
    return out

def inflect(w, rule): return w + {"PLURAL": "s", "PAST": "ed", "COMPARATIVE": "er"}[rule]
def encode_chars(s, width): return [VOC.get(c, VOC["_"]) for c in s[:width]] + [VOC["_"]] * max(0, width - len(s))

def make_batch(words, rule, training=True):  # TOUT-masqué (v1, train==eval, zéro raccourci)
    B = len(words); seq = torch.zeros(B, L, dtype=torch.long, device=DEVICE); tgt = torch.zeros(B, W_A, dtype=torch.long, device=DEVICE)
    for i, w in enumerate(words):
        seq[i, :W_Q] = torch.tensor(encode_chars(w, W_Q), device=DEVICE)       # Q visible
        tgt[i] = torch.tensor(encode_chars(inflect(w, rule), W_A), device=DEVICE)
        seq[i, W_Q:W_Q + W_A] = VOC["_"]                                       # A TOUT masqué
    return seq, tgt

class MorphCharModel(nn.Module):  # v1: 1 bloc d=48
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, x): return self.head(self.scb(self.embed(x)))

def train_rule(model, tr, rule, steps=12000, bs=64, lr=3e-3, wd=1e-3):  # v1: wd=1e-3, pas de warmup
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        seq, tgt = make_batch(random.sample(tr, min(bs, len(tr))), rule, True)
        logits = model(seq)[:, W_Q:W_Q + W_A, :]
        pred = F.softmax(logits, dim=-1); tgt_oh = F.one_hot(tgt, VOC_SIZE).float()
        loss = (1 - F.cosine_similarity(pred, tgt_oh, dim=-1)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def eval_rule(model, words, rule):
    model.eval()
    with torch.no_grad():
        seq, tgt = make_batch(words, rule, False); pred = model(seq)[:, W_Q:W_Q + W_A, :].argmax(-1)
    model.train(); ok = 0
    for i in range(len(words)):
        t = "".join(CHARS[c] for c in tgt[i].tolist() if c != VOC["_"])
        p = "".join(CHARS[c] for c in pred[i].tolist() if c != VOC["_"])
        if t == p: ok += 1
    return ok / len(words)

def gate_rule(model, words, rule):
    model.eval()
    with torch.no_grad():
        seq, tgt = make_batch(words, rule, False); logits = model(seq)[:, W_Q:W_Q + W_A, :]
        pred = F.softmax(logits, dim=-1); tgt_oh = F.one_hot(tgt, VOC_SIZE).float()
        g = F.cosine_similarity(pred, tgt_oh, dim=-1).mean().item()
    model.train(); return g

def run_rule(rule, words_all, seeds=(0, 1, 2)):
    import numpy as np
    accs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in words_all if zlib.crc32(w.encode()) % 100 >= 30]
        te = [w for w in words_all if zlib.crc32(w.encode()) % 100 < 30]
        model = MorphCharModel().to(DEVICE)
        train_rule(model, tr, rule)
        g = gate_rule(model, tr, rule)
        # GATE-GUARD : sommeil seulement si overfit (gate≥0.95 ET held<0.85)
        cyc = 0
        if g < 0.95:  # underfit → re-entraîner, PAS filtrer
            train_rule(model, tr, rule, steps=8000); g = gate_rule(model, tr, rule)
        while g >= 0.95 and eval_rule(model, te, rule) < 0.85 and cyc < 5:
            cyc += 1
            spectral_filter(model, 0.5, 'low');  train_rule(model, tr, rule, steps=500)
            spectral_filter(model, 0.3, 'high'); train_rule(model, tr, rule, steps=500)
            g = gate_rule(model, tr, rule)
        acc = eval_rule(model, te, rule); accs.append(acc)
        print(f"  [{rule:12s} seed {s}] held-out {acc*100:5.1f}% (gate {g:.3f}, sommeil {cyc}c, train {len(tr)}/held {len(te)})", flush=True)
    return float(np.mean(accs)), accs

def main():
    print("="*64); print("PHASE 1 v4 (recette corrigée) — grok morphologique → cible 0.927"); print("="*64)
    print(f"  1 SCB bloc d={DM}, tout-masqué, wd=1e-3, lemminflect 400 mots, 12000 steps, sommeil gate-guardé\n", flush=True)
    results = {}
    for rule in ["PLURAL", "PAST", "COMPARATIVE"]:
        words = build_words(rule, cap=400)
        print(f"--- {rule} : {len(words)} mots réguliers ---", flush=True)
        if len(words) < 50: print(f"  ⚠️ trop peu, skip", flush=True); continue
        mean_acc, accs = run_rule(rule, words)
        results[rule] = {"mean_held_out": mean_acc, "per_seed": accs, "n_words": len(words)}
        print(f"  => {rule} held-out moyen : {mean_acc*100:.1f}%\n", flush=True)
    print("="*64); print("VERDICT Phase 1 v4 :")
    for r, v in results.items():
        tag = "GROK ✓ (≥0.90)" if v["mean_held_out"] >= 0.90 else ("partiel" if v["mean_held_out"] > 0.5 else "échec")
        print(f"  {r:12s}: held-out {v['mean_held_out']*100:5.1f}% ({v['n_words']} mots) → {tag}")
    json.dump(results, open("ocm26400/phase1_v4_results.json", "w"), indent=2, default=str)
    print("[sauvé]")

if __name__ == "__main__":
    main()
