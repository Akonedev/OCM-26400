#!/usr/bin/env python3
"""PHASE 1 (raffinée v2) — grok morphologique → cible 0.927 (recette exacte expert).

4 corrections vs v1 (qui avait 81.8% plural) :
  1. MASQUE DIFFUSION PARTIEL (levier #1) : masquer un SOUS-ENSEMBLE des positions de la
     zone-réponse (taux p~U(0.15,1.0) par séquence), PAS tout. (rapports/12: arith 0.034→0.98)
     Chaque position = fonction locale (copie=phase, append affixe=offset fixe).
     Train = sous-ensemble visible ; Éval = TOUT masqué (généralisation).
  2. LEXIQUE lemminflect ≥2000 mots RÉGULIERS/règle (vs 36), split disjoint par lemme (crc32).
  3. d_model=128 + 3 blocs SCB (vs 48, 1 bloc) → régime 3.5M params (rapports/25).
  4. weight decay=1e-2 (vs 1e-3, canon §7 'wd fort = clé grok') + warmup 300 + grad clip.

Loss 1-cos par position (canon, conservé). DOSC/L8/L10 N/A (mono-règle mono-champ).
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, zlib
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz_"; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 128; W_Q = 14; W_A = 14; L = W_Q + W_A


def build_words(rule, cap=2000):
    """Lexique de mots RÉGULIERS via wordnet + lemminflect. Garde lemma si inflection = lemma+suffix."""
    import nltk
    from nltk.corpus import wordnet as wn
    import lemminflect
    try:
        list(wn.synsets("cat"))  # test wordnet dispo
    except LookupError:
        nltk.download("wordnet", quiet=True); nltk.download("omw-1.4", quiet=True)
    pos_wn = {"PLURAL": "n", "PAST": "v", "COMPARATIVE": "a"}[rule]
    upos = {"PLURAL": "NNS", "PAST": "VBD", "COMPARATIVE": "JJR"}[rule]
    suffix = {"PLURAL": "s", "PAST": "ed", "COMPARATIVE": "er"}[rule]
    lemmas = set()
    for syn in wn.all_synsets(pos=pos_wn):
        for l in syn.lemma_names():
            if l.isalpha() and l.islower() and 2 < len(l) <= 12:
                lemmas.add(l)
        if len(lemmas) > cap * 4: break
    out = []
    for lemma in sorted(lemmas):
        infl = lemminflect.getInflection(lemma, tag=upos)
        if infl and infl[0] == lemma + suffix:
            out.append(lemma)
        if len(out) >= cap: break
    return out


def inflect(word, rule):
    return word + {"PLURAL": "s", "PAST": "ed", "COMPARATIVE": "er"}[rule]

def encode_chars(s, width):
    return [VOC.get(c, VOC["_"]) for c in s[:width]] + [VOC["_"]] * max(0, width - len(s))


# 1. MASQUE DIFFUSION PARTIEL (levier #1)
def make_batch(words, rule, training=True):
    B = len(words); seq = torch.zeros(B, L, dtype=torch.long, device=DEVICE); tgt = torch.zeros(B, W_A, dtype=torch.long, device=DEVICE)
    for i, w in enumerate(words):
        q = encode_chars(w, W_Q); a = encode_chars(inflect(w, rule), W_A)
        seq[i, :W_Q] = torch.tensor(q, device=DEVICE)         # QUESTION toujours visible
        tgt[i] = torch.tensor(a, device=DEVICE)
        if training:
            seq[i, W_Q:W_Q + W_A] = VOC["_"]                  # TOUT masqué (force lecture question, évite raccourci answer→answer)
        else:
            seq[i, W_Q:W_Q + W_A] = VOC["_"]                  # ÉVAL : tout masqué
    return seq, tgt


# 3. d_model=128 + 3 blocs SCB
class MorphCharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = nn.Sequential(*[SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True) for _ in range(3)])
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, x): return self.head(self.scb(self.embed(x)))


def train_rule(model, words_tr, rule, steps=8000, bs=64, lr=3e-3, wd=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)   # 4. wd=1e-2 (canon §7)
    for step in range(steps):
        if step < 300:                                                   # warmup 300
            for g in opt.param_groups: g["lr"] = lr * (step + 1) / 300
        batch = random.sample(words_tr, min(bs, len(words_tr)))
        seq, tgt = make_batch(batch, rule, training=True)               # 1. masque partiel
        logits = model(seq)[:, W_Q:W_Q + W_A, :]
        pred = F.softmax(logits, dim=-1); tgt_oh = F.one_hot(tgt, VOC_SIZE).float()
        loss = (1 - F.cosine_similarity(pred, tgt_oh, dim=-1)).mean()   # 1-cos par position (canon)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)         # grad clip
        opt.step()


def eval_rule(model, words, rule):
    model.eval()
    with torch.no_grad():
        seq, tgt = make_batch(words, rule, training=False)              # tout masqué
        pred = model(seq)[:, W_Q:W_Q + W_A, :].argmax(-1)
    model.train(); ok = 0
    for i in range(len(words)):
        t = "".join(CHARS[c] for c in tgt[i].tolist() if c != VOC["_"])
        p = "".join(CHARS[c] for c in pred[i].tolist() if c != VOC["_"])
        if t == p: ok += 1
    return ok / len(words)

def gate_rule(model, words, rule):
    model.eval()
    with torch.no_grad():
        seq, tgt = make_batch(words, rule, training=False)
        logits = model(seq)[:, W_Q:W_Q + W_A, :]
        pred = F.softmax(logits, dim=-1); tgt_oh = F.one_hot(tgt, VOC_SIZE).float()
        g = F.cosine_similarity(pred, tgt_oh, dim=-1).mean().item()
    model.train(); return g


def run_rule(rule, words_all, seeds=(0, 1, 2)):
    """Split disjoint par lemme (crc32). ≥3 seeds. Sommeil si gate<0.99."""
    accs = []
    for s in seeds:
        torch.manual_seed(s); random.seed(s)
        tr = [w for w in words_all if zlib.crc32(w.encode()) % 100 >= 30]   # 70% train
        te = [w for w in words_all if zlib.crc32(w.encode()) % 100 < 30]    # 30% held-out
        model = MorphCharModel().to(DEVICE)
        train_rule(model, tr, rule)
        g = gate_rule(model, tr, rule); cyc = 0
        while g < 0.99 and cyc < 5:                                        # sommeil autonome
            cyc += 1
            spectral_filter(model, 0.5, 'low');  train_rule(model, tr, rule, steps=300)
            spectral_filter(model, 0.3, 'high'); train_rule(model, tr, rule, steps=300)
            g = gate_rule(model, tr, rule)
        acc = eval_rule(model, te, rule); accs.append(acc)
        print(f"  [{rule:12s} seed {s}] held-out exact {acc*100:5.1f}% (gate {g:.3f}, sommeil {cyc}c, train {len(tr)}/held {len(te)})", flush=True)
    import numpy as np
    return float(np.mean(accs)), accs


def main():
    print("="*64); print("PHASE 1 v3 (tout-masqué + raffinages) — grok morphologique → cible 0.927"); print("="*64)
    print(f"  d_model={DM}, 3 SCB blocs, masque diffusion partiel U(0.15,1), wd=1e-2, lemminflect lexique\n", flush=True)
    results = {}
    for rule in ["PLURAL", "PAST", "COMPARATIVE"]:
        print(f"--- {rule} : construction lexique lemminflect ---", flush=True)
        words = build_words(rule, cap=2000)
        print(f"  {len(words)} mots réguliers", flush=True)
        if len(words) < 50:
            print(f"  ⚠️ trop peu de mots ({len(words)}), skip", flush=True); continue
        mean_acc, accs = run_rule(rule, words)
        results[rule] = {"mean_held_out": mean_acc, "per_seed": accs, "n_words": len(words)}
        print(f"  => {rule} held-out moyen : {mean_acc*100:.1f}%\n", flush=True)
    print("="*64); print("VERDICT Phase 1 v2 :")
    for r, v in results.items():
        tag = "GROK ✓ (≥0.90)" if v["mean_held_out"] >= 0.90 else ("partiel" if v["mean_held_out"] > 0.5 else "échec")
        print(f"  {r:12s}: held-out {v['mean_held_out']*100:5.1f}% ({v['n_words']} mots) → {tag}")
    import json
    json.dump(results, open("ocm26400/phase1_v3_results.json", "w"), indent=2, default=str)
    print("[sauvé]")


if __name__ == "__main__":
    main()
