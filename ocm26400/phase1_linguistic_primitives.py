#!/usr/bin/env python3
"""PHASE 1 — grok des PRIMITIVES LINGUISTIQUES (ID→ID) en DOSC séquentiel.

Procédure officielle Phase 1 : grok SOLO (1-cos + sommeil autonome + gate L1≥0.99) les
primitives morphologiques RÉGULIÈRES, en curriculum DOSC (L7 : 1 primitive/phase, séquentiel).

Primitives RÉGULIÈRES (règle déterministe → grok possible, canon §8 frontière règle/perception) :
  - PLURAL : nom +s (cat→cats)       [règle +s]
  - PAST   : verbe +ed (walk→walked)  [règle +ed]
  - THIRD  : verbe +s (walk→walks)    [règle +s 3e personne]

Test CLÉ (grok vs mémorisation) : entraîner sur 70% des mots, tester sur 30% HELD-OUT.
  Si le SCB grok la règle → généralise aux mots non-vus (held-out haut).
  Si mémorise → held-out = hasard.

⚠️ Honnête (ADR-0016 double voie) : les primitives ARBITRAIRES (synonyme/antonyme/catégorie)
  n'ont PAS de règle → mémoire (lookup), pas grok. On les teste séparément pour confirmer
  qu'elles NE généralisent PAS (contrairement aux règles morphologiques).

DOSC (L7/L8) : 1 primitive à la fois, sommeil entre, anti-raccourci symétrique.
Canon respecté : L1 décomposition, L3 profondeur, grok sur règle (§8), B' AMV canonique.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random
from ocm26400.optimize_sleep import spectral_filter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 64; PART = 64


# ===================== LEXIQUE (mots réguliers + formes) =====================
REGULAR_NOUNS = ["cat","dog","book","car","tree","house","bird","fish","cup","pen",
                 "lamp","door","wall","ball","chair","key","ring","box","map","star",
                 "ship","rock","leaf","seed","drum","flag","rope","bone","gift","knot"]
REGULAR_VERBS = ["walk","talk","jump","play","look","ask","help","call","wait","wash",
                 "watch","cook","dance","paint","fill","fix","pack","lock","lift","pass",
                 "rest","test","mend","fold","grant","sort","hunt","warm","comb","bake"]


def build_pairs():
    """Construit les paires (word, form) pour chaque primitive. word et form → IDs uniques."""
    vocab = {}  # mot/forme → ID
    def wid(w):
        if w not in vocab: vocab[w] = len(vocab)
        return vocab[w]
    pairs = {"PLURAL": [], "PAST": [], "THIRD": []}
    for n in REGULAR_NOUNS:
        pairs["PLURAL"].append((wid(n), wid(n + "s")))
    for v in REGULAR_VERBS:
        pairs["PAST"].append((wid(v), wid(v + "ed")))
        pairs["THIRD"].append((wid(v), wid(v + "s")))
    return vocab, pairs


# ===================== MODÈLE (crown-jewel : embed → SCB-style → 1-cos) =====================
class ReasonerBlock(nn.Module):
    def __init__(self, d=D, h=128):
        super().__init__(); self.norm = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
        nn.init.normal_(self.f1.weight, std=0.02); nn.init.normal_(self.f2.weight, std=0.02)
        nn.init.zeros_(self.f1.bias); nn.init.zeros_(self.f2.bias)
    def forward(self, x):
        h = self.norm(x); h = torch.relu(self.f1(h)); h = self.f2(h); return x + h


class LinguisticGrokModel(nn.Module):
    """embed(word_id) → ReasonerBlock → 1-cos vers embed(form_id). ConceptVocab dense."""
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, PART); nn.init.normal_(self.embed.weight, std=0.02)
        self.block = ReasonerBlock(D)
    def forward(self, word_id):
        return self.block(self.embed(word_id))  # (B, D)


def make_canon(vocab_size):
    """Dictionnaire canonique figé (orthogonal) pour les IDs — la cible du 1-cos."""
    g = torch.Generator(device=DEVICE).manual_seed(42)
    C = torch.randn(vocab_size, PART, device=DEVICE, generator=g)
    return torch.linalg.qr(C.T)[0].T  # (V, PART) figé


# ===================== GROK SOLO (DOSC) =====================
def grok_primitive(model, pairs, canon, train_idx, n_steps=3000, bs=64, lr=3e-3, wd=1e-3):
    """Grok une primitive SOLO (1-cos) sur les paires train_idx. weight decay (canon §7 clé)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tr = torch.tensor(train_idx, device=DEVICE, dtype=torch.long)
    for _ in range(n_steps):
        idx = tr[torch.randint(0, len(tr), (bs,))]
        w = torch.tensor([pairs[i][0] for i in idx.tolist()], device=DEVICE)
        f = torch.tensor([pairs[i][1] for i in idx.tolist()], device=DEVICE)
        out = model(w)
        tgt = canon[f]
        loss = (1 - F.cosine_similarity(out, tgt, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def gate_and_acc(model, pairs, canon, idx):
    """gate (L_align moyen) + acc (argmax vs canon) sur les paires idx."""
    model.eval()
    with torch.no_grad():
        w = torch.tensor([pairs[i][0] for i in idx], device=DEVICE)
        f = torch.tensor([pairs[i][1] for i in idx], device=DEVICE)
        out = model(w); tgt = canon[f]
        gate = F.cosine_similarity(out, tgt, dim=-1).mean().item()
        pred = (out @ canon.t()).argmax(1)
        acc = (pred == f).float().mean().item()
    model.train(); return gate, acc


def sommeil_if_needed(model, pairs, canon, train_idx, tau=0.99, max_cyc=5):
    """Sommeil autonome (L7) tant que gate<τ. low/high-pass + replay."""
    g, _ = gate_and_acc(model, pairs, canon, train_idx); cyc = 0
    while g < tau and cyc < max_cyc:
        cyc += 1
        spectral_filter(model, 0.5, 'low');  grok_primitive(model, pairs, canon, train_idx, n_steps=300, wd=1e-3)
        spectral_filter(model, 0.3, 'high'); grok_primitive(model, pairs, canon, train_idx, n_steps=300, wd=1e-3)
        g, _ = gate_and_acc(model, pairs, canon, train_idx)
    return g, cyc


def run_primitive(name, pairs, canon, vocab_size, seed=0):
    """DOSC : grok 1 primitive. Train 70% / held-out 30%. Report gate, train acc, HELD-OUT acc."""
    torch.manual_seed(seed); random.seed(seed)
    n = len(pairs); perm = list(range(n)); random.shuffle(perm)
    n_tr = int(n * 0.7); tr_idx, te_idx = perm[:n_tr], perm[n_tr:]
    model = LinguisticGrokModel(vocab_size).to(DEVICE)
    grok_primitive(model, pairs, canon, tr_idx)
    g_tr, a_tr = gate_and_acc(model, pairs, canon, tr_idx)
    g_te, a_te = gate_and_acc(model, pairs, canon, te_idx)
    # sommeil autonome si gate train < τ
    g_sl, cyc = sommeil_if_needed(model, pairs, canon, tr_idx)
    g_tr2, a_tr2 = gate_and_acc(model, pairs, canon, tr_idx)
    g_te2, a_te2 = gate_and_acc(model, pairs, canon, te_idx)
    chance = 1.0 / n
    print(f"  [{name}] train {a_tr*100:5.1f}%→{a_tr2*100:5.1f}% (gate {g_tr:.3f}→{g_tr2:.3f}, sommeil {cyc}c) | "
          f"HELD-OUT {a_te*100:5.1f}%→{a_te2*100:5.1f}% (chance {chance*100:.0f}%)", flush=True)
    grok = a_te2 > chance * 5  # held-out >> chance = généralise (grok la règle)
    return {"train": a_tr2, "held_out": a_te2, "gate": g_tr2, "sommeil_cycles": cyc,
            "chance": chance, "grok": grok}


def main():
    print("="*64); print("PHASE 1 — grok primitives linguistiques RÉGULIÈRES (DOSC, sommeil, gate)"); print("="*64)
    vocab, pairs = build_pairs()
    canon = make_canon(len(vocab)).to(DEVICE)
    print(f"  Vocabulaire : {len(vocab)} IDs | PLURAL {len(pairs['PLURAL'])} | PAST {len(pairs['PAST'])} | THIRD {len(pairs['THIRD'])}\n", flush=True)
    print("  DOSC séquentiel (L7) : 1 primitive à la fois. Test grok = held-out 30%.\n", flush=True)
    results = {}
    for prim in ["PLURAL", "PAST", "THIRD"]:
        results[prim] = run_primitive(prim, pairs[prim], canon, len(vocab))
    print("\n" + "="*64); print("VERDICT Phase 1 (grok règle morphologique) :")
    for p, r in results.items():
        tag = "GROK ✓ (généralise)" if r["grok"] else "mémorise ✗ (held-out ≈ hasard)"
        print(f"  {p:7s}: train {r['train']*100:.0f}% | held-out {r['held_out']*100:.0f}% (chance {r['chance']*100:.0f}%) → {tag}")
    ngrok = sum(1 for r in results.values() if r["grok"])
    if ngrok == len(results):
        print("\n  => Phase 1 GROK les règles morphologiques ✓ (généralise aux mots non-vus).")
        print("     Le SCB apprend la RÈGLE (+s/+ed), pas la mémorisation. Comprehension morphologique.")
    else:
        print(f"\n  => Phase 1 : {ngrok}/{len(results)} primitives grokkent. Les autres = mémorisation.")
        print("     (Conforme canon §8 : grok sur règle. Les règles régulières devraient grokker.)")
    json.dump(results, open("ocm26400/phase1_results.json", "w"), indent=2, default=str)
    print("[sauvé]")


if __name__ == "__main__":
    main()
