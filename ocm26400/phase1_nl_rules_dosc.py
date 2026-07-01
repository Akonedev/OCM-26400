#!/usr/bin/env python3
"""NL RULES DOSC — grokker 15 règles linguistico-mathématiques (comme Phase 1 grok +s).

Recette expert : corpus CONTRÔLÉ de règles (pas GSM8K brut trop divers).
Chaque règle = un pattern NL fixe + nombres variables → opération.
Le SCB grok la règle (cue→op) comme il grok +s→PLURAL.
1 SCB d=48 + PE + CE + format aligné + DOSC (1 règle/phase, gate≥0.99).
Génération SYNTHÉTIQUE : 200 exemples par règle (nombres aléatoires, pattern fixe).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random, re
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789+-=_ ."; VOC = {c: i for i, c in enumerate(CHARS)}; VS = len(CHARS)
DM = 48; W_Q = 48; W_A = 16; L = W_Q + W_A

# 15 règles linguistico-mathématiques (pattern NL fixe → opération)
RULES = [
    ("{a} more than {b}", "+", "ba"),        # "5 more than 3" → 3+5=8
    ("{a} less than {b}", "-", "ab"),        # "5 less than 8" → 8-5=3
    ("total of {a} and {b}", "+", "ab"),     # "total of 3 and 5" → 3+5=8
    ("difference of {a} and {b}", "-", "ab"),# "difference of 8 and 5" → 8-5=3
    ("sum of {a} and {b}", "+", "ab"),
    ("{a} left after losing {b}", "-", "ab"),# "10 left after losing 3" → 10-3=7
    ("{a} and {b} combined", "+", "ab"),
    ("{a} minus {b}", "-", "ab"),
    ("{a} plus {b}", "+", "ab"),
    ("{a} added to {b}", "+", "ab"),
    ("{b} take away {a}", "-", "ab"),        # "8 take away 3" → 8-3=5
    ("{a} fewer than {b}", "-", "ab"),       # "3 fewer than 8" → 8-3=5
    ("{a} greater than {b}", "-", "ab"),     # "8 greater than 3" → 8-3=5
    ("combined {a} with {b}", "+", "ab"),
    ("{a} subtracted from {b}", "-", "ab"),  # "3 subtracted from 8" → 8-3=5
]


def gen_examples(n=200):
    """Génère n exemples par règle. Pattern NL fixe + nombres aléatoires → step alignée."""
    data = []
    for ri, (pattern, op, order) in enumerate(RULES):
        for _ in range(n):
            a = random.randint(1, 99); b = random.randint(1, 99)
            nl = pattern.format(a=a, b=b)
            if order == "ba": x, y = b, a  # "5 more than 3" → 3+5
            else: x, y = a, b
            r = x + y if op == "+" else x - y
            if r < 0: continue  # pas de négatifs
            step = f"{x:03d}{op}{y:03d}={r:04d}"  # format aligné "008+003=0011"
            data.append({"nl": nl, "step": step, "rule": ri, "op": op})
    return data


def enc(s, w): return [VOC.get(c, VOC[" "]) for c in s[:w]] + [VOC[" "]] * max(0, w - len(s))


class NLModel(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VS, DM)
        self.pos = nn.Parameter(torch.randn(L, DM) * 0.02)  # PE (fix #1)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VS)
    def forward(self, ids): return self.head(self.scb(self.embed(ids) + self.pos))


def make_seq(nl): s = [VOC[" "]] * L; s[:W_Q] = enc(nl.lower()[:W_Q], W_Q); return s


def train_rule(model, data, rule_id, steps=4000, bs=64, lr=3e-3, wd=1e-3):
    """DOSC : grok 1 règle SOLO (CE, fix #2)."""
    rd = [p for p in data if p["rule"] == rule_id]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(rd, min(bs, len(rd)))
        seqs = torch.tensor([make_seq(p["nl"]) for p in batch], device=DEVICE)
        tgts = torch.tensor([enc(p["step"], W_A) for p in batch], device=DEVICE)
        logits = model(seqs)[:, W_Q:W_Q + W_A, :]
        loss = F.cross_entropy(logits.reshape(-1, VS), tgts.reshape(-1))  # CE (fix #2)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def eval_rule(model, data, rule_id):
    model.eval(); rd = [p for p in data if p["rule"] == rule_id]; ok = 0
    for p in rd:
        with torch.no_grad():
            pred = model(torch.tensor([make_seq(p["nl"])], device=DEVICE))[0, W_Q:W_Q + W_A].argmax(-1).tolist()
        step = "".join(CHARS[c] for c in pred if CHARS[c] not in " _")
        if step == p["step"]: ok += 1
    model.train(); return ok / len(rd)


def main():
    print("="*64); print("NL RULES DOSC — grokker 15 règles linguistico-mathématiques"); print("="*64)
    random.seed(42)
    all_data = gen_examples(200)
    # split par règle : 70% train, 30% held-out (par règle)
    train_data = []; test_data = []
    for p in all_data:
        (train_data if random.random() < 0.7 else test_data).append(p)
    print(f"  {len(RULES)} règles × 200 exemples = {len(all_data)} total | train {len(train_data)} | test {len(test_data)}\n", flush=True)
    model = NLModel().to(DEVICE)
    # DOSC : grok chaque règle SOLO, puis eval
    print("  DOSC (1 règle/phase) :\n", flush=True)
    for ri in range(len(RULES)):
        train_rule(model, train_data, ri, steps=3000)
        tr_acc = eval_rule(model, [p for p in train_data if p["rule"] == ri], ri)
        te_acc = eval_rule(model, [p for p in test_data if p["rule"] == ri], ri)
        pattern = RULES[ri][0][:30]
        print(f"  R{ri:2d} {pattern:30s} : train {tr_acc*100:5.1f}% | held-out {te_acc*100:5.1f}% {'✓' if te_acc>0.5 else '✗'}", flush=True)
    # moyenne held-out
    te_accs = [eval_rule(model, [p for p in test_data if p["rule"] == ri], ri) for ri in range(len(RULES))]
    mean_te = sum(te_accs) / len(te_accs)
    n_grok = sum(1 for a in te_accs if a > 0.5)
    print(f"\n  => Held-out moyen : {mean_te*100:.1f}% | {n_grok}/{len(RULES)} règles grokkées (>50%)", flush=True)
    json.dump({"mean_held_out": mean_te, "per_rule": te_accs, "n_grok": n_grok, "n_rules": len(RULES)},
              open("ocm26400/nl_rules_dosc_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
