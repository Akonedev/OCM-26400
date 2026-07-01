#!/usr/bin/env python3
"""STFT BANK + SCB — lobe NL spectral avec DÉCOMPOSITION LOCALE.

Le SCB seul (FFT global) = 0% sur NL (ne localise pas les mots).
Fix : Conv1d multi-échelle (STFT bank, Fourier-Hann init) → features LOCALES (position×contenu)
→ SCB global (composition). Local-then-global, 100% spectral.

Test sur les 15 règles linguistico-mathématiques (corpus contrôlé de phase1_nl_rules_dosc.py).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random, math
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.phase1_nl_rules_dosc import RULES, gen_examples, enc, make_seq, CHARS, VOC, VS, W_Q, W_A, L
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DM = 48


class STFTBank(nn.Module):
    """Conv1d multi-échelle (Fourier-Hann init) → features locales (position × contenu)."""
    def __init__(self, d=DM, kernels=(3, 5, 9, 17)):
        super().__init__(); self.kernels = kernels; self.convs = nn.ModuleList()
        for W in kernels:
            c = nn.Conv1d(d, d, W, stride=1, padding=W // 2, bias=False)
            self._init_fourier_hann(c, W, d); self.convs.append(c)
        self.fuse = nn.Linear(d * len(kernels), d)
    @staticmethod
    def _init_fourier_hann(conv, W, d):
        n = torch.arange(W).float(); win = 0.5 - 0.5 * torch.cos(2 * math.pi * n / max(W - 1, 1))
        # init comme base de Fourier locale (reelle) × Hann
        Fk = W // 2 + 1
        basis = torch.zeros(d, d, W)
        for ki in range(min(d, Fk)):
            for ci in range(d):
                basis[ki % d, ci, :] = win * torch.cos(2 * math.pi * ki * n / W)
        conv.weight.data = basis + torch.randn(d, d, W) * 0.01
    def forward(self, x):  # (B,L,d) → (B,L,d)
        xt = x.transpose(1, 2); outs = [F.gelu(c(xt)).transpose(1, 2) for c in self.convs]
        return self.fuse(torch.cat(outs, dim=-1))


class STFTNLModel(nn.Module):
    """Embed → STFTBank (LOCAL) → SCB (GLOBAL) → head. 100% spectral."""
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VS, DM)
        self.pos = nn.Parameter(torch.randn(L, DM) * 0.02)
        self.stft = STFTBank(d=DM); self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VS)
    def forward(self, ids): return self.head(self.scb(self.stft(self.embed(ids) + self.pos)))


def train_rule(model, data, rule_id, steps=4000, bs=64, lr=3e-3, wd=1e-3):
    rd = [p for p in data if p["rule"] == rule_id]; opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(rd, min(bs, len(rd)))
        seqs = torch.tensor([make_seq(p["nl"]) for p in batch], device=DEVICE)
        tgts = torch.tensor([enc(p["step"], W_A) for p in batch], device=DEVICE)
        logits = model(seqs)[:, W_Q:W_Q + W_A, :]
        loss = F.cross_entropy(logits.reshape(-1, VS), tgts.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def eval_rule(model, data, rule_id):
    model.eval(); rd = [p for p in data if p["rule"] == rule_id]; ok = 0
    for p in rd:
        with torch.no_grad():
            pred = model(torch.tensor([make_seq(p["nl"])], device=DEVICE))[0, W_Q:W_Q + W_A].argmax(-1).tolist()
        if "".join(CHARS[c] for c in pred if CHARS[c] not in " _") == p["step"]: ok += 1
    model.train(); return ok / len(rd)


def main():
    print("="*64); print("STFT BANK + SCB — lobe NL avec décomposition locale (15 règles)"); print("="*64)
    random.seed(42); all_data = gen_examples(200)
    train_data = []; test_data = []
    for p in all_data: (train_data if random.random() < 0.7 else test_data).append(p)
    print(f"  {len(RULES)} règles | train {len(train_data)} | test {len(test_data)}\n", flush=True)
    model = STFTNLModel().to(DEVICE)
    print("  DOSC (1 règle/phase, STFT+SCB) :\n", flush=True)
    for ri in range(len(RULES)):
        train_rule(model, train_data, ri, steps=3000)
        tr = eval_rule(model, [p for p in train_data if p["rule"] == ri], ri)
        te = eval_rule(model, [p for p in test_data if p["rule"] == ri], ri)
        print(f"  R{ri:2d} {RULES[ri][0][:30]:30s} : train {tr*100:5.1f}% | held-out {te*100:5.1f}% {'✓' if te>0.5 else '✗'}", flush=True)
    te_accs = [eval_rule(model, [p for p in test_data if p["rule"] == ri], ri) for ri in range(len(RULES))]
    mean_te = sum(te_accs) / len(te_accs); n_grok = sum(1 for a in te_accs if a > 0.5)
    print(f"\n  => STFT+SCB held-out moyen : {mean_te*100:.1f}% | {n_grok}/{len(RULES)} règles grokkées", flush=True)
    print(f"     (vs SCB-seul : 0.6%, 0/15)", flush=True)
    json.dump({"mean_held_out": mean_te, "per_rule": te_accs, "n_grok": n_grok, "model": "STFT+SCB"},
              open("ocm26400/stft_nl_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
