#!/usr/bin/env python3
"""NL PAR RÈGLES v2 — 3 FIXES (PE + CE + format aligné).

v1 = 0% (3 bugs : pas de PE, 1-cos au lieu de CE, pas de format aligné).
v2 applique les 3 fixes identifiées par les experts :
  FIX 1: ENCODING POSITIONNEL (Parameter(L,DM)) → brise l'équivariance FFT → localise les mots.
  FIX 2: LOSS CE (pas 1-cos) → le NL = perception (canon : CE pour lobes, 1-cos pour cœur).
  FIX 3: FORMAT ALIGNÉ → step zero-paddée "016-003=0013" (offset fixe, Fourier-native §24).
100% spectral. Clone phase1_morphology_v4 + les 3 fixes. Cible ≥0.10 (rapports/21 = 0.105).
"""
import re, torch, torch.nn as nn, torch.nn.functional as F, json, random
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.scaleup_gsm8k import train_op, cascade_op
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789+-=_ ."; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 48; W_Q = 56; W_A = 14; L = W_Q + W_A


def parse_gsm8k_1step(path):
    out = []
    for line in open(path):
        p = json.loads(line); ans = p["answer"]
        gold_m = re.search(r"####\s*(-?\d+)", ans)
        if not gold_m: continue
        steps = re.findall(r"<<([\d.]+)\s*([+\-*/])\s*([\d.]+)=([\d.]+)>>", ans)
        if len(steps) != 1: continue
        a, op, b, c = int(float(steps[0][0])), steps[0][1], int(float(steps[0][2])), int(float(steps[0][3]))
        if op not in "+-": continue
        # FIX 3: format aligné zero-padded (offset fixe)
        step_str = f"{a:04d}{op}{b:04d}={c:04d}"  # ex "0016-0003=0013"
        if len(step_str) > W_A: continue
        out.append({"q": p["question"], "step": step_str, "a": a, "op": op, "b": b, "gold": int(gold_m.group(1))})
    return out


def enc(s, w): return [VOC.get(c, VOC[" "]) for c in s[:w]] + [VOC[" "]] * max(0, w - len(s))


class NLRuleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOC_SIZE, DM)
        self.pos = nn.Parameter(torch.randn(L, DM) * 0.02)  # FIX 1: encoding positionnel
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, ids):
        return self.head(self.scb(self.embed(ids) + self.pos))  # FIX 1: PE added


def make_seq(q):
    s = [VOC[" "]] * L; s[:W_Q] = enc(q.lower()[:W_Q], W_Q); return s  # A zone = masked


def train(model, data, steps=12000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); n = len(data)
    for _ in range(steps):
        batch = random.sample(data, min(bs, n))
        seqs = torch.tensor([make_seq(p["q"]) for p in batch], device=DEVICE)
        tgts = torch.tensor([enc(p["step"], W_A) for p in batch], device=DEVICE)
        logits = model(seqs)[:, W_Q:W_Q + W_A, :]
        # FIX 2: CE (pas 1-cos) — NL = perception
        loss = F.cross_entropy(logits.reshape(-1, VOC_SIZE), tgts.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def eval_model(model, data, arith):
    model.eval(); ok = tot = 0
    for p in data:
        with torch.no_grad():
            seq = torch.tensor([make_seq(p["q"])], device=DEVICE)
            pred = model(seq)[0, W_Q:W_Q + W_A].argmax(-1).tolist()
        step_str = "".join(CHARS[c] for c in pred if CHARS[c] not in " _")
        m = re.match(r"(\d+)([+\-])(\d+)", step_str)
        if not m: tot += 1; continue
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        ans = cascade_op(arith[op], a, b, op)
        tot += 1
        if ans == p["gold"]: ok += 1
    model.train(); return ok / max(tot, 1), ok, tot


def main():
    print("="*64); print("NL PAR RÈGLES v2 — 3 FIXES (PE + CE + format aligné)"); print("="*64)
    train_data = parse_gsm8k_1step("data/gsm8k_train.jsonl")
    test_data = parse_gsm8k_1step("data/gsm8k_test.jsonl")
    print(f"  Train : {len(train_data)} | Test : {len(test_data)}\n", flush=True)
    if len(train_data) < 50: print("⚠️ trop peu", flush=True); return
    arith = {op: train_op(op, steps=2500) for op in "+-"}  # exécuteur 100% (une fois)
    random.seed(0); model = NLRuleModelV2().to(DEVICE)
    train(model, train_data)
    acc, ok, tot = eval_model(model, test_data, arith)
    print(f"\n  NL→step→cascade→réponse E2E (1-step +−) : {ok}/{tot} = {acc*100:.1f}%", flush=True)
    tag = "≥0.10 ✓ (2.5× le 4%)" if acc >= 0.10 else ("partiel" if acc > 0.04 else "≤ 4%")
    print(f"  => {tag}", flush=True)
    json.dump({"acc": acc, "ok": ok, "tot": tot, "n_train": len(train_data), "fixes": ["PE", "CE", "aligned"]},
              open("ocm26400/nl_rules_v2_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
