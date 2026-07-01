#!/usr/bin/env python3
"""NL PAR RÈGLES (spectral, comprehend-by-rules) — clone phase1_morphology_v4.

Le SCB grok la RÈGLE NL→step (cue "left/remaining"→-, "more than"→+) comme il grok +s→PLURAL.
Input : NL question chars → Output : step arithmétique chars "NN-NN=NN" (COPY nombres + APPLY règle).
PAS de regex, PAS d'extraction mécanique. Le SCB lit le NL, comprend la règle, restitue la step.
1 SCB d=48, char-layout-fixe, 1-cos, wd=1e-3, diffusion-fill (clone phase1_morphology_v4).
Cible ≥0.10 (2.5× le 4% historique, via comprehension-by-rules spectral).
"""
import re, torch, torch.nn as nn, torch.nn.functional as F, json, random, zlib
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.scaleup_gsm8k import train_op, cascade_op
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789+-=_ ."; VOC = {c: i for i, c in enumerate(CHARS)}; VOC_SIZE = len(CHARS)
DM = 48; W_Q = 56; W_A = 14; L = W_Q + W_A


def parse_gsm8k_1step(path):
    """GSM8K 1-step +− : (question NL, step 'a op b = c'). Le SCB doit comprendre NL→step."""
    out = []
    for line in open(path):
        p = json.loads(line); ans = p["answer"]
        gold_m = re.search(r"####\s*(-?\d+)", ans)
        if not gold_m: continue
        steps = re.findall(r"<<([\d.]+)\s*([+\-*/])\s*([\d.]+)=([\d.]+)>>", ans)
        if len(steps) != 1: continue
        a, op, b, c = int(float(steps[0][0])), steps[0][1], int(float(steps[0][2])), int(float(steps[0][3]))
        if op not in "+-": continue
        step_str = f"{a}{op}{b}={c}"
        if len(step_str) > W_A: continue
        out.append({"q": p["question"], "step": step_str, "gold": int(gold_m.group(1))})
    return out


def encode(s, w): return [VOC.get(c, VOC[" "]) for c in s[:w]] + [VOC[" "]] * max(0, w - len(s))

def make_seq(q, step=None):
    """[Q: NL question (W_Q)] + [A: step TOUT-masquée (W_A)]."""
    s = [VOC[" "]] * L; s[:W_Q] = encode(q.lower()[:W_Q], W_Q)
    # A zone = masked (diffusion-fill, tout-masqué COPY rule)
    return s


class NLRuleModel(nn.Module):  # clone MorphCharModel (1 SCB d=48)
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, ids): return self.head(self.scb(self.embed(ids)))


def train(model, data, steps=12000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd); n = len(data)
    for _ in range(steps):
        batch = random.sample(data, min(bs, n))
        seqs = torch.tensor([make_seq(p["q"]) for p in batch], device=DEVICE)
        tgts = torch.tensor([encode(p["step"], W_A) for p in batch], device=DEVICE)
        logits = model(seqs)[:, W_Q:W_Q + W_A, :]
        pred = F.softmax(logits, -1); oh = F.one_hot(tgts, VOC_SIZE).float()
        loss = (1 - F.cosine_similarity(pred, oh, -1)).mean()  # 1-cos (COMPUTE grok)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def eval_model(model, data):
    """E2E : NL → step prédite → cascade crown-jewel → réponse. Compare à gold."""
    model.eval(); ok = tot = 0
    for p in data:
        with torch.no_grad():
            seq = torch.tensor([make_seq(p["q"])], device=DEVICE)
            pred = model(seq)[0, W_Q:W_Q + W_A].argmax(-1).tolist()
        step_str = "".join(CHARS[c] for c in pred if CHARS[c] not in " _")
        # decode step → cascade → answer
        m = re.match(r"(-?\d+)([+\-])(-?\d+)", step_str)
        if not m: tot += 1; continue
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        # utiliser le cascade crown-jewel pour exécuter
        arith = {"+": train_op("+", steps=500), "-": train_op("-", steps=500)}  # quick models
        ans = cascade_op(arith[op], a, b, op)
        tot += 1
        if ans == p["gold"]: ok += 1
    model.train(); return ok / max(tot, 1), ok, tot


def main():
    print("="*64); print("NL PAR RÈGLES — GSM8K 1-step (grok cue→op, spectral, no regex)"); print("="*64)
    train_data = parse_gsm8k_1step("data/gsm8k_train.jsonl")
    test_data = parse_gsm8k_1step("data/gsm8k_test.jsonl")
    print(f"  Train : {len(train_data)} | Test : {len(test_data)}\n", flush=True)
    if len(train_data) < 50: print("⚠️ trop peu", flush=True); return
    random.seed(0); model = NLRuleModel().to(DEVICE)
    train(model, train_data)
    acc, ok, tot = eval_model(model, test_data)
    print(f"\n  NL→step→cascade→réponse E2E (1-step +−) : {ok}/{tot} = {acc*100:.1f}%", flush=True)
    tag = "≥0.10 ✓ (2.5× le 4%)" if acc >= 0.10 else ("partiel" if acc > 0.04 else "≤ 4%")
    print(f"  => {tag}", flush=True)
    json.dump({"acc": acc, "ok": ok, "tot": tot, "n_train": len(train_data)},
              open("ocm26400/nl_rules_v1_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
