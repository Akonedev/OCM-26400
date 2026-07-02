#!/usr/bin/env python3
"""NL PIPELINE B' + DOSC L7/L8 — perception Conv1d → SCB raisonnement DOSC par règle.

DOSC (L7) : 1 règle/phase, gate cos≥0.99, puis règle suivante. Même SCB (pas de reset).
L8 : interleaved final + miroir symétrique (anti-raccourci).
Règle 7 : 1 bloc d=48, tout-masqué, wd=1e-3, 1-cos (config COPY inchangée).
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.nl_canonical_pipeline import (RULES, gen_examples, enc, CHARS, VOC, VS,
    W_FIXED, W_OUT, L_SCB, SLOT_CUE, PerceptionLobe, train_perception, eval_perception)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DM = 48; TAU = 0.99; MAX_STEPS = 4000


class ReasoningCore(nn.Module):
    def __init__(self, d=DM):
        super().__init__(); self.embed = nn.Embedding(VS, d)
        self.pos = nn.Parameter(torch.randn(L_SCB, d) * 0.02)
        self.scb = SpectralCoreBlock(d_model=d, seq_len=L_SCB, bidirectional=True)
        self.head = nn.Linear(d, VS)
    def forward(self, ids): return self.head(self.scb(self.embed(ids) + self.pos))


def gate_rule(model, data_i):
    model.eval(); cos = 0.0; n = 0
    with torch.no_grad():
        for p in data_i:
            seq = enc(p["fixed"], W_FIXED) + [VOC["_"]] * W_OUT
            tgts = enc(p["step"], W_OUT)
            logits = model(torch.tensor([seq], device=DEVICE))[0, W_FIXED:]
            pred = F.softmax(logits, -1); oh = F.one_hot(torch.tensor(tgts, device=DEVICE), VS).float()
            cos += F.cosine_similarity(pred, oh, -1).mean().item(); n += 1
    model.train(); return cos / max(n, 1)


def acc_rule(model, data_i):
    model.eval(); ok = tot = 0
    for p in data_i:
        seq = enc(p["fixed"], W_FIXED) + [VOC["_"]] * W_OUT
        with torch.no_grad():
            pred = model(torch.tensor([seq], device=DEVICE))[0, W_FIXED:].argmax(-1).tolist()
        step = "".join(CHARS[c] for c in pred if CHARS[c] not in " _")
        ok += (step == p["step"]); tot += 1
    model.train(); return ok / max(tot, 1)


def train_steps(model, data_i, opt, steps, bs=64):
    for _ in range(steps):
        batch = random.sample(data_i, min(bs, len(data_i)))
        seqs = []; tgts = []
        for p in batch:
            seqs.append(enc(p["fixed"], W_FIXED) + [VOC["_"]] * W_OUT)
            tgts.append(enc(p["step"], W_OUT))
        seqs = torch.tensor(seqs, device=DEVICE); tgts = torch.tensor(tgts, device=DEVICE)
        logits = model(seqs)[:, W_FIXED:, :]
        pred = F.softmax(logits, -1); oh = F.one_hot(tgts, VS).float()
        loss = (1 - F.cosine_similarity(pred, oh, -1)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def train_dosc(model, train_d, test_d):
    """L7 : 1 règle/phase, gate≥0.99, même SCB (pas de reset)."""
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
    for ri in range(len(RULES)):
        train_i = [p for p in train_d if p["ri"] == ri]
        test_i = [p for p in test_d if p["ri"] == ri]
        if not train_i: continue
        g = gate_rule(model, train_i); steps_done = 0
        while g < TAU and steps_done < MAX_STEPS:
            train_steps(model, train_i, opt, 200); steps_done += 200; g = gate_rule(model, train_i)
        held = acc_rule(model, test_i)
        print(f"  [DOSC] R{ri:2d} {RULES[ri][0]:14s} gate={g:.3f} held={held*100:5.1f}% steps={steps_done}", flush=True)


def train_interleaved(model, train_d, test_d, steps=4000, bs=64):
    """L8 : interleaved 12 règles (consolidation)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(0, steps, 200): train_steps(model, train_d, opt, 200, bs)


def main():
    print("="*64); print("NL DOSC PIPELINE — perception → SCB raisonnement DOSC L7/L8"); print("="*64)
    random.seed(42); all_data = gen_examples(200)
    train_d = []; test_d = []
    for p in all_data: (train_d if random.random() < 0.7 else test_d).append(p)
    print(f"  {len(RULES)} règles | train {len(train_d)} | test {len(test_d)}\n", flush=True)

    # Phase 1 : perception
    perc = PerceptionLobe(nl_len=32, d=48).to(DEVICE)
    train_perception(perc, train_d, steps=8000)
    te_perc = eval_perception(perc, test_d)
    print(f"  perception held-out : {te_perc*100:.1f}%\n", flush=True)

    # Phase 2 : DOSC raisonnement
    print("  --- DOSC L7 (1 règle/phase, gate≥0.99) ---", flush=True)
    reason = ReasoningCore().to(DEVICE)
    train_dosc(reason, train_d, test_d)
    print("\n  --- L8 interleaved (consolidation) ---", flush=True)
    train_interleaved(reason, train_d, test_d, steps=4000)

    # Eval
    te_accs = [acc_rule(reason, [p for p in test_d if p["ri"] == ri]) for ri in range(len(RULES))]
    tr_accs = [acc_rule(reason, [p for p in train_d if p["ri"] == ri]) for ri in range(len(RULES))]
    mean_te = sum(te_accs) / len(te_accs); mean_tr = sum(tr_accs) / len(tr_accs)
    print(f"\n  raisonnement : train {mean_tr*100:.1f}% | held-out {mean_te*100:.1f}%", flush=True)

    # E2E
    perc.eval(); reason.eval(); ok = tot = 0
    for p in test_d:
        with torch.no_grad():
            pred_perc = perc(torch.tensor([enc(p["nl"].lower(), 32)], device=DEVICE))[0].argmax(-1).tolist()
            fixed_pred = "".join(CHARS[c] for c in pred_perc[:W_FIXED])
            seq = enc(fixed_pred, W_FIXED) + [VOC["_"]] * W_OUT
            pred_step = reason(torch.tensor([seq], device=DEVICE))[0, W_FIXED:].argmax(-1).tolist()
            step_str = "".join(CHARS[c] for c in pred_step if CHARS[c] not in " _")
        tot += 1
        if step_str == p["step"]: ok += 1
    e2e = ok / tot
    print(f"  E2E NL→step : {ok}/{tot} = {e2e*100:.1f}%", flush=True)
    tag = "≥0.85 ✓" if e2e >= 0.85 else ("≥0.70 ✓" if e2e >= 0.70 else "partiel")
    print(f"  => {tag} (vs 69.4% sans DOSC)", flush=True)
    json.dump({"perception": te_perc, "reasoning_train": mean_tr, "reasoning_test": mean_te,
               "e2e": e2e, "per_rule_test": te_accs}, open("ocm26400/nl_dosc_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
