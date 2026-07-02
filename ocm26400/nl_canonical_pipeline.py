#!/usr/bin/env python3
"""NL PIPELINE B' — lobe perception (Conv1d slot) → format fixe → SCB grok règle.

Architecture B' : lobe perception DÉTECTE les nombres/cues → les SLOT à positions FIXES
→ le SCB grok la RÈGLE (cue→op) à offset fixe (comme +s→PLURAL).

Phase 1 (perception) : Conv1d multi-échelle détecte "15", "more than", "8" dans le NL
→ les place aux slots fixes [NUM_A][CUE][NUM_B]. Loss CE (perception).

Phase 2 (raisonnement) : SCB reçoit [NUM_A][CUE][NUM_B] (offsets fixes)
→ grok cue→op + copie nombres (offset fixe = phase Fourier)
→ output step "015+008=0023". Loss 1-cos (raisonnement).

100% spectral. Conv1d = filtre spectral local. SCB = FFT global. Pas de transformer.
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random, re, math
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Vocabulaire char
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789+-=_ ."; VOC = {c: i for i, c in enumerate(CHARS)}; VS = len(CHARS)

# Format canonique SLOTS FIXES : [NUM_A(3)][CUE(8)][NUM_B(3)][OP(1)][RESULT(4)] = 19 positions
SLOT_NA = 3; SLOT_CUE = 8; SLOT_NB = 3; SLOT_OP = 1; SLOT_RES = 4
W_FIXED = SLOT_NA + SLOT_CUE + SLOT_NB  # 14 positions input fixe pour le SCB
W_OUT = SLOT_OP + SLOT_RES  # 5 positions output (op + résultat)
L_SCB = W_FIXED + W_OUT  # 19

# Cues → op
RULES = [
    ("more than", "+", "ba"), ("less than", "-", "ab"), ("total of", "+", "ab"),
    ("difference of", "-", "ab"), ("sum of", "+", "ab"), ("minus", "-", "ab"),
    ("plus", "+", "ab"), ("combined", "+", "ab"), ("fewer than", "-", "ab"),
    ("greater than", "-", "ab"), ("take away", "-", "ab"), ("added to", "+", "ba"),
]

CUE_IDS = {cue: i for i, cue in enumerate(r[0] for r in RULES)}


def gen_examples(n_per_rule=200):
    data = []
    for ri, (cue, op, order) in enumerate(RULES):
        for _ in range(n_per_rule):
            a = random.randint(1, 99); b = random.randint(1, 99)
            # NL brut (variable)
            nl = f"{a} {cue} {b}"
            # Format canonique slots FIXES (ce que le lobe perception doit produire)
            na_str = f"{a:03d}"; nb_str = f"{b:03d}"
            cue_str = cue.ljust(SLOT_CUE)[:SLOT_CUE]
            fixed_input = na_str + cue_str + nb_str  # 14 chars fixes
            # Step de sortie (ce que le SCB doit produire)
            if order == "ba": x, y = b, a
            else: x, y = a, b
            r = x + y if op == "+" else x - y
            if r < 0: continue
            step = f"{op}{r:04d}"  # 5 chars: op + 4 digits
            data.append({"nl": nl, "fixed": fixed_input, "step": step, "ri": ri, "a": a, "b": b, "r": r})
    return data


def enc(s, w): return [VOC.get(c, VOC[" "]) for c in s[:w]] + [VOC[" "]] * max(0, w - len(s))


# ============ PHASE 1 : LOBE PERCEPTION (Conv1d → slots fixes) ============
class PerceptionLobe(nn.Module):
    """Conv1d multi-échelle détecte nombres/cues dans NL → slots fixes.
    Input : NL brut (variable). Output : format canonique slots fixes. Loss CE."""
    def __init__(self, nl_len=32, d=48):
        super().__init__()
        self.embed = nn.Embedding(VS, d)
        self.pos = nn.Parameter(torch.randn(nl_len, d) * 0.02)
        # Conv1d multi-échelle (STFT bank, Fourier-local)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, k, padding=k // 2) for k in (3, 5, 9)])
        self.fuse = nn.Linear(d * 3, d)
        self.scb = SpectralCoreBlock(d_model=d, seq_len=nl_len, bidirectional=True)
        self.head = nn.Linear(d, VS)
        self.nl_len = nl_len
    def forward(self, nl_ids):
        h = self.embed(nl_ids) + self.pos
        xt = h.transpose(1, 2); outs = [F.gelu(c(xt)).transpose(1, 2) for c in self.convs]
        h = self.fuse(torch.cat(outs, dim=-1))
        h = self.scb(h)
        return self.head(h)  # (B, nl_len, VS)


def train_perception(model, data, steps=8000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(data, min(bs, len(data)))
        nl_ids = torch.tensor([enc(p["nl"].lower(), model.nl_len) for p in batch], device=DEVICE)
        tgt = torch.tensor([enc(p["fixed"], model.nl_len) for p in batch], device=DEVICE)
        logits = model(nl_ids)
        loss = F.cross_entropy(logits[:, :W_FIXED, :].reshape(-1, VS), tgt[:, :W_FIXED].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def eval_perception(model, data):
    model.eval(); ok_exact = ok_char = tot = 0
    for p in data:
        with torch.no_grad():
            pred = model(torch.tensor([enc(p["nl"].lower(), model.nl_len)], device=DEVICE))[0].argmax(-1).tolist()
        pred_str = "".join(CHARS[c] for c in pred[:W_FIXED])
        if pred_str == p["fixed"]: ok_exact += 1
        tot += 1
    model.train(); return ok_exact / tot


# ============ PHASE 2 : SCB RAISONNEMENT (slots fixes → step) ============
class ReasoningCore(nn.Module):
    """SCB reçoit le format canonique slots FIXES → grok cue→op + copie (offset fixe).
    Input : [NUM_A(3)][CUE(8)][NUM_B(3)] fixe. Output : [OP(1)][RESULT(4)]. Loss 1-cos."""
    def __init__(self, d=48, n_blocks=1):
        super().__init__()
        self.embed = nn.Embedding(VS, d)
        self.pos = nn.Parameter(torch.randn(L_SCB, d) * 0.02)
        self.scbs = nn.Sequential(*[SpectralCoreBlock(d_model=d, seq_len=L_SCB, bidirectional=True) for _ in range(n_blocks)])
        self.head = nn.Linear(d, VS)
    def forward(self, ids): return self.head(self.scbs(self.embed(ids) + self.pos))


def train_reasoning(model, data, steps=10000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(data, min(bs, len(data)))
        seqs = []; tgts = []
        for p in batch:
            inp = enc(p["fixed"], W_FIXED)
            out_masked = [VOC["_"]] * W_OUT
            seqs.append(inp + out_masked)
            tgts.append(enc(p["step"], W_OUT))
        seqs = torch.tensor(seqs, device=DEVICE); tgts = torch.tensor(tgts, device=DEVICE)
        logits = model(seqs)[:, W_FIXED:, :]
        pred = F.softmax(logits, -1); oh = F.one_hot(tgts, VS).float()
        loss = (1 - F.cosine_similarity(pred, oh, -1)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def eval_reasoning(model, data):
    model.eval(); ok = tot = 0
    for p in data:
        seq = enc(p["fixed"], W_FIXED) + [VOC["_"]] * W_OUT
        with torch.no_grad():
            pred = model(torch.tensor([seq], device=DEVICE))[0, W_FIXED:].argmax(-1).tolist()
        step = "".join(CHARS[c] for c in pred if CHARS[c] not in " _")
        if step == p["step"]: ok += 1
        tot += 1
    model.train(); return ok / tot


def main():
    print("="*64); print("NL PIPELINE B' — perception Conv1d (slots) → SCB grok (offset fixe)"); print("="*64)
    random.seed(42); all_data = gen_examples(200)
    train_d = []; test_d = []
    for p in all_data: (train_d if random.random() < 0.7 else test_d).append(p)
    print(f"  {len(RULES)} règles | train {len(train_d)} | test {len(test_d)}\n", flush=True)

    # Phase 1 : perception (NL → slots fixes)
    print("  --- Phase 1 : perception Conv1d (NL → slots fixes) ---", flush=True)
    perc = PerceptionLobe(nl_len=32, d=48).to(DEVICE)
    train_perception(perc, train_d, steps=8000)
    tr_perc = eval_perception(perc, train_d); te_perc = eval_perception(perc, test_d)
    print(f"  perception : train {tr_perc*100:.1f}% | held-out {te_perc*100:.1f}%\n", flush=True)

    # Phase 2 : raisonnement SCB (slots fixes → step)
    print("  --- Phase 2 : SCB raisonnement (slots fixes → step) ---", flush=True)
    reason = ReasoningCore(d=48).to(DEVICE)
    train_reasoning(reason, train_d, steps=6000)
    tr_reas = eval_reasoning(reason, train_d); te_reas = eval_reasoning(reason, test_d)
    print(f"  raisonnement : train {tr_reas*100:.1f}% | held-out {te_reas*100:.1f}%\n", flush=True)

    # E2E : NL → perception → slots → SCB → step
    print("  --- E2E : NL → perception → SCB → step ---", flush=True)
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
    tag = "≥0.10 ✓" if e2e >= 0.10 else ("partiel" if e2e > 0.01 else "échec")
    print(f"  => {tag} (vs 0% SCB-seul)", flush=True)
    json.dump({"perception": {"train": tr_perc, "test": te_perc},
               "reasoning": {"train": tr_reas, "test": te_reas}, "e2e": e2e},
              open("ocm26400/nl_canonical_pipeline_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
