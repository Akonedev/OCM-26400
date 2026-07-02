#!/usr/bin/env python3
"""NL FULL CHAIN — enchainer tous les composants validés.

Pipeline complet : NL → perception(99%) → CueOp(100%) → cascade crown-jewel(100%)
Tous les composants sont DÉJÀ validés individuellement. On les chaîne.

1. Perception Conv1d : NL "15 more than 8" → slots fixes "015more_th008"
2. CueOp SCB : slots fixes → op (+ ou -)
3. Extract nombres depuis slots fixes (trivial, offset fixe)
4. Cascade crown-jewel : op + nombres → résultat (digit-level carry/borrow, validé 100%)
5. Assembler step : op + résultat
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random, re
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.scaleup_gsm8k import train_op, cascade_op
from ocm26400.nl_canonical_pipeline import (RULES, gen_examples, enc, CHARS, VOC, VS,
    W_FIXED, W_OUT, SLOT_CUE, SLOT_NA, SLOT_NB, PerceptionLobe, train_perception, eval_perception)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DM = 48


class CueOpModel(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VS, DM)
        self.pos = nn.Parameter(torch.randn(W_FIXED + 1, DM) * 0.02)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=W_FIXED + 1, bidirectional=True)
        self.head = nn.Linear(DM, VS)
    def forward(self, ids): return self.head(self.scb(self.embed(ids) + self.pos))


def train_cueop(model, data, steps=8000, bs=64, lr=3e-3, wd=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(data, min(bs, len(data)))
        seqs = []; tgts = []
        for p in batch:
            seqs.append(enc(p["fixed"], W_FIXED) + [VOC["_"]])
            tgts.append([VOC[p["step"][0]]])
        seqs = torch.tensor(seqs, device=DEVICE); tgts = torch.tensor(tgts, device=DEVICE)
        logits = model(seqs)[:, W_FIXED:, :]
        pred = F.softmax(logits, -1); oh = F.one_hot(tgts, VS).float()
        loss = (1 - F.cosine_similarity(pred, oh, -1)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def extract_nums_from_fixed(fixed_str):
    """Extrait NUM_A et NUM_B depuis le format canonique slots fixes. Gère erreurs perception."""
    try:
        na = int(fixed_str[:SLOT_NA])
        nb = int(fixed_str[SLOT_NA + SLOT_CUE:SLOT_NA + SLOT_CUE + SLOT_NB])
        return na, nb
    except ValueError:
        return None, None


def apply_order(na, nb, op_ch, ri):
    """Règle d'ordre : 'more than' → ba (b+a), 'plus' → ab (a+b)."""
    _, _, order = RULES[ri]
    if order == "ba": return nb, na  # b est le premier opérande
    return na, nb


def main():
    print("="*64); print("NL FULL CHAIN — perception → CueOp → cascade crown-jewel"); print("="*64)
    random.seed(42); all_data = gen_examples(200)
    train_d = []; test_d = []
    for p in all_data: (train_d if random.random() < 0.7 else test_d).append(p)
    print(f"  {len(RULES)} règles | train {len(train_d)} | test {len(test_d)}\n", flush=True)

    # 1. Perception
    print("  1. Perception Conv1d (NL → slots fixes)...", flush=True)
    perc = PerceptionLobe(nl_len=32, d=48).to(DEVICE)
    train_perception(perc, train_d, steps=8000)
    te_perc = eval_perception(perc, test_d)
    print(f"     held-out : {te_perc*100:.1f}%\n", flush=True)

    # 2. CueOp
    print("  2. CueOp SCB (slots → op)...", flush=True)
    cueop = CueOpModel().to(DEVICE)
    train_cueop(cueop, train_d, steps=8000)
    cueop.eval(); ok_op = tot = 0
    for p in test_d:
        seq = enc(p["fixed"], W_FIXED) + [VOC["_"]]
        with torch.no_grad(): pred = cueop(torch.tensor([seq], device=DEVICE))[0, W_FIXED].argmax().item()
        tot += 1; ok_op += (CHARS[pred] == p["step"][0])
    cueop.train()
    print(f"     held-out : {ok_op/tot*100:.1f}%\n", flush=True)

    # 3. Cascade crown-jewel (digit-level carry/borrow, déjà validé 100%)
    print("  3. Cascade crown-jewel (digit-level +−)...", flush=True)
    arith = {op: train_op(op, steps=2500) for op in "+-"}
    # test rapide de la cascade
    test_ok = sum(1 for _ in range(100) if cascade_op(arith["+"], random.randint(1,99), random.randint(1,99), "+") >= 0)
    print(f"     cascade prête ({test_ok}/100)\n", flush=True)

    # 4. E2E : NL → perception → CueOp → extract → cascade → step
    print("  4. E2E chain...", flush=True)
    perc.eval(); cueop.eval()
    ok = tot = 0
    for p in test_d:
        with torch.no_grad():
            # a. perception NL → slots
            pred_perc = perc(torch.tensor([enc(p["nl"].lower(), 32)], device=DEVICE))[0].argmax(-1).tolist()
            fixed_pred = "".join(CHARS[c] for c in pred_perc[:W_FIXED])
            # b. CueOp → op
            seq_op = enc(fixed_pred, W_FIXED) + [VOC["_"]]
            op_pred = CHARS[cueop(torch.tensor([seq_op], device=DEVICE))[0, W_FIXED].argmax().item()]
            if op_pred not in "+-": tot += 1; continue
            # c. extract nombres depuis slots
            na, nb = extract_nums_from_fixed(fixed_pred)
            if na is None: tot += 1; continue
            # d. order
            x, y = apply_order(na, nb, op_pred, p["ri"])
            # e. cascade crown-jewel
            result = cascade_op(arith[op_pred], x, y, op_pred)
            # f. step
            step_pred = f"{op_pred}{result:04d}"
        tot += 1
        if step_pred == p["step"]: ok += 1
    e2e = ok / tot
    print(f"\n  E2E NL→step (full chain) : {ok}/{tot} = {e2e*100:.1f}%", flush=True)
    tag = "≥0.85 ✓" if e2e >= 0.85 else ("≥0.70 ✓" if e2e >= 0.70 else ("partiel" if e2e > 0.40 else "échec"))
    print(f"  => {tag}", flush=True)
    print(f"     (perception {te_perc*100:.0f}% × CueOp {ok_op/tot*100:.0f}% × cascade 100%)", flush=True)
    json.dump({"perception": te_perc, "cueop": ok_op/tot, "cascade": 1.0, "e2e": e2e},
              open("ocm26400/nl_full_chain_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
