#!/usr/bin/env python3
"""NL PIPELINE B' SPLIT (Règle 8) — séparer COPY et COMPUTE.

Le monolithe 69.4% mélange COPY (copier nombres) + COMPUTE (arithmétique) dans 1 SCB.
Règle 8 : séparer en deux voies canoniques :
  1. COPY : cascade de modèles Phase-1 (copie NUM_A, NUM_B → slots résultat) — config COPY
  2. COMPUTE : single-SCB L8 (détermine op depuis cue, calcule résultat) — config COMPUTE

Ici implémentation concrète :
  - SCB-A grok cue→op (raisonnement, 1-cos, COPY config) — déjà marche
  - SCB-B COMPUTE a±b (single-SCB, L8 anti-raccourci symétrique) — séparé du COPY
  - COPY des nombres = trivial (offset fixe, copie verbatim, canon §24)
"""
import torch, torch.nn as nn, torch.nn.functional as F, json, random
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.nl_canonical_pipeline import (RULES, gen_examples, enc, CHARS, VOC, VS,
    W_FIXED, W_OUT, L_SCB, SLOT_CUE, SLOT_NA, SLOT_NB, PerceptionLobe, train_perception, eval_perception)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DM = 48


class CueOpModel(nn.Module):
    """SCB grok cue→op uniquement. Input slots fixes → output OP (1 char).
    Config COPY (règle 7) : 1 bloc, tout-masqué, wd=1e-3, 1-cos."""
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VS, DM)
        self.pos = nn.Parameter(torch.randn(W_FIXED + 1, DM) * 0.02)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=W_FIXED + 1, bidirectional=True)
        self.head = nn.Linear(DM, VS)
    def forward(self, ids): return self.head(self.scb(self.embed(ids) + self.pos))


class ComputeModel(nn.Module):
    """SCB COMPUTE a±b→résultat. Input = [NUM_A(3)][OP(1)][NUM_B(3)] + [RESULT masqué(4)].
    L8 anti-raccourci : miroir a↔b pendant l'entraînement.
    Config COMPUTE : 1 bloc, masque partiel (OP visible), wd=1e-2."""
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VS, DM)
        self.pos = nn.Parameter(torch.randn(11, DM) * 0.02)  # 3+1+3+4=11
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=11, bidirectional=True)
        self.head = nn.Linear(DM, VS)
    def forward(self, ids): return self.head(self.scb(self.embed(ids) + self.pos))


def train_cueop(model, data, steps=8000, bs=64, lr=3e-3, wd=1e-3):
    """COPY config : grok cue→op. Input=[slots fixes], output=[OP char]."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        batch = random.sample(data, min(bs, len(data)))
        seqs = []; tgts = []
        for p in batch:
            inp = enc(p["fixed"], W_FIXED)  # 14 chars slots
            op_char = p["step"][0]  # '+' ou '-'
            seqs.append(inp + [VOC["_"]])  # +1 masked output
            tgts.append([VOC[op_char]])
        seqs = torch.tensor(seqs, device=DEVICE); tgts = torch.tensor(tgts, device=DEVICE)
        logits = model(seqs)[:, W_FIXED:, :]
        pred = F.softmax(logits, -1); oh = F.one_hot(tgts, VS).float()
        loss = (1 - F.cosine_similarity(pred, oh, -1)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def train_compute(model, data, steps=8000, bs=64, lr=3e-3, wd=1e-2):
    """COMPUTE config : a±b→result. L8 miroir a↔b (anti-raccourci). Masque partiel (OP visible)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # dataset + miroir L8
    mirrored = []
    for p in data:
        mirrored.append(p)
        # miroir : swapper NUM_A↔NUM_B, inverser op si nécessaire
        cue, op, order = RULES[p["ri"]]
        a_val, b_val = p["b"], p["a"]  # swap
        x, y = (b_val, a_val) if order == "ba" else (a_val, b_val)
        r = x + y if op == "+" else x - y
        if r >= 0:
            na = f"{a_val:03d}"; nb = f"{b_val:03d}"; cue_s = cue.ljust(SLOT_CUE)[:SLOT_CUE]
            mirrored.append({"fixed": na + cue_s + nb, "step": f"{op}{r:04d}", "ri": p["ri"], "a": a_val, "b": b_val, "r": r})
    for _ in range(steps):
        batch = random.sample(mirrored, min(bs, len(mirrored)))
        seqs = []; tgts = []
        for p in batch:
            # Input: [NUM_A(3)][OP_visible(1)][NUM_B(3)] + [RESULT_masked(4)]
            na = p["fixed"][:3]; op_ch = p["step"][0]; nb = p["fixed"][3+SLOT_CUE:3+SLOT_CUE+3]
            inp = enc(na + op_ch + nb, 7) + [VOC["_"]] * 4  # 7+4=11
            seqs.append(inp)
            tgts.append(enc(p["step"][1:], 4))  # RESULT only (4 digits)
        seqs = torch.tensor(seqs, device=DEVICE); tgts = torch.tensor(tgts, device=DEVICE)
        logits = model(seqs)[:, 7:, :]  # output zone = 4 positions résultat
        pred = F.softmax(logits, -1); oh = F.one_hot(tgts, VS).float()
        loss = (1 - F.cosine_similarity(pred, oh, -1)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def main():
    print("="*64); print("NL SPLIT PIPELINE (Règle 8) — CueOp(COPY) + Compute(L8) séparés"); print("="*64)
    random.seed(42); all_data = gen_examples(200)
    train_d = []; test_d = []
    for p in all_data: (train_d if random.random() < 0.7 else test_d).append(p)
    print(f"  {len(RULES)} règles | train {len(train_d)} | test {len(test_d)}\n", flush=True)

    # Phase 1 : perception
    perc = PerceptionLobe(nl_len=32, d=48).to(DEVICE)
    train_perception(perc, train_d, steps=8000)
    te_perc = eval_perception(perc, test_d)
    print(f"  perception held-out : {te_perc*100:.1f}%\n", flush=True)

    # Phase 2a : CueOp (COPY config)
    print("  --- CueOp grok (COPY config, 1-cos) ---", flush=True)
    cueop = CueOpModel().to(DEVICE)
    train_cueop(cueop, train_d, steps=8000)
    # eval CueOp
    cueop.eval(); ok_op = tot = 0
    for p in test_d:
        seq = enc(p["fixed"], W_FIXED) + [VOC["_"]]
        with torch.no_grad(): pred = cueop(torch.tensor([seq], device=DEVICE))[0, W_FIXED].argmax().item()
        tot += 1; ok_op += (CHARS[pred] == p["step"][0])
    cueop.train()
    print(f"  CueOp held-out : {ok_op/tot*100:.1f}%\n", flush=True)

    # Phase 2b : Compute (COMPUTE config + L8 miroir)
    print("  --- Compute (COMPUTE config + L8 miroir, masque partiel OP visible) ---", flush=True)
    compute = ComputeModel().to(DEVICE)
    train_compute(compute, train_d, steps=8000)
    # eval Compute
    compute.eval(); ok_comp = tot = 0
    for p in test_d:
        na = p["fixed"][:3]; op_ch = p["step"][0]; nb = p["fixed"][3+SLOT_CUE:3+SLOT_CUE+3]
        seq = enc(na + op_ch + nb, 7) + [VOC["_"]] * 4
        with torch.no_grad(): pred = compute(torch.tensor([seq], device=DEVICE))[0, 7:].argmax(-1).tolist()
        result_pred = "".join(CHARS[c] for c in pred if CHARS[c] in "0123456789")
        expected = p["step"][1:]  # result digits
        tot += 1; ok_comp += (result_pred == expected)
    compute.train()
    print(f"  Compute held-out : {ok_comp/tot*100:.1f}%\n", flush=True)

    # E2E : perception → CueOp → Compute
    perc.eval(); cueop.eval(); compute.eval(); ok = tot = 0
    for p in test_d:
        with torch.no_grad():
            # 1. perception NL → slots
            pred_perc = perc(torch.tensor([enc(p["nl"].lower(), 32)], device=DEVICE))[0].argmax(-1).tolist()
            fixed_pred = "".join(CHARS[c] for c in pred_perc[:W_FIXED])
            # 2. CueOp → OP
            seq_op = enc(fixed_pred, W_FIXED) + [VOC["_"]]
            op_pred = CHARS[cueop(torch.tensor([seq_op], device=DEVICE))[0, W_FIXED].argmax().item()]
            # 3. Compute → result
            na = fixed_pred[:3]; nb = fixed_pred[3+SLOT_CUE:3+SLOT_CUE+3]
            seq_comp = enc(na + op_pred + nb, 7) + [VOC["_"]] * 4
            pred_res = compute(torch.tensor([seq_comp], device=DEVICE))[0, 7:].argmax(-1).tolist()
            step_str = op_pred + "".join(CHARS[c] for c in pred_res if CHARS[c] in "0123456789")
        tot += 1
        if step_str == p["step"]: ok += 1
    e2e = ok / tot
    print(f"  E2E NL→step (split) : {ok}/{tot} = {e2e*100:.1f}%", flush=True)
    tag = "≥0.85 ✓" if e2e >= 0.85 else ("≥0.70 ✓" if e2e >= 0.70 else ("partiel" if e2e > 0.40 else "échec"))
    print(f"  => {tag} (vs 69.4% monolithique)", flush=True)
    json.dump({"perception": te_perc, "cueop": ok_op/tot, "compute": ok_comp/tot, "e2e": e2e},
              open("ocm26400/nl_split_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
