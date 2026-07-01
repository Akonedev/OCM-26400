#!/usr/bin/env python3
"""NL PARSING HYBRIDE — GSM8K 1-step : NL → extract nombres → SCB binding/op → cascade → réponse.

Approche hybride (expert) : extracteur regex (nombres, ~95% recall) + SCB char-layout-fixe
(diffusion-fill, 1-cos) pour le binding (quel nombre va dans quel slot) + op + ordre +
cascade crown-jewel (100% validé) comme exécuteur.
Cible 1-step ≥0.10 (2.5× le 4% historique).
"""
import re, torch, torch.nn as nn, torch.nn.functional as F, json, random, zlib
from ocm26400.spectral_core import SpectralCoreBlock
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789+-*/=_ ."; VOC = {c: i for i, c in enumerate(CHARS)}
VOC_SIZE = len(CHARS); DM = 64; W_Q = 64; W_A = 6; L = W_Q + W_A  # output: 2 slots × 3 (idx_a, op, idx_b)
WORD_NUMS = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "one": 1, "two": 2, "half": 0.5}
OP_VOC = {"+": VOC["+"], "-": VOC["-"], "*": VOC["*"], "/": VOC["/"]}


def extract_numbers(text):
    nums = []
    for m in re.finditer(r"\d+\.?\d*", text): nums.append(float(m.group()))
    for w, v in WORD_NUMS.items():
        if re.search(r"\b" + w + r"\b", text.lower()): nums.append(float(v))
    return nums

def normalize(q):
    q = re.sub(r"[^a-z0-9+\-*/ .]", " ", q.lower()); q = re.sub(r"\s+", " ", q).strip()
    return [VOC.get(c, VOC[" "]) for c in q[:W_Q].ljust(W_Q)]

def make_input(q):
    """Full sequence [Q(64) | A_masked(6)] pour le diffusion-fill."""
    return normalize(q) + [VOC["_"]] * W_A


def parse_1step(path):
    out = []
    for line in open(path):
        p = json.loads(line); ans = p["answer"]
        gold_m = re.search(r"####\s*(-?\d+)", ans)
        if not gold_m: continue
        gold = int(gold_m.group(1))
        steps = re.findall(r"<<([\d.]+)\s*([+\-*/])\s*([\d.]+)=([\d.]+)>>", ans)
        if len(steps) != 1: continue
        a, op, b = int(float(steps[0][0])), steps[0][1], int(float(steps[0][2]))
        if op not in "+-": continue
        nums = extract_numbers(p["question"])
        if not nums: continue
        nums_int = [int(round(n)) for n in nums]
        ia = min(range(len(nums_int)), key=lambda k: abs(nums_int[k] - a)) if a in nums_int else 0
        ib = min(range(len(nums_int)), key=lambda k: abs(nums_int[k] - b)) if b in nums_int else (1 if len(nums_int) > 1 else 0)
        tgt = [VOC["_"]] * W_A
        tgt[0] = VOC[str(ia)] if ia < 10 else VOC["0"]
        tgt[1] = OP_VOC[op]
        tgt[2] = VOC[str(ib)] if ib < 10 else VOC["0"]
        out.append({"q": p["question"], "nums": nums_int, "tgt": tgt, "gold": gold, "step": (a, op, b)})
    return out


class NLParser(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(VOC_SIZE, DM)
        self.scb = SpectralCoreBlock(d_model=DM, seq_len=L, bidirectional=True)
        self.head = nn.Linear(DM, VOC_SIZE)
    def forward(self, ids): return self.head(self.scb(self.embed(ids)))


def loss_1cos(logits, tgt_zone):
    pred = F.softmax(logits[:, W_Q:W_Q + W_A, :], -1)
    oh = F.one_hot(tgt_zone, VOC_SIZE).float()
    return (1 - F.cosine_similarity(pred, oh, -1)).mean()


def train_op_model(op_ch, steps=2500):
    """Digit-reasoner (+ ou -, carry) — exécuteur cascade validé 100%."""
    from ocm26400.scaleup_gsm8k import train_op as _train
    return _train(op_ch, steps=steps)


def cascade_op_simple(m, a, b, op_ch):
    """Cascade multi-digit simplifiée (+ et - seulement)."""
    from ocm26400.scaleup_gsm8k import cascade_op
    return cascade_op(m, a, b, op_ch)


def main():
    print("="*64); print("NL PARSING HYBRIDE — GSM8K 1-step (extract + SCB binding + cascade)"); print("="*64)
    # exécuteur arithmétique (100% validé)
    arith = {op: train_op_model(op) for op in "+-"}
    # données 1-step
    train_data = parse_1step("data/gsm8k_train.jsonl")
    test_data = parse_1step("data/gsm8k_test.jsonl")
    print(f"  Train 1-step (+−) : {len(train_data)} | Test : {len(test_data)}\n", flush=True)
    if len(train_data) < 50 or len(test_data) < 10: print("⚠️ trop peu", flush=True); return
    # split par lemme (crc32 du question hash)
    random.seed(0)
    model = NLParser().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
    # entraînement
    for ep in range(30):
        random.shuffle(train_data)
        for p in train_data:
            q_ids = torch.tensor([make_input(p["q"])], device=DEVICE)
            tgt = torch.tensor([p["tgt"]], device=DEVICE)
            loss = loss_1cos(model(q_ids), tgt)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 10 == 0: print(f"  ep {ep} loss {loss.item():.4f}", flush=True)
    # eval E2E : NL → decode → cascade → compare gold
    model.eval(); ok = tot = 0
    for p in test_data:
        with torch.no_grad():
            q_ids = torch.tensor([make_input(p["q"])], device=DEVICE)
            pred = model(q_ids)[0, W_Q:W_Q + W_A].argmax(-1).tolist()
        # decode : idx_a, op, idx_b → nombres → cascade
        ci, op_c, cj = pred[0], pred[1], pred[2]
        op_ch = next((o for o, c in OP_VOC.items() if c == op_c), None)
        if op_ch is None: tot += 1; continue
        nums = p["nums"]
        ia = int(CHARS[ci]) if CHARS[ci].isdigit() and int(CHARS[ci]) < len(nums) else 0
        ib = int(CHARS[cj]) if CHARS[cj].isdigit() and int(CHARS[cj]) < len(nums) else 0
        a = nums[ia] if ia < len(nums) else 0; b = nums[ib] if ib < len(nums) else 0
        pred_ans = cascade_op_simple(arith[op_ch], a, b, op_ch)
        tot += 1
        if pred_ans == p["gold"]: ok += 1
    acc = ok / max(tot, 1)
    print(f"\n  NL→réponse E2E (1-step +−) : {ok}/{tot} = {acc*100:.1f}%", flush=True)
    tag = "≥0.10 (2.5× le 4%) ✓" if acc >= 0.10 else ("partiel" if acc > 0.04 else "≤ 4% historique")
    print(f"  => {tag}", flush=True)
    json.dump({"acc": acc, "ok": ok, "tot": tot, "n_train": len(train_data)},
              open("ocm26400/nl_parsing_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
