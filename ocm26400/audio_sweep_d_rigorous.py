#!/usr/bin/env python3
"""SWEEP d RIGOUREUX (validé DA+Juges) — comparaisons publishables.

Correctifs vs version précédente :
  1. VAL SPLIT : 10% du train → val. Sélection meilleur checkpoint SUR VAL.
     Test évalué UNE SEULE FOIS à la fin (plus de max-sur-test).
  2. 3 SEEDS {0,1,2} par d → report mean ± std.
  3. FRONT-END FIGÉ : convs 1→32→32→64→64 constants + projection Linear(64→d)
     avant les SCB → seul d (largeur SCB) varie vraiment.
  4. Ratio P/d abandonné (P=35 fixe, sans sens).
Template reste fixe : 7 SCB + dropout 0.1 + attn-pool + AdamW fused + AMP bf16 + 100k.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.audio_unified_m5scb import load_wav

device = "cuda"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
N_SCB = 7


class M5DeepSCB(nn.Module):
    """Front-end FIGÉ (1→32→32→64→64) + proj Linear(64→d) + 7 SCB(d) + attn-pool."""
    def __init__(s, nw, d, n_scb=N_SCB):
        super().__init__()
        # front-end figé (ne dépend PAS de d)
        s.c1 = nn.Conv1d(1, 32, 80, stride=16); s.b1 = nn.BatchNorm1d(32); s.p1 = nn.MaxPool1d(4)
        s.c2 = nn.Conv1d(32, 32, 3); s.b2 = nn.BatchNorm1d(32); s.p2 = nn.MaxPool1d(4)
        s.c3 = nn.Conv1d(32, 64, 3); s.b3 = nn.BatchNorm1d(64)
        s.c4 = nn.Conv1d(64, 64, 3); s.b4 = nn.BatchNorm1d(64)
        s.proj = nn.Linear(64, d)            # projection figée 64 → d (seul d varie)
        s.blocks = nn.ModuleList([SpectralCoreBlock(d_model=d, seq_len=62, bidirectional=True) for _ in range(n_scb)])
        s.drops = nn.ModuleList([nn.Dropout(0.1) for _ in range(n_scb)])
        s.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        s.fc = nn.Linear(d, nw)

    def forward(s, w):
        x = w.unsqueeze(1)
        x = s.p1(F.relu(s.b1(s.c1(x)))); x = s.p2(F.relu(s.b2(s.c2(x))))
        x = F.relu(s.b3(s.c3(x))); x = F.relu(s.b4(s.c4(x)))   # (B,62,64) figé
        frames = s.proj(x.transpose(1, 2))                     # (B,62,d) — seul d varie
        for blk, dr in zip(s.blocks, s.drops):
            frames = dr(blk(frames))
        attn = (frames @ s.pool_q.transpose(-1, -2)).softmax(1)
        return s.fc((frames * attn).sum(1))


def gpu_spd(wv, rates):
    B, L = wv.shape
    arange = torch.arange(L, device=wv.device)
    idx = (arange.unsqueeze(0) / rates.unsqueeze(1)).clamp(0, L - 1)
    lo = idx.long(); hi = (lo + 1).clamp(max=L - 1); frac = idx - lo
    bi = torch.arange(B, device=wv.device).unsqueeze(1)
    return wv[bi, lo] * (1 - frac) + wv[bi, hi] * frac


def load_data_val(words, val_frac=0.1):
    """Split officiel test + split val (10% du train). Retourne tr, val, te."""
    ts = set(l.strip() for l in open(os.path.join(SC, "testing_list.txt")) if l.strip())
    tr, val, te = {}, {}, {}
    rng = np.random.RandomState(123)
    for wi, w in enumerate(words):
        a = sorted(glob.glob(os.path.join(SC, w, "*.wav"))); rp = []
        for p in a:
            if os.path.relpath(p, SC) in ts: te.setdefault(wi, []).append(p)
            else: rp.append(p)
        if len(rp) >= 50 and len(te.get(wi, [])) >= 5:
            rng.shuffle(rp)
            n_val = max(5, int(len(rp) * val_frac))
            tr[wi] = torch.stack([load_wav(p) for p in rp[n_val:]])
            val[wi] = torch.stack([load_wav(p) for p in rp[:n_val]])
            te[wi] = torch.stack([load_wav(p) for p in te[wi]])
    return tr, val, te


@torch.no_grad()
def eval_acc(m, data_gpu):
    m.eval(); ok = tot = 0
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for w in data_gpu:
            for j in range(0, len(data_gpu[w]), 64):
                p = m(data_gpu[w][j:j+64]).argmax(1)
                ok += (p == w).sum().item(); tot += len(p)
    return ok / max(tot, 1)


def run_one(d, seed, tr_gpu, val_gpu, te_gpu, keys, NW, n=100000, bs=32, lr=1e-3, ev=5000):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m = M5DeepSCB(NW, d).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-4, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n)
    best_val = 0.0; best_state = None; L = 16000
    for step in range(n):
        ks = [random.choice(keys) for _ in range(bs)]
        wv = torch.stack([tr_gpu[k][torch.randint(0, len(tr_gpu[k]), (1,), device=device).squeeze()] for k in ks])
        wv = gpu_spd(wv, torch.empty(bs, device=device).uniform_(0.9, 1.1))
        wi = torch.tensor(ks, device=device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(m(wv), wi)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % ev == 0 or step == n - 1:
            val_acc = eval_acc(m, val_gpu)          # sélection SUR VAL (pas test !)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            print(f"  d={d} seed{seed} step {step:>5} val {val_acc*100:.1f}% (best_val {best_val*100:.1f}%) t={time.time():.0f}", flush=True)
    # report test UNE SEULE FOIS avec le meilleur checkpoint val
    m.load_state_dict(best_state)
    test_acc = eval_acc(m, te_gpu)
    return {"d": d, "seed": seed, "best_val": best_val, "test_acc": test_acc}


def main():
    print("="*64); print("SWEEP d RIGOUREUX — val split + 3 seeds + front-end figé"); print("="*64)
    ws = sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(ws)
    print("chargement + splits train/val/test + précharge GPU...", flush=True)
    tr, val, te = load_data_val(ws); keys = list(tr.keys())
    tr_gpu = {k: v.to(device) for k, v in tr.items()}
    val_gpu = {k: v.to(device) for k, v in val.items()}
    te_gpu = {k: v.to(device) for k, v in te.items()}
    print(f"  {NW} mots | train {sum(v.shape[0] for v in tr_gpu.values())} | val {sum(v.shape[0] for v in val_gpu.values())} | test {sum(v.shape[0] for v in te_gpu.values())}", flush=True)
    raw = []
    for d in [64, 128, 192, 256]:
        for seed in [0, 1, 2]:
            t0 = time.time()
            r = run_one(d, seed, tr_gpu, val_gpu, te_gpu, keys, NW)
            r["time_s"] = time.time() - t0
            raw.append(r)
            print(f"[d={d} seed{seed}] best_val={r['best_val']*100:.2f}% test={r['test_acc']*100:.2f}% t={r['time_s']/60:.1f}min", flush=True)
            torch.cuda.empty_cache()
    # agrégation mean ± std sur test
    print("\n" + "="*64); print("RÉSULTATS RIGOUREUX (test, mean ± std sur 3 seeds)"); print("="*64)
    print(f"{'d':>5} {'params':>10} {'test mean':>10} {'± std':>8}")
    summary = []
    for d in [64, 128, 192, 256]:
        tests = [r["test_acc"] for r in raw if r["d"] == d]
        mu = float(np.mean(tests)); sd = float(np.std(tests))
        nparams = sum(p.numel() for p in M5DeepSCB(NW, d).parameters())
        summary.append({"d": d, "nparams": nparams, "test_mean": mu, "test_std": sd, "seeds": tests})
        print(f"{d:>5} {nparams:>10,} {mu*100:>9.2f}% {sd*100:>7.2f}%")
    best = max(summary, key=lambda x: x["test_mean"])
    print(f"\nMEILLEUR: d={best['d']} = {best['test_mean']*100:.2f}% ± {best['test_std']*100:.2f}%")
    json.dump({"raw": raw, "summary": summary}, open("ocm26400/audio_sweep_d_rigorous_results.json", "w"), indent=2)
    print("[sauvé] ocm26400/audio_sweep_d_rigorous_results.json")


if __name__ == "__main__":
    main()
