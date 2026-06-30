#!/usr/bin/env python3
"""TEST 3-BRAS — l'entropie (incertitude) enrichit-elle le modèle ? (active learning)

Hypothèse : échantillonner les exemples d'entraînement par ENTROPIE de prédiction
(favoriser les incertains) → meilleur apprentissage, surtout sur les cas durs.
Test falsifiable, compute égal, 3 bras :
  A = échantillonnage par entropie (favorise incertains)   ← active learning
  B = échantillonnage uniforme (baseline)
  C = échantillonnage par confiance (favorise sûrs)        ← anti-hypothèse
Mesure : accuracy globale + accuracy sur les MOTS DURS (confusables, baseline faible).
Tous les bras font le MÊME nombre de forwards (rescore inclus) → compute égal.
"""
import torch, torch.nn as nn, torch.nn.functional as F, glob, os, numpy as np, random, json, time
from ocm26400.audio_unified_m5scb import M5Unified, load_wav, spd
device = "cuda"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
STEPS = 20000; BS = 32; RESCORE_EVERY = 500; RESCORE_POOL = 512; TAU = 0.5


def build_index(tr):
    """Flat index : (word, sample_idx) pour tout le train."""
    idx = []
    for wi in tr:
        for si in range(len(tr[wi])):
            idx.append((wi, si))
    return idx


def rescore_entropy(m, tr, idx, pool):
    """Forward un pool d'échantillons, retourne entropie par échantillon du pool."""
    m.eval()
    sample = random.sample(range(len(idx)), min(pool, len(idx)))
    entropies = {}
    with torch.no_grad():
        for k in range(0, len(sample), 64):
            sub = sample[k:k+64]
            wv = torch.stack([tr[idx[i][0]][idx[i][1]] for i in sub]).to(device)
            p = F.softmax(m(wv.unsqueeze(1).squeeze(1) if False else m_forward(m, wv)), dim=-1)
            # forward via M5Unified (attend wav 1D)
            pass
    return entropies  # placeholder


def m_forward(m, wv):
    # M5Unified.forward attend (B, 16000)
    return m(wv)


def train_arm(mode, NW, tr, tr_flat, te, hard_words, seed=0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    m = M5Unified(NW).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    # poids par échantillon (init uniforme)
    weights = np.ones(len(tr_flat), dtype=np.float64)
    t0 = time.time()
    for step in range(STEPS):
        # rescore périodique (tous les bras le font = compute égal ; A/C utilisent les poids)
        if step % RESCORE_EVERY == 0:
            pool = random.sample(range(len(tr_flat)), min(RESCORE_POOL, len(tr_flat)))
            m.eval()
            with torch.no_grad():
                for k in range(0, len(pool), 64):
                    sub = pool[k:k+64]
                    wv = torch.stack([tr[tr_flat[i][0]][tr_flat[i][1]] for i in sub]).to(device)
                    p = F.softmax(m_forward(m, wv), dim=-1)
                    H = -(p * (p + 1e-12).log()).sum(1).cpu().numpy()  # entropie
                    for j, i in enumerate(sub):
                        weights[i] = H[j]
            m.train()
            # normaliser en probas selon le mode
            w = weights.copy()
            if mode == 'uniform':
                probs = np.ones(len(tr_flat)) / len(tr_flat)
            elif mode == 'entropy':
                probs = np.exp(w / TAU); probs /= probs.sum()
            elif mode == 'confidence':
                probs = np.exp(-w / TAU); probs /= probs.sum()
            cum = np.cumsum(probs)
        # échantillonner un batch selon probs
        r = np.random.rand(BS)
        batch_idx = np.searchsorted(cum, r)
        batch_idx = np.clip(batch_idx, 0, len(tr_flat)-1)
        # construire le batch + speed-perturb
        wv = torch.stack([torch.from_numpy(spd(tr[tr_flat[i][0]][tr_flat[i][1]].numpy())) for i in batch_idx]).to(device)
        wi = torch.tensor([tr_flat[i][0] for i in batch_idx], device=device)
        logits = m_forward(m, wv)
        loss = F.cross_entropy(logits, wi)
        opt.zero_grad(); loss.backward(); opt.step()
    acc_all = eval_acc(m, te, list(te.keys()))
    acc_hard = eval_acc(m, te, hard_words)
    return acc_all, acc_hard, time.time()-t0


def eval_acc(m, te, words, bs=64):
    m.eval(); ok = tot = 0
    with torch.no_grad():
        for wi in words:
            if wi not in te: continue
            for j in range(0, len(te[wi]), bs):
                p = m_forward(m, te[wi][j:j+bs].to(device)).argmax(1).cpu()
                ok += (p == wi).sum().item(); tot += len(p)
    return ok / max(tot, 1)


def main():
    ws = sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(ws)
    print(f"[ENTROPY 3-ARM] chargement...", flush=True)
    ts = set(l.strip() for l in open(os.path.join(SC, "testing_list.txt")) if l.strip())
    tr, te = {}, {}
    for wi, w in enumerate(ws):
        a = sorted(glob.glob(os.path.join(SC, w, "*.wav"))); rp, tp = [], []
        for p in a: (tp if os.path.relpath(p, SC) in ts else rp).append(p)
        if len(rp) >= 50 and len(tp) >= 5:
            tr[wi] = torch.stack([load_wav(p) for p in rp])
            te[wi] = torch.stack([load_wav(p) for p in tp])
    tr_flat = build_index(tr)
    print(f"  {NW} mots, {len(tr_flat)} samples train", flush=True)

    # identifier mots durs : entraîne un quick modèle baseline pour classer les mots par difficulté
    print("[ENTROPY 3-ARM] quick baseline pour mots durs...", flush=True)
    torch.manual_seed(0)
    m0 = M5Unified(NW).to(device); opt0 = torch.optim.AdamW(m0.parameters(), lr=1e-3, weight_decay=1e-4)
    for step in range(5000):
        bi = [random.choice(list(tr.keys())) for _ in range(BS)]
        wv = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi]).to(device)
        wi = torch.tensor(bi, device=device)
        opt0.zero_grad(); F.cross_entropy(m0(wv), wi).backward(); opt0.step()
    per_word = {wi: eval_acc(m0, te, [wi]) for wi in te}
    hard_words = sorted(per_word, key=lambda w: per_word[w])[:10]  # 10 + durs
    print(f"  mots durs (baseline faible) : {[ws[w] for w in hard_words]}", flush=True)
    del m0, opt0; torch.cuda.empty_cache()

    results = {}
    for mode in ['uniform', 'entropy', 'confidence']:
        acc_all, acc_hard, t = train_arm(mode, NW, tr, tr_flat, te, hard_words)
        results[mode] = {"acc_all": acc_all, "acc_hard": acc_hard, "t": t}
        print(f"  [{mode:10s}] all={acc_all*100:.1f}%  hard(10)={acc_hard*100:.1f}%  t={t/60:.1f}min", flush=True)
        torch.cuda.empty_cache()

    print("\n" + "="*64); print("VERDICT ENTROPY (active learning) :")
    a_all = results['entropy']['acc_all']; b_all = results['uniform']['acc_all']
    a_hard = results['entropy']['acc_hard']; b_hard = results['uniform']['acc_hard']
    print(f"  all : entropy {a_all*100:.1f}% vs uniform {b_all*100:.1f}%  (Δ {(a_all-b_all)*100:+.1f}pt)")
    print(f"  hard: entropy {a_hard*100:.1f}% vs uniform {b_hard*100:.1f}%  (Δ {(a_hard-b_hard)*100:+.1f}pt)")
    verdict = "GAGNE" if (a_hard > b_hard + 0.01) else ("cosmétique ✗" if abs(a_hard-b_hard) < 0.01 else "PERD")
    print(f"  => sur les mots durs : {verdict}")
    json.dump(results, open("ocm26400/entropy_active_learning_3arm_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
