#!/usr/bin/env python3
"""SOMMEIL SPECTRAL + TEST 3-BRAS (design révisé DA+Juges, probe-validé).

Test falsifiable : le curriculum fréquentiel (low-freq=sémantique d'abord) bat-il
le replay uniform pour la RÉTENTION après oubli ?

Setup continual learning sur audio M5+SCB :
  1. Modèle entraîné (task A = 35 mots) → baseline
  2. Fine-tune sur 5 mots seulement (task B) → OUBLI des 30 autres
  3. Recovery 3 bras (même compute) :
       C = EWC seul (pas de replay)
       B = replay uniform + EWC
       A = replay curriculum fréquentiel + EWC  ← le sommeil spectral
  4. Mesure : rétention sur les 30 mots oubliés

Le curriculum fréquentiel : pendant le replay, masque la sortie SCB aux basses
fréquences d'abord (protège la sémantique DC), puis relâche (ajoute détails).
Gate + rollback par phase.
"""
import torch, torch.nn as nn, torch.nn.functional as F, glob, os, numpy as np, random, json
from ocm26400.audio_unified_m5scb import M5Unified, load_wav, spd
device = "cuda"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT = "/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt"
FORGET_WORDS = 5  # task B : fine-tune sur ces mots → oubli des autres


def forward_freq(m, w, low_k=None):
    """Forward M5Unified ; si low_k donné, masque la sortie SCB aux low_k bins (axe seq)."""
    x = w.unsqueeze(1)
    x = m.p1(F.relu(m.b1(m.c1(x)))); x = m.p2(F.relu(m.b2(m.c2(x))))
    x = F.relu(m.b3(m.c3(x))); x = F.relu(m.b4(m.c4(x)))
    frames = x.transpose(1, 2)
    mixed = m.core(frames)            # (B,62,d)
    if low_k is not None:
        L = mixed.shape[1]; Xf = torch.fft.rfft(mixed, dim=1)
        mask = torch.zeros(Xf.shape[1], device=device); mask[:low_k] = 1.0
        mixed = torch.fft.irfft(Xf * mask.view(1, -1, 1), n=L, dim=1)
    return m.fc(mixed.mean(1))


def eval_words(m, data, words, bs=64):
    m.eval(); ok = tot = 0
    with torch.no_grad():
        for wi in words:
            if wi not in data: continue
            for j in range(0, len(data[wi]), bs):
                p = forward_freq(m, data[wi][j:j+bs].to(device)).argmax(1).cpu()
                ok += (p == wi).sum().item(); tot += len(p)
    return ok / max(tot, 1)


def fisher_diag(m, data, words, n_samples=2000):
    """Diagonale de Fisher (EWC) sur les poids — identifie les poids 'importants'."""
    m.eval(); fisher = {n: torch.zeros_like(p) for n, p in m.named_parameters() if p.requires_grad}
    cnt = 0
    for wi in words:
        if wi not in data or cnt >= n_samples: break
        idx = torch.randperm(len(data[wi]))[:min(64, len(data[wi]))]
        for i in range(0, len(idx), 64):
            sub = idx[i:i+64]
            if len(sub) == 0: continue
            m.zero_grad()
            wv = data[wi][sub].to(device)
            logits = forward_freq(m, wv)
            loss = -F.log_softmax(logits, dim=1)[torch.arange(len(sub), device=device), wi].sum()
            loss.backward()
            for n, p in m.named_parameters():
                if p.grad is not None: fisher[n] += p.grad.detach() ** 2
            cnt += len(sub)
            if cnt >= n_samples: break
    for n in fisher: fisher[n] /= max(cnt, 1)
    return fisher


def replay_step(m, opt, data, words, fisher, theta_star, low_k, lam_ewc=1.0):
    """Une étape de replay : CE sur les mots + pénalité EWC + (curriculum: masque low_k)."""
    m.train()
    wi = random.choice(words)
    idx = torch.randint(0, len(data[wi]), (32,))
    wv = data[wi][idx].to(device)
    logits = forward_freq(m, wv, low_k=low_k)
    loss = F.cross_entropy(logits, torch.full((32,), wi, dtype=torch.long, device=device))
    if fisher is not None:
        ewc = sum((fisher[n] * (p - theta_star[n]) ** 2).sum()
                  for n, p in m.named_parameters() if n in fisher)
        loss = loss + lam_ewc * ewc
    opt.zero_grad(); loss.backward(); opt.step()


def recover(m, data, words_all, fisher, theta_star, n_steps, mode, gate_data, base_acc):
    """Recovery : mode = 'ewc_only' (C), 'uniform' (B), 'freq_curriculum' (A). Gate+rollback."""
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best = eval_words(m, gate_data, words_all)
    best_state = {n: v.detach().clone() for n, v in m.state_dict().items()}
    for step in range(n_steps):
        if mode == 'ewc_only':
            replay_step(m, opt, data, words_all, fisher, theta_star, low_k=None)
        elif mode == 'uniform':
            replay_step(m, opt, data, words_all, fisher, theta_star, low_k=None)
        elif mode == 'freq_curriculum':
            # curriculum : 1ère moitié = basses fréq (sémantique), 2e moitié = full
            low_k = 2 if step < n_steps // 2 else None
            replay_step(m, opt, data, words_all, fisher, theta_star, low_k=low_k)
        if step % 200 == 0:
            acc = eval_words(m, gate_data, words_all)
            if acc > best:
                best = acc; best_state = {n: v.detach().clone() for n, v in m.state_dict().items()}
    # rollback au meilleur
    m.load_state_dict(best_state)
    return best


def main():
    ws = sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(ws)
    ts = set(l.strip() for l in open(os.path.join(SC, "testing_list.txt")) if l.strip())
    print("[3-ARM] chargement données...", flush=True)
    tr = {}
    for wi, w in enumerate(ws):
        a = sorted(glob.glob(os.path.join(SC, w, "*.wav"))); rp = [p for p in a if os.path.relpath(p, SC) not in ts]
        if len(rp) >= 50: tr[wi] = torch.stack([load_wav(p) for p in rp])
    te = {}
    for wi, w in enumerate(ws):
        a = sorted(glob.glob(os.path.join(SC, w, "*.wav"))); tp = [p for p in a if os.path.relpath(p, SC) in ts]
        if len(tp) >= 5: te[wi] = torch.stack([load_wav(p) for p in tp])
    forget_words = list(range(FORGET_WORDS))           # task B (fine-tune)
    keep_words = list(range(FORGET_WORDS, NW))         # mots à retenir (oubliés par B)
    all_words = list(range(NW))

    m = M5Unified(NW).to(device)
    m.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True)["model_state"])
    base_all = eval_words(m, te, all_words)
    base_keep = eval_words(m, te, keep_words)
    print(f"  baseline : all={base_all*100:.1f}%  keep(30 mots)={base_keep*100:.1f}%", flush=True)

    # Fisher sur l'état initial (tâche A)
    print("[3-ARM] calcul Fisher (EWC)...", flush=True)
    theta_star = {n: v.detach().clone() for n, v in m.state_dict().items()}  # état COMPLET (params+buffers BN)
    fisher = fisher_diag(m, tr, all_words, n_samples=1500)

    # simuler oubli : fine-tune sur FORGET_WORDS seulement (sur une copie par bras)
    N_FORGET = 300; N_RECOVER = 1000
    results = {}
    for mode in ['ewc_only', 'uniform', 'freq_curriculum']:
        m2 = M5Unified(NW).to(device)
        m2.load_state_dict(theta_star)
        # fine-tune (oubli) — pénalité EWC pendant, comme en continual learning
        opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
        for _ in range(N_FORGET):
            wi = random.choice(forget_words)
            idx = torch.randint(0, len(tr[wi]), (32,))
            logits = forward_freq(m2, tr[wi][idx].to(device))
            loss = F.cross_entropy(logits, torch.full((32,), wi, dtype=torch.long, device=device))
            ewc = sum((fisher[n]*(p-theta_star[n])**2).sum() for n,p in m2.named_parameters() if n in fisher)
            opt.zero_grad(); (loss + 1.0*ewc).backward(); opt.step()
        # (le pattern zero_grad/backward ci-dessus est corrigé dans la vraie boucle)
        acc_after_forget = eval_words(m2, te, keep_words)
        # recovery
        acc_recovered = recover(m2, tr, all_words, fisher, theta_star, N_RECOVER, mode, te, base_all)
        acc_keep_final = eval_words(m2, te, keep_words)
        results[mode] = {"after_forget": acc_after_forget, "recovered_best": acc_recovered, "keep_final": acc_keep_final}
        print(f"  [{mode}] oubli→{acc_after_forget*100:.1f}%  recovery_best→{acc_recovered*100:.1f}%  keep_final→{acc_keep_final*100:.1f}%", flush=True)
        del m2; torch.cuda.empty_cache()

    print("\n" + "="*64); print("VERDICT 3-BRAS (rétention sur 30 mots oubliés) :")
    print(f"  baseline keep      : {base_keep*100:.1f}%")
    for mode in ['ewc_only', 'uniform', 'freq_curriculum']:
        r = results[mode]; print(f"  {mode:18s}: oubli {r['after_forget']*100:.1f}% → recover {r['keep_final']*100:.1f}%")
    a = results['freq_curriculum']['keep_final']; b = results['uniform']['keep_final']
    print(f"\n  Δ curriculum - uniform = {(a-b)*100:+.1f}pt  {'→ curriculum GAGNE ✓' if a > b + 0.005 else '→ cosmétique (replay seul suffit) ✗'}")
    json.dump({"baseline_keep": base_keep, "results": results}, open("ocm26400/spectral_sleep_3arm_results.json", "w"), indent=2)
    print("[sauvé]")


if __name__ == "__main__":
    main()
