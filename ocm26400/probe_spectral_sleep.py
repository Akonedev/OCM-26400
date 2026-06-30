#!/usr/bin/env python3
"""PROBE FALSIFIABLE — le sommeil spectral mérite-t-il d'exister ?

Hypothèse (des Juges) : sur l'axe SÉQUENCE, les basses fréquences des activations SCB
portent la SÉMANTIQUE, les hautes fréquences les DÉTAILS. Si vrai → un sommeil par bandes
FFT a du sens. Si faux → c'est de la cosmétique, on garde replay+EWC brut.

Test : sur le modèle audio M5+SCB (séquence 62 frames), pour chaque sample :
  1. extraire l'activation SCB (B,62,d)
  2. FFT axe séquence (dim=1) — JAMAIS axe features (ordre arbitraire)
  3. reconstruire low-pass (k basses fréq) et high-pass (le reste)
  4. classifier chaque reconstruction
Décision : low-pass préserve l'accuracy (sémantique) ET high-pass la perd (détail) ?
  → si OUI (low ≈ full, high << full) : low-freq=sémantique CONFIRMÉ, sommeil justifié
  → si NON : réfuté, abandonner le framing FFT
"""
import torch, torch.nn.functional as F, glob, os, numpy as np, json
from ocm26400.audio_unified_m5scb import M5Unified, load_wav
device = "cuda"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT = "/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt"


def freq_split_classify(m, wavs, canon_fc, low_k):
    """Pour un batch : FFT axe seq, reconstruit low-pass (low_k freq) + high-pass, classify."""
    # reproduire forward jusqu'au SCB output
    x = wavs.unsqueeze(1)
    x = m.p1(F.relu(m.b1(m.c1(x)))); x = m.p2(F.relu(m.b2(m.c2(x))))
    x = F.relu(m.b3(m.c3(x))); x = F.relu(m.b4(m.c4(x)))
    frames = x.transpose(1, 2)          # (B,62,d)
    mixed = m.core(frames)              # (B,62,d) — activation SCB
    B, L, D = mixed.shape
    # FFT axe séquence (dim=1)
    Xf = torch.fft.rfft(mixed, dim=1)   # (B, F, D), F = L//2+1 = 32
    F_tot = Xf.shape[1]
    # low-pass : garde les low_k premières fréquences
    mask_lo = torch.zeros(F_tot, device=device); mask_lo[:low_k] = 1.0
    Xf_lo = Xf * mask_lo.view(1, -1, 1)
    Xf_hi = Xf * (1 - mask_lo.view(1, -1, 1))
    mixed_lo = torch.fft.irfft(Xf_lo, n=L, dim=1)
    mixed_hi = torch.fft.irfft(Xf_hi, n=L, dim=1)
    # classifier (mean pool + fc)
    p_full = m.fc(mixed.mean(1))
    p_lo = m.fc(mixed_lo.mean(1))
    p_hi = m.fc(mixed_hi.mean(1))
    return p_full.argmax(1), p_lo.argmax(1), p_hi.argmax(1)


def main():
    ws = sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(ws)
    ts = set(l.strip() for l in open(os.path.join(SC, "testing_list.txt")) if l.strip())
    m = M5Unified(NW).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=True)
    m.load_state_dict(ck["model_state"]); m.eval()
    print(f"[PROBE] modèle chargé (best {ck.get('best',0)*100:.1f}%), seq=62 frames, F=32 bins", flush=True)

    # teste plusieurs coupes low_k (combien de basses fréq = sémantique ?)
    results = {}
    for low_k in [1, 2, 4, 8, 16]:
        ok_full = ok_lo = ok_hi = tot = 0
        for wi, w in enumerate(ws):
            a = sorted(glob.glob(os.path.join(SC, w, "*.wav")))
            tp = [p for p in a if os.path.relpath(p, SC) in ts]
            if len(tp) < 5: continue
            for j in range(0, len(tp), 64):
                batch = tp[j:j+64]
                wavs = torch.stack([load_wav(p) for p in batch]).to(device)
                pf, pl, ph = freq_split_classify(m, wavs, m.fc, low_k)
                ok_full += (pf.cpu() == wi).sum().item()
                ok_lo += (pl.cpu() == wi).sum().item()
                ok_hi += (ph.cpu() == wi).sum().item()
                tot += len(batch)
        acc_full = ok_full/tot; acc_lo = ok_lo/tot; acc_hi = ok_hi/tot
        results[low_k] = {"full": acc_full, "low": acc_lo, "high": acc_hi}
        print(f"  low_k={low_k:>2} bins: full={acc_full*100:5.1f}% | LOW-pass={acc_lo*100:5.1f}% | HIGH-pass={acc_hi*100:5.1f}%", flush=True)

    print("\n" + "="*64)
    print("VERDICT DU PROBE (low-freq = sémantique ?) :")
    # critère : existe-t-il un low_k où LOW ≈ full (sémantique préservée) ET HIGH << full (détail perdu) ?
    validated = False
    for low_k, r in results.items():
        lo_preserves = r["low"] >= r["full"] * 0.90   # LOW garde ≥90% de l'accuracy
        hi_loses = r["high"] <= r["full"] * 0.50      # HIGH perd ≥50%
        if lo_preserves and hi_loses:
            print(f"  low_k={low_k}: LOW={r['low']*100:.1f}% (≥90% full={r['full']*100:.1f}%) ✓ ET HIGH={r['high']*100:.1f}% (≤50%) ✓ → CONFIRMÉ")
            validated = True
    if validated:
        print("\n=> LOW-FREQ = SÉMANTIQUE CONFIRMÉ. Le sommeil spectral (curriculum fréquentiel) est JUSTIFIÉ.")
    else:
        print("\n=> NON confirmé : LOW et HIGH ne se séparent pas nettement. Sommeil spectral = cosmétique → replay+EWC brut.")
    json.dump(results, open("ocm26400/probe_spectral_sleep_results.json", "w"), indent=2)
    print("[sauvé] ocm26400/probe_spectral_sleep_results.json")


if __name__ == "__main__":
    main()
