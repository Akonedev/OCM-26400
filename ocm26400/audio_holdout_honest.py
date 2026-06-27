#!/usr/bin/env python3
"""Éval HONNÊTE audio sur HOLDOUT PROPRE (wavs [100:130], jamais vus par la baseline).

Corrige le leak : deep_encoder a été entraîné sur wavs [:100]/mot. Toute éval dans [:100]
= train leak. Ici on évalue sur [100:130] = VRAIMENT non-vus par la baseline ET par
tout entraînement antérieur. Fixe seed pour reproductibilité. Compare baseline (K=1)
vs autocorrect (K=1 et K=4) sur le MÊME holdout => comparaison valide."""
import torch, torch.nn.functional as F, glob, os
from ocm26400.audio_autocorrect import AudioAutoCorrect, evaluate as _eval
from ocm26400.amv import PART
from ocm26400.learned_vocab import LearnedVocab
from train_deep_encoder_v2 import load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
HOLD_START, HOLD_END = 100, 130   # wavs JAMAIS vus par la baseline (qui a vu [:100])

def main():
    torch.manual_seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    # holdout propre : wavs [100:130] par mot
    hold = {}
    for wi, w in enumerate(words):
        ws = [load_wav(p) for p in sorted(glob.glob(os.path.join(SC, w, "*.wav")))[HOLD_START:HOLD_END]]
        if len(ws) >= 5:
            hold[wi] = torch.stack(ws).to(device)
    print(f"[HOLDOUT PROPRE] {sum(len(v) for v in hold.values())} wavs sur {len(hold)}/{NW} mots "
          f"(indices [{HOLD_START}:{HOLD_END}], jamais vus par baseline)", flush=True)

    # charge le modèle autocorrect (baseline rechargée + calib)
    model = AudioAutoCorrect(NW).to(device)
    ck = torch.load("/media/akone/SAVENVME2/Datasets/ocm26400/audio_autocorrect_trained.pt",
                    map_location=device, weights_only=True)
    model.load_state_dict(ck["model_state"]); model.eval()

    @torch.no_grad()
    def acc(K):
        ok = tot = 0
        for wi in hold:
            for j in range(len(hold[wi])):
                wav = hold[wi][j:j+1]
                ent = model.ent(model.encode(wav, K=K))
                ok += ((ent @ canon.t()).argmax(1).item() == wi); tot += 1
        return ok/max(tot,1), tot
    a1, n1 = acc(K=1); a4, n4 = acc(K=4)
    print(f"\n{'='*60}\nÉVAL HONNÊTE — HOLDOUT PROPRE [100:130] (baseline n'a vu que [:100])\n{'='*60}")
    print(f"  K_depth=1 : {a1*100:.1f}%  ({n1} wavs)")
    print(f"  K_depth=4 : {a4*100:.1f}%  ({n4} wavs)")
    print(f"  Réf baseline deep_encoder = 42.7% (sur SON split, [:100] 80/20)")
    print(f"  Δ K=1 vs 42.7%: {a1*100-42.7:+.1f}pt  |  Δ K=4: {a4*100-42.7:+.1f}pt")
    import json
    json.dump({"holdout":"[100:130]_clean","K1_acc":a1,"K4_acc":a4,
               "delta_K1_vs_42p7":a1*100-42.7,"delta_K4_vs_42p7":a4*100-42.7,
               "n_wavs":n1}, open("ocm26400/audio_holdout_honest_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_holdout_honest_results.json")

if __name__ == "__main__":
    main()
