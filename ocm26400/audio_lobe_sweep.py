#!/usr/bin/env python3
"""SWEEP profondeur/largeur du lobe audio — trouve la taille IDEALE (cœur 675K fixé).

Question user : « quelle serait la taille idéale ? le nbre de param idéal ? prends en
compte la profondeur ». Réponse empirique : balaie n_blocks (profondeur, L4) et hidden
(largeur), mesure la holdout [100:130], identifie le genou (knee).

Loi unifiée D=k^1.98·P^1.06·d^-2.38 : cœur (raisonnement) idéal = petit/d=256 (fixé 675K);
lobe sensoriel = lever empirique (reconnaissance aime la capacité jusqu'au genou).

Chaque config : 8000 steps, 1-cos crown-jewel, Adam 3e-3, seed 0, InstanceNorm+SpecAugment.
"""
import torch, torch.nn.functional as F, os, json, time, random
from ocm26400.audio_deep_lobe import DeepAudioLobe
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.audio_invariant_ids import _data
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class SweepModel(nn.Module):
    def __init__(self, n_words, n_blocks, hidden, n_mels=128):
        super().__init__()
        self.lobe = DeepAudioLobe(n_mels=n_mels, hidden=hidden, n_blocks=n_blocks)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)   # CŒUR FIXE 675K
        self.head = nn.Linear(D_MODEL, PART)
    def forward(self, wav, augment=False):
        return self.head(self.core(self.lobe(wav, augment=augment).unsqueeze(1)).squeeze(1))


@torch.no_grad()
def _eval(model, canon, te):
    model.eval(); ok = tot = 0
    for wi in te:
        for j in range(len(te[wi])):
            ok += ((model(te[wi][j:j+1]) @ canon.t()).argmax(1).item() == wi); tot += 1
    model.train(); return ok/max(tot,1)


def run_config(n_blocks, hidden, words, tr, te, canon, keys, n_steps=8000, batch=64):
    torch.manual_seed(0); random.seed(0)
    model = SweepModel(len(words), n_blocks, hidden).to(device)
    lobe_p = sum(p.numel() for p in model.lobe.parameters())
    core_p = sum(p.numel() for p in model.core.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    best = 0.0; t0 = time.time()
    for step in range(n_steps):
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        ent = model(wavs, augment=True)
        cos = F.cosine_similarity(ent, canon[wi_t], -1).clamp(-1,1)
        loss = (1-cos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4000 == 0 or step == n_steps-1:
            acc = _eval(model, canon, te); best = max(best, acc)
    return best, lobe_p, core_p, time.time()-t0


def main():
    torch.manual_seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    tr, te = _data(words); keys = list(tr.keys())
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)

    # SWEEP : profondeur (L4) d'abord, puis largeur
    configs = [(2,128),(4,128),(8,128),(16,128),(8,64),(8,256)]
    print(f"SWEEP lobe audio — cœur fixé, {len(configs)} configs × 8000 steps\n", flush=True)
    results = []
    for nb, hd in configs:
        acc, lp, cp, dt = run_config(nb, hd, words, tr, te, canon, keys)
        tot = lp + cp + sum(p.numel() for p in SweepModel(NW,nb,hd).head.parameters())
        print(f"  n_blocks={nb:>2} hidden={hd:>3} | lobe={lp/1e3:>5.0f}K cœur={cp/1e3:.0f}K "
              f"total={tot/1e3:>5.0f}K | holdout {acc*100:>5.1f}% | {dt:.0f}s", flush=True)
        results.append({"n_blocks":nb,"hidden":hd,"lobe_K":lp/1e3,"total_K":tot/1e3,"holdout":acc})
    best = max(results, key=lambda r: r["holdout"])
    print(f"\n{'='*60}\nSWEEP — IDEAL\n{'='*60}")
    print(f"  Meilleur: n_blocks={best['n_blocks']} hidden={best['hidden']} "
          f"-> holdout {best['holdout']*100:.1f}% (lobe {best['lobe_K']:.0f}K, total {best['total_K']:.0f}K)")
    print(f"  Cœur raisonnement: FIXE 675K (toutes configs, conforme règle)")
    print(f"  Réf baseline 45.9% | SOTA 96%")
    # courbe profondeur (hidden=128)
    depth_curve = [(r["n_blocks"], r["holdout"]) for r in results if r["hidden"]==128]
    print(f"  Courbe PROFONDEUR (hidden=128): " + " -> ".join(f"b{b}:{a*100:.0f}%" for b,a in depth_curve))
    json.dump({"configs":results, "best":best, "core_fixed_K":cp/1e3,
               "baseline_45p9":45.9, "method":"depth/width sweep, core fixed 675K"},
              open("ocm26400/audio_lobe_sweep_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_lobe_sweep_results.json")


if __name__ == "__main__":
    main()
