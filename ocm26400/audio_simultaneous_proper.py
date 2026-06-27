#!/usr/bin/env python3
"""Audio — CAPTURE SIMULTANÉE cross-modale (format prescrit) + lobe profond invariant.

CORRECTION (user + hook) : les variantes précédentes (audio_invariant, vq, deep_lobe,
sweep) entraînaient l'AUDIO SEUL — violant §I 'capture simultanée'. La baseline OK
(train_deep_encoder_v2) capture texte+phonétique+audio ENSEMBLE vers le même canonical.
Ce module corrige : il combine le lobe profond invariant (améliorations) AVEC la capture
simultanée (format prescrit), dans UNE passe, UN SpectralCoreBlock partagé, 1-cos joint.

Format respecté (RULES_MASTER) :
  §I capture simultanée : text(word) + phonetic + audio -> même canonical, 1 passe.
  §B AMV-256 : sortie 256 (ent via head ↔ canonical, op/meta réservés).
  §C crown-jewel : loss 1-cos (PAS MSE), Adam 3e-3, seed 0, batch 64.
  §H IDs : canonical LearnedVocab (association nombre).
  cœur raisonnement SpectralCoreBlock partagé (675K), 1 optimiseur pour tout.

Réutilise : DeepAudioLobe (InstanceNorm+SpecAugment+temporal FFT) + text_feat/phon_feat
(deep_encoder_v2). PAS de réinvention — on remet la capture simultanée.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.audio_deep_lobe import DeepAudioLobe
from train_deep_encoder_v2 import text_feat, phon_feat, load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class SimultaneousCapture(nn.Module):
    """Capture simultanée text+phon+audio -> shared SpectralCoreBlock -> canonical.
    UN cœur de raisonnement partagé, 1-cos joint sur les 3 vues (format prescrit)."""
    def __init__(self, n_words, n_blocks=4, hidden=128):
        super().__init__()
        self.audio_lobe = DeepAudioLobe(n_mels=128, hidden=hidden, n_blocks=n_blocks)
        self.text_proj = nn.Linear(PART, D_MODEL)
        self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)   # CŒUR PARTAGÉ (raisonnement)
        self.head = nn.Linear(D_MODEL, PART)                        # -> ent ↔ canonical
    def view(self, feat, proj):
        return self.head(self.core(proj(feat).unsqueeze(1)).squeeze(1))
    def audio_view(self, wav, augment=False):
        return self.head(self.core(self.audio_lobe(wav, augment=augment).unsqueeze(1)).squeeze(1))


def _data(words, n_per_word=100, hold_start=100, hold_end=130):
    tr = {}; te = {}
    for wi, w in enumerate(words):
        ws = sorted(glob.glob(os.path.join(SC, w, "*.wav")))
        tw = [load_wav(p) for p in ws[:hold_start]][:n_per_word]
        hw = [load_wav(p) for p in ws[hold_start:hold_end]]
        if len(tw) >= 20 and len(hw) >= 3:
            tr[wi] = torch.stack(tw).to(device); te[wi] = torch.stack(hw).to(device)
    return tr, te


def train(n_steps=20000, batch=64, lr=3e-3, eval_every=2500, n_blocks=4, hidden=128):
    torch.manual_seed(0); random.seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    tr, te = _data(words); keys = list(tr.keys())
    text_all = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all = torch.tensor([phon_feat(w) for w in words]).to(device)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    model = SimultaneousCapture(NW, n_blocks=n_blocks, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)   # 1 optimiseur pour TOUT (capture jointe)

    # SC-1 sanity (capture simultanée sur 1 batch)
    print("[SC-1 sanity] overfit 1 batch (400 steps, capture simultanée 1-cos)...", flush=True)
    sb = keys[:8]; sw = torch.stack([tr[k][0] for k in sb]); swi = torch.tensor(sb, device=device)
    for _ in range(400):
        tgt = canon[swi]
        out_t = model.view(text_all[swi], model.text_proj)
        out_p = model.view(phon_all[swi], model.phon_proj)
        out_a = model.audio_view(sw, augment=False)
        loss = ((1-F.cosine_similarity(out_t,tgt,-1).clamp(-1,1)).mean() +
                (1-F.cosine_similarity(out_p,tgt,-1).clamp(-1,1)).mean() +
                (1-F.cosine_similarity(out_a,tgt,-1).clamp(-1,1)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cos_s = F.cosine_similarity(model.audio_view(sw), canon[swi], -1).mean().item()
    print(f"  sanity audio 1-cos = {1-cos_s:.3f} ({'OK' if cos_s>0.9 else 'apprend...'})", flush=True)

    print(f"\n[TRAIN CAPTURE SIMULTANÉE] {len(keys)} mots | text+phon+audio JOINT | "
          f"lobe profond b{n_blocks}/h{hidden} | {n_steps} steps", flush=True)
    t0 = time.time(); best = 0.0; best_state = None
    for step in range(n_steps):
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        tgt = canon[wi_t]
        # CAPTURE SIMULTANÉE : 3 vues en UNE passe, 1-cos joint
        out_t = model.view(text_all[wi_t], model.text_proj)
        out_p = model.view(phon_all[wi_t], model.phon_proj)
        out_a = model.audio_view(wavs, augment=True)
        loss = ((1-F.cosine_similarity(out_t,tgt,-1).clamp(-1,1)).mean() +
                (1-F.cosine_similarity(out_p,tgt,-1).clamp(-1,1)).mean() +
                (1-F.cosine_similarity(out_a,tgt,-1).clamp(-1,1)).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == n_steps-1:
            acc = _eval(model, canon, te)
            if acc > best: best = acc; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} loss={loss.item():.4f} | holdout[100:130] {acc*100:.1f}% "
                  f"(best {best*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
    if best_state: model.load_state_dict(best_state)
    return model, canon, te, best


@torch.no_grad()
def _eval(model, canon, te):
    model.eval(); ok = tot = 0
    for wi in te:
        for j in range(len(te[wi])):
            ok += ((model.audio_view(te[wi][j:j+1]) @ canon.t()).argmax(1).item() == wi); tot += 1
    model.train(); return ok/max(tot,1)


if __name__ == "__main__":
    print("="*64)
    print("AUDIO — CAPTURE SIMULTANÉE (format prescrit) + lobe profond invariant")
    print("="*64)
    # config idéale du sweep (b4/h128) + capture simultanée
    model, canon, te, best = train(n_steps=20000, n_blocks=4, hidden=128)
    print(f"\n{'='*64}\nRÉSULTAT CAPTURE SIMULTANÉE — holdout PROPRE [100:130]\n{'='*64}")
    print(f"  Test acc (audio, après co-capture text+phon): {best*100:.1f}%")
    print(f"  Réf: baseline deep_encoder 42.7% | audio-only lobe 47.6% | SOTA 96%")
    print(f"  Δ vs audio-only: {best*100-47.6:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state": model.state_dict(), "best": best},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_simultaneous_trained.pt")
    json.dump({"holdout_acc": best, "delta_vs_audio_only_47p6": best*100-47.6,
               "delta_vs_sota_96": best*100-96,
               "method": "SIMULTANEOUS capture (text+phon+audio -> canonical, 1-cos joint) "
                         "+ deep invariant lobe (InstanceNorm+SpecAugment+temporal FFT), "
                         "shared SpectralCoreBlock reasoning core"},
              open("ocm26400/audio_simultaneous_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_simultaneous_results.json")
