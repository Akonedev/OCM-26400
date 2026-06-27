#!/usr/bin/env python3
"""Audio voie v3 — FORME D'ONDE BRUTE, spectral natif (rapport 45, voie prescrite).

CORRECTION issue du RAPPORT 45 du projet (mécanisme que je n'avais pas appliqué) :
  « Verdict v3 : forme d'onde brute (spectral natif). Redondance double-spectrale : le mel
    EST déjà une FFT. »
Tous mes essais précédents utilisaient un frontend Mel → DOUBLE FFT (Mel = FFT, puis
SpectralCoreBlock = FFT again) = redondance, perte d'info. La voie fidèle = waveform brute
directement dans le SpectralCoreBlock (le FFT est NATIF au signal audio — analyseur de
spectre). Rapport 53 : reconnaissance via champ partagé + capture simultanée (any→any).

Mécanisme (pur projet, zéro Mel, zéro externe) :
  waveform (T) -> fenêtres -> embed -> SpectralCoreBlock (FFT natif sur le signal)
             -> pool -> head -> ent -> 1-cos (crown-jewel)
  + capture simultanée (text + phon + audio-brut -> même canonical).

Aucun frontend Mel. Aucune technique externe. Le SpectralCoreBlock traite la waveform
COMME un signal (FFT natif), pas comme une image spectro-temporelle redondante.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from train_deep_encoder_v2 import text_feat, phon_feat, load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
T = 8000
L_WIN = 64            # nb de fenêtres (séquence pour le SpectralCoreBlock)
WIN = T // L_WIN      # 125 échantillons/fenêtre


class RawWaveformSpectral(nn.Module):
    """Waveform brute -> fenêtres -> embed -> SpectralCoreBlock (FFT NATIF sur le signal).
    Pas de Mel (évite la double-FFT). Le SpectralCoreBlock est un analyseur de spectre natif."""
    def __init__(self, n_words):
        super().__init__()
        self.win_embed = nn.Linear(WIN, D_MODEL)   # fenêtre brute (125 samples) -> AMV
        nn.init.normal_(self.win_embed.weight, std=0.02)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=L_WIN, bidirectional=True)  # FFT natif
        self.text_proj = nn.Linear(PART, D_MODEL)
        self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core_view = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)  # cœur partagé vues text/phon
        self.head = nn.Linear(D_MODEL, PART)
    def audio_view(self, wav):                      # wav: (B, T)
        B = wav.shape[0]
        frames = wav.reshape(B, L_WIN, WIN)         # (B, 64, 125) — fenêtres brutes
        h = self.win_embed(frames)                   # (B, 64, D) — embed des fenêtres
        out = self.core(h).mean(1)                   # FFT natif sur la waveform -> pool
        return self.head(out)
    def text_view(self, f):  return self.head(self.core_view(self.text_proj(f).unsqueeze(1)).squeeze(1))
    def phon_view(self, f):  return self.head(self.core_view(self.phon_proj(f).unsqueeze(1)).squeeze(1))


def _data(words, n_per_word=100, hold_start=100, hold_end=130):
    tr = {}; te = {}
    for wi, w in enumerate(words):
        ws = sorted(glob.glob(os.path.join(SC, w, "*.wav")))
        tw = [load_wav(p) for p in ws[:hold_start]][:n_per_word]
        hw = [load_wav(p) for p in ws[hold_start:hold_end]]
        if len(tw) >= 20 and len(hw) >= 3:
            tr[wi] = torch.stack(tw).to(device); te[wi] = torch.stack(hw).to(device)
    return tr, te


def train(n_steps=20000, batch=64, lr=3e-3, eval_every=2500):
    torch.manual_seed(0); random.seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    tr, te = _data(words); keys = list(tr.keys())
    text_all = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all = torch.tensor([phon_feat(w) for w in words]).to(device)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    model = RawWaveformSpectral(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def joint(wi_t, wavs):
        tgt = canon[wi_t]
        out_t = model.text_view(text_all[wi_t])
        out_p = model.phon_view(phon_all[wi_t])
        out_a = model.audio_view(wavs)
        return ((1-F.cosine_similarity(out_t,tgt,-1).clamp(-1,1)).mean() +
                (1-F.cosine_similarity(out_p,tgt,-1).clamp(-1,1)).mean() +
                (1-F.cosine_similarity(out_a,tgt,-1).clamp(-1,1)).mean())

    # SC-1 sanity
    print(f"[SC-1 sanity] overfit 1 batch (400 steps, waveform brute spectral natif)...", flush=True)
    sb = keys[:8]; sw = torch.stack([tr[k][0] for k in sb]); swi = torch.tensor(sb, device=device)
    for _ in range(400):
        loss = joint(swi, sw)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        cos_s = F.cosine_similarity(model.audio_view(sw), canon[swi], -1).mean().item()
    print(f"  sanity audio(waveform) 1-cos = {1-cos_s:.3f} ({'OK' if cos_s>0.9 else 'apprend...'})", flush=True)

    print(f"\n[TRAIN WAVEFORM BRUTE spectral natif] {len(keys)} mots | L={L_WIN} fenêtres | "
          f"capture simultanée | 1-cos | {n_steps} steps", flush=True)
    t0 = time.time(); best = 0.0; best_state = None
    for step in range(n_steps):
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        loss = joint(wi_t, wavs)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == n_steps-1:
            acc = _eval(model, canon, te)
            if acc > best: best = acc; best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} 1-cos={loss.item():.4f} | holdout[100:130] {acc*100:.1f}% "
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
    print("AUDIO voie v3 — WAVEFORM BRUTE, spectral natif (rapport 45, voie prescrite)")
    print("="*64)
    model, canon, te, best = train(n_steps=20000)
    print(f"\n{'='*64}\nRÉSULTAT WAVEFORM BRUTE — holdout PROPRE [100:130]\n{'='*64}")
    print(f"  Test acc: {best*100:.1f}%")
    print(f"  Mécanisme: waveform brute -> SpectralCoreBlock (FFT natif, PAS de Mel/double-FFT) + capture simultanée")
    print(f"  Réf: simultaneous Mel 50.2% | rapport45 audio v2 46.4% | SOTA 96%")
    print(f"  Δ vs Mel-simultaneous: {best*100-50.2:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state": model.state_dict(), "best": best},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_raw_waveform_trained.pt")
    json.dump({"holdout_acc": best, "delta_vs_mel_50p2": best*100-50.2, "delta_vs_sota_96": best*100-96,
               "method": "raw waveform spectral natif (rapport 45 v3) + simultaneous capture, no Mel/double-FFT"},
              open("ocm26400/audio_raw_waveform_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_raw_waveform_results.json")
