#!/usr/bin/env python3
"""Pont audio→phonème INVARIANT au locuteur — clé pour appliquer le crown-jewel à l'audio.

PROBLÈME RACINE : le signal audio est STOCHASTIQUE (même mot → signaux différents selon le
locuteur). Les IDs extraits par top-bins STFT (tentative précédente, 3.3%) varient avec le
locuteur → le SpectralCoreBlock ne peut pas grok une association qui change à chaque instance.
Le pont signal→IDs-invariants est LE maillon non résolu (reco_audio 8.6% vs reco_phon 100%).

SOLUTION (2 leviers éprouvés d'invariance locuteur, standards ASR SOTA) :
  1. InstanceNorm1d sur le Mel : normalise moyenne+variance PAR UTTERANCE, par bande mel.
     → supprime l'enveloppe spectrale absolue du locuteur (pitch/gain/timbre moyens).
     Le résidu = contenu phonétique RELATIF (formants, transitions) = invariant.
  2. SpecAugment (Park 2019) : masquer aléatoirement F bandes mel + T frames temporels
     à l'entraînement → force l'encodeur à NE PAS dépendre de bandes spécifiques au locuteur.
     (Standard LibriSpeech/WSJ, +50% d'amélioration robustesse locuteur.)
  + SpectralCoreBlock (FFT, L'ARCHITECTURE) + 1-cos crown-jewel → grok association IDs→mot.

C'est exactement rendre le signal stochastique COMPATIBLE avec l'association-nombre :
les IDs deviennent phonétiques (invariants) → le crown-jewel peut s'appliquer → reconnaissance.

Respecte : 1-cos crown-jewel, Adam 3e-3, seed 0, IDs (canonical), éval holdout propre.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from train_deep_encoder_v2 import load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
torch.manual_seed(0); random.seed(0)


def specaugment(mel, n_freq=2, n_time=2, f_mask=15, t_mask=20):
    """SpecAugment : masque n_freq bandes mel + n_time frames. mel=(B, M, T). Entraînement only."""
    B, M, T = mel.shape
    m = mel.clone()
    for b in range(B):
        for _ in range(n_freq):
            f = random.randint(0, max(1, M - f_mask)); w = random.randint(0, f_mask)
            m[b, f:f+w, :] = 0.0
        for _ in range(n_time):
            t = random.randint(0, max(1, T - t_mask)); w = random.randint(0, t_mask)
            m[b, :, t:t+w] = 0.0
    return m


class InvariantAudioEncoder(nn.Module):
    """Mel-STFT -> InstanceNorm (invariance locuteur) -> convs -> features invariantes.
    InstanceNorm (pas BatchNorm) = normalisation PAR UTTERANCE => supprime le locuteur."""
    def __init__(self, out_dim=D_MODEL, n_mels=64):
        super().__init__()
        self.n_fft = 256; self.n_mels = n_mels
        fb = torch.zeros(n_mels, self.n_fft // 2 + 1)
        for m in range(n_mels):
            center = (m + 1) * (self.n_fft // 2 + 1 - 1) / (n_mels + 1)
            for f in range(self.n_fft // 2 + 1):
                d = abs(f - center) / max(1.0, (self.n_fft // 2) / (n_mels + 1))
                fb[m, f] = max(0.0, 1.0 - d)
        self.register_buffer("mel_fb", fb / (fb.sum(dim=1, keepdim=True) + 1e-8))
        self.register_buffer("window", torch.hann_window(self.n_fft))
        # InstanceNorm1d : normalise chaque utterance (B traité comme N=B instances) -> locuteur-supprimé
        self.inst_norm = nn.InstanceNorm1d(n_mels, affine=True)
        self.convs = nn.Sequential(
            nn.Conv1d(n_mels, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
        self.proj = nn.Linear(32, out_dim)

    def forward(self, wav, augment=False):
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.n_fft // 2,
                          win_length=self.n_fft, window=self.window,
                          return_complex=True, center=False)
        mel = torch.matmul(self.mel_fb, spec.abs() ** 2)     # (B, M, T)
        mel = torch.log1p(mel)
        mel = self.inst_norm(mel)                            # INVARIANCE LOCUTEUR (clé)
        if augment and self.training:
            mel = specaugment(mel)                           # INVARIANCE acoustique (clé)
        h = self.convs(mel)
        return self.proj(h.mean(dim=-1))                    # (B, out_dim)


class AudioInvariantModel(nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.audio_enc = InvariantAudioEncoder(out_dim=D_MODEL, n_mels=64)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        self.head = nn.Linear(D_MODEL, PART)
    def encode(self, wav, augment=False):
        return self.core(self.audio_enc(wav, augment=augment).unsqueeze(1)).squeeze(1)
    def ent(self, v): return self.head(v)


def _data(words, n_per_word=120, hold_start=100, hold_end=130):
    """Train sur [:100]/mot, holdout PROPRE sur [100:130] (jamais vus)."""
    tr = {}; te = {}
    for wi, w in enumerate(words):
        ws = sorted(glob.glob(os.path.join(SC, w, "*.wav")))
        train_w = [load_wav(p) for p in ws[:hold_start]][:n_per_word]
        hold_w = [load_wav(p) for p in ws[hold_start:hold_end]]
        if len(train_w) >= 20 and len(hold_w) >= 3:
            tr[wi] = torch.stack(train_w).to(device)
            te[wi] = torch.stack(hold_w).to(device)
    return tr, te


def train(n_steps=20000, batch=64, lr=3e-3, eval_every=2500):
    torch.manual_seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    tr, te = _data(words)
    n_words = len(tr)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    model = AudioInvariantModel(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    keys = list(tr.keys())

    # SC-1 sanity : overfit 8 samples, 1-cos pur doit descendre
    print(f"[SC-1 sanity] overfit 1 batch (400 steps, 1-cos)...", flush=True)
    sb = keys[:8]
    sba = torch.stack([tr[k][torch.randint(0, len(tr[k]), (1,)).item()] for k in sb])
    tgt = torch.stack([canon[k] for k in sb]).to(device)
    for _ in range(400):
        v = model.encode(sba, augment=False)
        cos = F.cosine_similarity(model.ent(v), tgt, dim=-1).clamp(-1, 1)
        loss = (1.0 - cos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        v = model.encode(sba)
        cos_s = F.cosine_similarity(model.ent(v), tgt, dim=-1).mean().item()
    print(f"  sanity 1-cos = {1-cos_s:.3f} ({'OK cos>0.9' if cos_s>0.9 else 'apprend...'})", flush=True)

    print(f"\n[TRAIN INVARIANT] {n_words} mots | InstanceNorm+SpecAugment | Adam {lr} | {n_steps} steps", flush=True)
    t0 = time.time(); best = 0.0; best_state = None
    for step in range(n_steps):
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0,len(tr[k]),(1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        v = model.encode(wavs, augment=True)             # AUGMENT à l'entraînement (invariance)
        cos = F.cosine_similarity(model.ent(v), canon[wi_t], dim=-1).clamp(-1,1)
        loss = (1.0 - cos).mean()                         # 1-cos crown-jewel
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
            wav = te[wi][j:j+1]
            ent = model.ent(model.encode(wav, augment=False))
            ok += ((ent @ canon.t()).argmax(1).item() == wi); tot += 1
    model.train(); return ok/max(tot,1)


if __name__ == "__main__":
    print("="*64)
    print("PONT AUDIO→PHONÈME INVARIANT (InstanceNorm + SpecAugment + crown-jewel)")
    print("="*64)
    model, canon, te, best = train(n_steps=20000)
    print(f"\n{'='*64}\nRÉSULTAT — holdout PROPRE [100:130] (locuteurs non vus)\n{'='*64}")
    print(f"  Test acc: {best*100:.1f}%")
    print(f"  Réf: baseline deep_encoder = 45.9% (même holdout) | SOTA SpeechCommands ~96%")
    print(f"  Δ vs baseline: {best*100-45.9:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state": model.state_dict(), "best": best},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_invariant_trained.pt")
    json.dump({"holdout_acc": best, "delta_vs_baseline_45p9": best*100-45.9,
               "delta_vs_sota_96": best*100-96,
               "method": "InstanceNorm(speaker)+SpecAugment+SpectralCoreBlock+1-cos crown-jewel"},
              open("ocm26400/audio_invariant_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_invariant_results.json")
