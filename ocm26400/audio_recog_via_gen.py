#!/usr/bin/env python3
"""Reconnaissance VIA GÉNÉRATION — inverser l'asymétrie (le seul levier paradigm-aligned non testé).

Le paradigme GÉNÈRE depuis la compréhension à 97% (force). Jusqu'ici reconnaissance directe
(audio→mot, dur, 62.5%). ICI : analysis-by-synthesis — générer le Mel CANONIQUE de chaque
mot candidat (force 97%), puis matcher l'audio d'entrée aux 35 templates générés → mot.
Utilise la FORCE (génération) POUR la reconnaissance.

Reconstruction du générateur depuis rule_generation.pt (state dict):
  phon_embed(16,256) | understand SCS(s=16) | concept_head(64,256) | gen_proj(256,64)
  generate SCS(s=16) | mel_head(32,256) | rec_proj(256,32) | recognize SCS(s=16)
Représentation Mel = 32 bandes mel, moyennées sur frames (log-mel 32-dim summary).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, json
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT = "/media/akone/SAVENVME2/Datasets/ocm26400/rule_generation.pt"


class RuleGenReconstructed(nn.Module):
    """Reconstruit depuis le state dict. seq_len=16 (filter shape (9,256))."""
    def __init__(self):
        super().__init__()
        self.phon_embed = nn.Embedding(16, 256)
        self.understand = SpectralCoreBlock(256, seq_len=16, bidirectional=True)
        self.concept_head = nn.Linear(256, 64)
        self.gen_proj = nn.Linear(64, 256)
        self.generate = SpectralCoreBlock(256, seq_len=16, bidirectional=True)
        self.mel_head = nn.Linear(256, 32)
        self.rec_proj = nn.Linear(32, 256)
        self.recognize = SpectralCoreBlock(256, seq_len=16, bidirectional=True)
    def generate_mel(self, concept_64):               # concept (B,64) -> Mel (B,32)
        h = self.gen_proj(concept_64)                  # (B,256)
        h = h.unsqueeze(1).expand(-1, 16, -1)          # (B,16,256) broadcast
        h = self.generate(h).mean(dim=1)               # (B,256) pool
        return self.mel_head(h)                        # (B,32)


def mel32(wav_np, n_mels=32, n_fft=256, sr=16000):
    """Mel 32-dim summary (reconstruction de la repr. d'entraînement)."""
    if len(wav_np) < n_fft: wav_np = np.pad(wav_np, (0, n_fft-len(wav_np)))
    win = np.hanning(n_fft)
    # banc mel 32
    fb = np.zeros((n_mels, n_fft//2+1))
    for m in range(n_mels):
        c = (m+1)*(n_fft//2+1-1)/(n_mels+1)
        for f in range(n_fft//2+1): fb[m,f]=max(0,1-abs(f-c)/max(1,(n_fft//2)/(n_mels+1)))
    fb = fb/(fb.sum(1,keepdims=True)+1e-8)
    frames=[]
    for s in range(0, len(wav_np)-n_fft, n_fft//2):
        seg=wav_np[s:s+n_fft]*win; pw=np.abs(np.fft.rfft(seg))**2
        frames.append(fb@pw)
    if not frames: return np.zeros(n_mels, dtype=np.float32)
    m = np.log1p(np.array(frames).mean(0))            # (32,) summary
    return m.astype(np.float32)


def load_wav_np(p):
    y,sr=sf.read(p); y=y.astype(np.float32)
    if y.ndim>1: y=y.mean(1)
    return y


def main():
    ck = torch.load(CKPT, map_location=device, weights_only=True)
    words = ck["words"]; canon = ck["canon"].to(device)   # (n_words, 64) concept embeddings
    NW = len(words)
    model = RuleGenReconstructed().to(device)
    try:
        model.load_state_dict(ck["model_state"], strict=True)
        print("[RECONSTRUCTION] poids chargés strict=True OK")
    except Exception as e:
        print(f"[RECONSTRUCTION] strict load échoué ({str(e)[:80]}), essai strict=False")
        model.load_state_dict(ck["model_state"], strict=False)
    model.eval()

    # 1) GÉNÈRE le Mel canonique 32-dim de chaque mot (concept → generate → mel)
    with torch.no_grad():
        gen_mels = model.generate_mel(canon).cpu().numpy()   # (NW, 32)
    gen_mels = gen_mels / (np.linalg.norm(gen_mels, axis=1, keepdims=True)+1e-8)

    # 2) split officiel + extract Mel32 des wavs de test
    test_set = set(l.strip() for l in open(os.path.join(SC, "testing_list.txt")) if l.strip())
    ok = tot = 0
    by_word = {w: 0 for w in words}; tot_word = {w: 0 for w in words}
    for wi, w in enumerate(words):
        wavs = sorted(glob.glob(os.path.join(SC, w, "*.wav")))
        tested = 0
        for p in wavs:
            if os.path.relpath(p, SC) not in test_set: continue
            if tested >= 50: break                       # cap 50/mot pour la vitesse
            tested += 1
            m = mel32(load_wav_np(p))
            m = m / (np.linalg.norm(m)+1e-8)
            sims = gen_mels @ m                          # (NW,) cos aux templates générés
            pred = int(sims.argmax())
            ok += (pred == wi); tot += 1
            tot_word[w] += 1; by_word[w] += (pred == wi)
    acc = ok/max(tot,1)
    print(f"\n{'='*60}\nRECONNAISSANCE VIA GÉNÉRATION — test officiel\n{'='*60}")
    print(f"  Test acc: {acc*100:.1f}% ({ok}/{tot})")
    print(f"  Réf: reconnaissance directe 62.5% | SOTA 96%")
    print(f"  Δ vs directe: {acc*100-62.5:+.1f}pt | Δ vs SOTA: {acc*100-96:+.1f}pt")
    json.dump({"test_acc_official": acc, "delta_vs_direct_62p5": acc*100-62.5,
               "delta_vs_sota_96": acc*100-96,
               "method": "recognition via generation (analysis-by-synthesis): generate canonical Mel per word (97% generator), match test audio"},
              open("ocm26400/audio_recog_via_gen_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_recog_via_gen_results.json")


if __name__ == "__main__":
    main()
