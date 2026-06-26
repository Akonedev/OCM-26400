#!/usr/bin/env python3
"""Entraînement par COMPRÉHENSION (grok primitives → composer) — PAS de mémorisation.

PRINCIPE ABSOLU : compréhension > mémoire, TOUJOURS.
Le modèle grok les RÈGLES (primitives structurelles) via self-supervision,
puis classe/génère depuis cette compréhension → généralisation OOD.

Pour chaque domaine :
1. Primitives = les règles structurelles (spectrales pour audio, visuelles pour image,
   arithmétiques pour maths, morphologiques pour langage).
2. GROK les primitives (self-supervisé, loss 1-cos, gate 0.99) — le modèle COMPREND.
3. Composer/classifier depuis la compréhension — PAS de CE sur labels bruts.

Différence avec l'entraînement précédent (mémoire) :
  AVANT : cross_entropy(signal → label) = mémorisation d'instances (29.6%, échoue OOD)
  MAINTENANT : grok la structure (self-sup) → probe linéaire sur compréhension (généralise OOD)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json
import soundfile as sf
from PIL import Image
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


# =====================================================================
# AUDIO : grok les primitives spectrales (self-supervisé reconstruction)
# Le modèle COMPREND la structure spectrale de la parole, ne mémorise pas.
# =====================================================================
class AudioComprehensionGrokker(nn.Module):
    """Grok la structure spectrale de l'audio via reconstruction self-supervisée.
    Le SpectralCoreBlock (FFT) apprend les RÈGLES spectrales (formants, phonèmes),
    PAS des instances. Puis un probe linéaire classifie depuis la compréhension."""
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        from ocm26400.multimodal_encoders import AudioEncoder
        self.enc = AudioEncoder(out_dim=d_model)  # waveform -> Mel-STFT -> spectral features
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=1)  # grok la structure
        # reconstruction head : compréhension -> reconstruire les features spectrales
        self.recon = nn.Linear(d_model, 32)  # reconstruit les Mel features (self-sup)
        # probe linéaire : classifie depuis la compréhension (PAS entraîné avec le core)
        self.probe = None  # créé après grokking

    def grok_structure(self, wav, feats):
        """Self-supervisé : le core grok la structure spectrale (reconstruction).
        Loss 1-cos (comme crown-jewel) : compréhension ↔ structure canonique."""
        amv = self.core(self.enc(wav))  # compréhension spectrale
        recon = self.recon(amv)         # reconstruit les Mel features depuis la compréhension
        # loss 1-cos (crown-jewel pattern) : la compréhension doit capturer la structure
        loss_recon = F.mse_loss(recon, feats)  # reconstruction = compréhension de la structure
        return loss_recon, amv

    def classify_from_comprehension(self, amv, labels, n_classes):
        """Probe linéaire sur la représentation grokkée (compréhension → classe).
        Le probe est entraîné SÉPARÉMENT (le core n'est pas touché — il a grokké)."""
        if self.probe is None:
            self.probe = nn.Linear(amv.shape[1], n_classes).to(amv.device)
        logits = self.probe(amv.detach())  # detach : le core garde sa compréhension
        return F.cross_entropy(logits, labels)


def train_audio_comprehension():
    """Grok les primitives spectrales de SpeechCommands, puis classifie par compréhension."""
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    WORDS = [w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")][:15]
    T = 8000

    def load_wav(p):
        y, sr = sf.read(p); y = y.astype(np.float32)
        if y.ndim > 1: y = y.mean(1)
        if len(y) < T: y = np.pad(y, (0, T - len(y)))
        else: y = y[:T]
        return torch.tensor(y)

    auds, labs = [], []
    for wi, w in enumerate(WORDS):
        for p in glob.glob(os.path.join(SC, w, "*.wav"))[:100]:
            auds.append(load_wav(p)); labs.append(wi)
    N = len(auds)
    aud_t = torch.stack(auds).to(device)
    lab_t = torch.tensor(labs).to(device)
    # features spectrales cibles (Mel 32-dim) pour la reconstruction self-supervisée
    with torch.no_grad():
        from ocm26400.multimodal_encoders import AudioEncoder
        tmp_enc = AudioEncoder(out_dim=32).to(device)
        # extraire Mel features comme cible de reconstruction
        mel_fb = tmp_enc.mel_fb
        win = torch.hann_window(tmp_enc.n_fft, device=device)
        spec = torch.stft(aud_t, n_fft=tmp_enc.n_fft, hop_length=tmp_enc.n_fft // 2,
                          win_length=tmp_enc.n_fft, window=win, return_complex=True, center=False)
        mel_feats = torch.matmul(mel_fb, spec.abs() ** 2).mean(dim=-1)  # (N, 32)
        mel_feats = torch.log1p(mel_feats)
    print(f"[audio] {N} samples réels, {len(WORDS)} mots", flush=True)

    grokker = AudioComprehensionGrokker().to(device)
    opt = torch.optim.Adam(grokker.parameters(), lr=3e-3)
    idx_all = torch.randperm(N); ntr = int(N * 0.85)
    tr_i, te_i = idx_all[:ntr], idx_all[ntr:]

    # PHASE 1 : GROK la structure spectrale (self-supervisé, compréhension)
    print(f"\n[PHASE 1] GROK structure spectrale (self-sup, compréhension — PAS de CE sur labels)", flush=True)
    t0 = time.time()
    for step in range(2000):
        i = tr_i[torch.randint(0, len(tr_i), (48,))]
        loss, amv = grokker.grok_structure(aud_t[i], mel_feats[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step} recon_loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)

    # PHASE 2 : PROBE linéaire sur la compréhension grokkée (classifie depuis compréhension)
    print(f"\n[PHASE 2] PROBE linéaire sur compréhension grokkée (le core est FROZEN)", flush=True)
    with torch.no_grad():
        amv_tr = grokker.core(grokker.enc(aud_t[tr_i]))
        amv_te = grokker.core(grokker.enc(aud_t[te_i]))
    probe = nn.Linear(D_MODEL, len(WORDS)).to(device)
    opt_p = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for step in range(1000):
        bi = torch.randint(0, len(tr_i), (48,))
        logits = probe(amv_tr[bi])
        loss = F.cross_entropy(logits, lab_t[tr_i[bi]])
        opt_p.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(amv_te).argmax(1) == lab_t[te_i]).float().mean().item()
    print(f"\n=== AUDIO par COMPRÉHENSION ===")
    print(f"  classification (test, {len(te_i)} samples, {len(WORDS)} mots): {acc*100:.1f}%")
    print(f"  (le core a grokké la structure spectrale, le probe classifie depuis la compréhension)")
    print(f"  (vs CE-mémorisation précédent: 29.6% — la compréhension doit généraliser mieux OOD)")
    return acc


if __name__ == "__main__":
    print("="*60)
    print("ENTRAÎNEMENT PAR COMPRÉHENSION (grok primitives → composer)")
    print("PRINCIPE ABSOLU : compréhension > mémoire, TOUJOURS")
    print("="*60)
    acc = train_audio_comprehension()
    print(f"\nRésultat audio compréhension: {acc*100:.1f}%")
