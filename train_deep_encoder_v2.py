#!/usr/bin/env python3
"""Encodeur audio PROFOND + cross-modal simultané → vers SOTA.

PRINCIPE : profondeur > params (ton principe). L'encodeur actuel (1 conv) est trop
léger → features bruitées → le FFT ne peut pas grok. Un encodeur PROFOND
(4 conv layers) apprend l'INVARIANT statistique (ce qui fait qu'un mot EST ce mot,
indépendamment du locuteur). Le SpectralCoreBlock grok sur ces features propres.

Le cross-modal simultané (texte + phonétique + audio profond → même ID) ancre
l'apprentissage de l'invariant. La 1-cos loss (crown-jewel) guide le grokking.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
T = 8000


class DeepAudioEncoder(nn.Module):
    """Encodeur audio PROFOND : Mel-STFT → 4 conv layers → features invariantes.
    Profondeur (pas largeur) pour capturer l'invariant statistique du mot."""
    def __init__(self, out_dim=D_MODEL, n_mels=128):
        super().__init__()
        self.n_fft = 256
        self.n_mels = n_mels
        # banc de filtres Mel (fixe)
        fb = torch.zeros(n_mels, self.n_fft // 2 + 1)
        for m in range(n_mels):
            center = (m + 1) * (self.n_fft // 2 + 1 - 1) / (n_mels + 1)
            for f in range(self.n_fft // 2 + 1):
                d = abs(f - center) / max(1.0, (self.n_fft // 2) / (n_mels + 1))
                fb[m, f] = max(0.0, 1.0 - d)
        self.register_buffer("mel_fb", fb / (fb.sum(dim=1, keepdim=True) + 1e-8))
        self.register_buffer("window", torch.hann_window(self.n_fft))
        # 4 conv layers (PROFONDEUR) → features invariantes
        self.convs = nn.Sequential(
            nn.Conv1d(n_mels, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64), nn.Conv1d(64, 32, 3, padding=1), nn.ReLU())
        self.proj = nn.Linear(32, out_dim)

    def forward(self, wav):
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.n_fft // 2,
                          win_length=self.n_fft, window=self.window,
                          return_complex=True, center=False)
        mel = torch.matmul(self.mel_fb, spec.abs() ** 2)
        mel = torch.log1p(mel)           # (B, n_mels, frames)
        h = self.convs(mel)              # (B, 32, frames) — 4 layers profond
        pooled = h.mean(dim=-1)         # (B, 32) — invariant global
        return self.proj(pooled)        # (B, out_dim)


class CrossModalDeep(nn.Module):
    """Cross-modal avec encodeur audio PROFOND. UN SpectralCoreBlock partagé."""
    def __init__(self, n_concepts):
        super().__init__()
        self.audio_enc = DeepAudioEncoder(out_dim=D_MODEL, n_mels=128)
        self.text_proj = nn.Linear(PART, D_MODEL)
        self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
        self.head = nn.Linear(D_MODEL, PART)

    def forward_view(self, feat, proj):
        return self.head(self.core(proj(feat).unsqueeze(1)).squeeze(1))

    def forward_audio(self, wav):
        return self.head(self.core(self.audio_enc(wav).unsqueeze(1)).squeeze(1))


def text_feat(word):
    v = np.zeros(PART, dtype=np.float32)
    for c in word.lower(): v[(ord(c) * 167) % PART] += 1.0
    return v

def phon_feat(word):
    w = word.lower(); vw = sum(1 for c in w if c in "aeiou"); cs = len(w) - vw
    pat = "".join("v" if c in "aeiou" else "c" for c in w)[:8]
    v = np.zeros(PART, dtype=np.float32)
    for c in pat: v[(ord(c) * 167) % PART] += 1.0
    v[(vw * 7) % PART] += 1.0; v[(cs * 11 + PART // 2) % PART] += 1.0
    return v

def load_wav(p):
    y, sr = sf.read(p); y = y.astype(np.float32)
    if y.ndim > 1: y = y.mean(1)
    if len(y) < T: y = np.pad(y, (0, T - len(y)))
    else: y = y[:T]
    return torch.tensor(y)


def train():
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    print(f"[deep encoder + cross-modal] {NW} mots", flush=True)

    # données
    audio_by_word = {}
    for wi, w in enumerate(words):
        wavs = [load_wav(p) for p in glob.glob(os.path.join(SC, w, "*.wav"))[:100]]
        audio_by_word[wi] = torch.stack(wavs).to(device)
    text_all = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all = torch.tensor([phon_feat(w) for w in words]).to(device)

    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW <= PART else "random", seed=0)
    cv.freeze(); canon = cv._matrix().to(device)

    model = CrossModalDeep(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # splits
    audio_tr, audio_te = {}, {}
    for wi in range(NW):
        n = len(audio_by_word[wi]); p = torch.randperm(n); n_te = max(1, n // 5)
        audio_te[wi] = p[:n_te]; audio_tr[wi] = p[n_te:]

    print(f"  encodeur: 4 conv layers (profond) + SpectralCoreBlock\n", flush=True)
    print(f"[GROK cross-modal + encodeur profond — 20000 steps]", flush=True)
    t0 = time.time()
    for step in range(30000):
        wi_batch = torch.randint(0, NW, (32,))
        tgt = canon[wi_batch]
        # text + phonetic views
        out_t = model.forward_view(text_all[wi_batch], model.text_proj)
        out_p = model.forward_view(phon_all[wi_batch], model.phon_proj)
        # audio view (deep encoder)
        wavs = torch.stack([audio_by_word[wi.item()][audio_tr[wi.item()][torch.randint(0, len(audio_tr[wi.item()]), (1,)).item()]]
                            for wi in wi_batch])
        out_a = model.forward_audio(wavs)
        # 1-cos loss (crown-jewel)
        loss = ((1 - F.cosine_similarity(out_t, tgt).clamp(-1, 1)).mean() +
                (1 - F.cosine_similarity(out_p, tgt).clamp(-1, 1)).mean() +
                (1 - F.cosine_similarity(out_a, tgt).clamp(-1, 1)).mean())
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 5000 == 0:
            model.eval()
            with torch.no_grad():
                ok = 0; tot = 0
                for wi in range(NW):
                    for j in audio_te[wi][:2]:
                        wav = audio_by_word[wi][j:j+1]
                        pred = (model.forward_audio(wav) @ canon.t()).argmax(1).item()
                        ok += (pred == wi); tot += 1
                # train acc aussi
                ok_tr = 0; tot_tr = 0
                for wi in range(NW):
                    for j in audio_tr[wi][:1]:
                        wav = audio_by_word[wi][j:j+1]
                        pred = (model.forward_audio(wav) @ canon.t()).argmax(1).item()
                        ok_tr += (pred == wi); tot_tr += 1
            print(f"  step {step:>5} loss={loss.item():.4f} | train={ok_tr}/{tot_tr} "
                  f"test={ok}/{tot} ({ok/max(tot,1)*100:.1f}%) "
                  f"t={time.time()-t0:.0f}s", flush=True)
            model.train()

    # final eval
    model.eval()
    with torch.no_grad():
        ok = sum(1 for wi in range(NW) for j in audio_te[wi]
                 if (model.forward_audio(audio_by_word[wi][j:j+1]) @ canon.t()).argmax(1).item() == wi)
        tot = sum(len(audio_te[wi]) for wi in range(NW))
    print(f"\n{'='*60}")
    print(f"AUDIO CROSS-MODAL + ENCODEUR PROFOND (4 conv)")
    print(f"{'='*60}")
    print(f"  TEST (OOD): {ok}/{tot} = {ok/max(tot,1)*100:.1f}% (hasard {100/NW:.1f}%)")
    print(f"  SOTA SpeechCommands: ~96%")
    print(f"  Gap: {ok/max(tot,1)*100 - 96:+.1f}pt")
    print(f"  temps: {time.time()-t0:.0f}s, 20000 steps")
    print(f"  encodeur: Mel(64) → Conv1d(128) → Conv1d(128) → Conv1d(64) → Conv1d(32) → proj")
    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/deep_encoder_trained.pt"
    torch.save({"model_state": model.state_dict(), "test_acc": ok/max(tot,1),
                "words": words, "method": "deep 4-conv encoder + cross-modal + 1-cos"}, ckpt)
    print(f"  [SAUVÉ] {ckpt}")


if __name__ == "__main__":
    train()
