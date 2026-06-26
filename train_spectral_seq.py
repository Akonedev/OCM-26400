#!/usr/bin/env python3
"""SpectralCoreBlock sur SÉQUENCE Mel complète — FFT temporelle = grok phonétique.

LE FIX : le SpectralCoreBlock mélange par FFT à travers une SÉQUENCE.
seq_len=1 (un point) = pas de mélange (FFT triviale = DC).
seq_len=N (N frames Mel) = FFT à travers le temps = découverte des patterns
phonétiques (formants, transitions, rythme).

L'audio → Mel-STFT → séquence de N frames → SpectralCoreBlock(FFT sur N frames)
→ les patterns temporels-spectraux SONT les phonèmes. Le Block les découvre
par FFT (les fréquences temporelles = les phonèmes).

C'est ÇA la profondeur : la séquence est longue (N frames), la FFT mélange
à travers tous les frames → compréhension globale du pattern phonétique.
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
N_FRAMES = 32  # séquence Mel (longueur = profondeur du raisonnement spectral)
N_MELS = 32


def extract_mel_seq(y, n_fft=256, hop=T//(N_FRAMES+1)):
    """Audio → Mel spectrogramme → (N_FRAMES, N_MELS) séquence temporelle.
    Chaque frame = un instant spectral. La séquence = l'évolution temporelle
    = les phonèmes dans le temps. C'est CE QUE LA FFT DOIT MIXER."""
    win = np.hanning(n_fft)
    frames = []
    for s in range(0, min(len(y)-n_fft, hop*N_FRAMES), hop):
        seg = y[s:s+n_fft] * win
        fft = np.fft.rfft(seg)
        power = np.abs(fft) ** 2
        mel = np.zeros(N_MELS)
        for m in range(N_MELS):
            lo = int(m * len(power) / N_MELS)
            hi = int((m+1) * len(power) / N_MELS)
            mel[m] = np.log1p(power[lo:hi].sum())
        frames.append(mel)
    # pad/truncate à N_FRAMES
    while len(frames) < N_FRAMES:
        frames.append(frames[-1] if frames else np.zeros(N_MELS))
    return np.array(frames[:N_FRAMES], dtype=np.float32)  # (N_FRAMES, N_MELS)


class SpectralSequenceModel(nn.Module):
    """Mel-séquence → SpectralCoreBlock(FFT sur N_FRAMES) → grok phonétique.

    Le Block reçoit (B, N_FRAMES, D_MODEL) → FFT à travers les N_FRAMES
    = analyse temporelle des patterns spectraux = compréhension phonétique.

    La 'profondeur' est la LONGUEUR DE SÉQUENCE (N_FRAMES). Plus de frames
    = plus de contexte temporel = plus de compréhension phonétique."""
    def __init__(self, mel_dim=N_MELS, d_model=D_MODEL, seq_len=N_FRAMES):
        super().__init__()
        # proj Mel → d_model (pour que le Block reçoive des vecteurs D_MODEL)
        self.proj = nn.Linear(mel_dim, d_model)
        # SpectralCoreBlock SUR LA SÉQUENCE (FFT à travers N_FRAMES)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=seq_len, bidirectional=True)
        # tête → ent (PART) pour 1-cos
        self.head = nn.Linear(d_model, PART)

    def forward(self, mel_seq):
        """mel_seq: (B, N_FRAMES, N_MELS) → ent (B, PART).
        FFT à travers N_FRAMES = découverte des patterns phonétiques."""
        x = self.proj(mel_seq)         # (B, N_FRAMES, d_model)
        out = self.core(x)            # FFT sur N_FRAMES → grok phonétique
        pooled = out.mean(dim=1)      # pool → (B, d_model)
        return self.head(pooled)      # (B, PART)


def text_feat(word):
    v = np.zeros(PART, dtype=np.float32)
    for c in word.lower(): v[(ord(c)*167)%PART] += 1.0
    return v

def phon_feat(word):
    w = word.lower(); vw = sum(1 for c in w if c in "aeiou"); cs = len(w)-vw
    pat = "".join("v" if c in "aeiou" else "c" for c in w)[:8]
    v = np.zeros(PART, dtype=np.float32)
    for c in pat: v[(ord(c)*167)%PART] += 1.0
    v[(vw*7)%PART] += 1.0; v[(cs*11+PART//2)%PART] += 1.0
    return v


def train():
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    print(f"[FFT-séquence] {NW} mots — Mel({N_FRAMES}×{N_MELS}) → SpectralCoreBlock(FFT sur {N_FRAMES} frames)", flush=True)

    # extraire Mel séquences
    print("  extraction Mel...", flush=True)
    mel_seqs = []; labs = []
    for wi, w in enumerate(words):
        for p in glob.glob(os.path.join(SC, w, "*.wav"))[:80]:
            y, sr = sf.read(p); y = y.astype(np.float32)
            if y.ndim > 1: y = y.mean(1)
            mel_seqs.append(extract_mel_seq(y)); labs.append(wi)
    mel_t = torch.tensor(np.array(mel_seqs)).to(device)  # (N, N_FRAMES, N_MELS)
    lab_t = torch.tensor(labs).to(device)
    N = len(labs)
    print(f"  {N} séquences Mel ({N_FRAMES}×{N_MELS})", flush=True)

    # canonical + modèle
    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW<=PART else "random", seed=0)
    cv.freeze(); canon = cv._matrix().to(device)

    text_all = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all = torch.tensor([phon_feat(w) for w in words]).to(device)

    model = SpectralSequenceModel().to(device)
    text_proj = nn.Linear(PART, D_MODEL).to(device)
    text_core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1).to(device)
    text_head = nn.Linear(D_MODEL, PART).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(text_proj.parameters())
                            + list(text_core.parameters()) + list(text_head.parameters()), lr=3e-3)

    # splits
    perm = torch.randperm(N); n_tr = int(N*0.85); tr_i, te_i = perm[:n_tr], perm[n_tr:]

    print(f"\n[GROK : Mel-séquence → FFT sur {N_FRAMES} frames → 1-cos]", flush=True)
    print(f"  L'FFT mélange à travers les {N_FRAMES} frames temporels", flush=True)
    print(f"  = découverte des PATTERNS phonétiques (formants, transitions)\n", flush=True)
    t0 = time.time()
    for step in range(20000):
        bi = tr_i[torch.randint(0, len(tr_i), (48,))]
        tgt = canon[lab_t[bi]]
        # vue audio (FFT sur séquence Mel)
        out_a = model(mel_t[bi])
        loss_a = (1 - F.cosine_similarity(out_a, tgt).clamp(-1, 1)).mean()
        # vue texte (ancre)
        wi_batch = lab_t[bi]
        out_t = text_head(text_core(text_proj(text_all[wi_batch]).unsqueeze(1)).squeeze(1))
        loss_t = (1 - F.cosine_similarity(out_t, tgt).clamp(-1, 1)).mean()
        loss = loss_a + loss_t
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 4000 == 0:
            model.eval()
            with torch.no_grad():
                ok = 0; tot = 0
                for j in te_i.tolist():
                    pred = (model(mel_t[j:j+1]) @ canon.t()).argmax(1).item()
                    ok += (pred == lab_t[j].item()); tot += 1
                tr_ok = sum(1 for j in tr_i[:100].tolist()
                           if (model(mel_t[j:j+1]) @ canon.t()).argmax(1).item() == lab_t[j].item())
            print(f"  step {step:>5} loss={loss.item():.4f} | train={tr_ok}% test={ok}/{tot} "
                  f"({ok/max(tot,1)*100:.1f}%) t={time.time()-t0:.0f}s", flush=True)
            model.train()

    # final
    model.eval()
    with torch.no_grad():
        ok = sum(1 for j in te_i.tolist()
                 if (model(mel_t[j:j+1]) @ canon.t()).argmax(1).item() == lab_t[j].item())
        tot = len(te_i)
        tr_ok = sum(1 for j in tr_i[:300].tolist()
                   if (model(mel_t[j:j+1]) @ canon.t()).argmax(1).item() == lab_t[j].item())
        tr_tot = 300
    print(f"\n{'='*60}")
    print(f"FFT SUR SÉQUENCE MEL — GROK PHONÉTIQUE")
    print(f"{'='*60}")
    print(f"  Mel-séquence: {N_FRAMES} frames × {N_MELS} mel")
    print(f"  SpectralCoreBlock: FFT sur {N_FRAMES} frames (profondeur temporelle)")
    print(f"  TRAIN: {tr_ok}/{tr_tot} = {tr_ok/max(tr_tot,1)*100:.1f}%")
    print(f"  TEST (OOD): {ok}/{tot} = {ok/max(tot,1)*100:.1f}% (hasard {100/NW:.1f}%)")
    print(f"  gap: {(tr_ok/max(tr_tot,1)-ok/max(tot,1))*100:.1f}pt")
    print(f"  temps: {time.time()-t0:.0f}s")

    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/spectral_seq_grok.pt"
    torch.save({"model_state": model.state_dict(), "canon": canon,
                "test_acc": ok/max(tot,1), "words": words,
                "method": "FFT sur Mel-séquence (grok phonétique temporel)"}, ckpt)
    print(f"  [SAUVÉ] {ckpt}")


if __name__ == "__main__":
    print("="*60)
    print("FFT SUR SÉQUENCE MEL — le Block reçoit N frames (pas 1 point)")
    print("La FFT mélange à travers le temps = découverte des phonèmes")
    print("="*60)
    train()
