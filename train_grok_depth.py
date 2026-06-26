#!/usr/bin/env python3
"""GROK DEPTH (pas encoder depth) — itérer le SpectralCoreBlock profondément.

PRINCIPE : profondeur > params. L4: récurrence ⊥ params ⊥ longueur.
Le crown-jewel marche à 100% grâce à la PROFONDEUR D'ITÉRATION du Block,
pas grâce à un gros encodeur. Même principe pour l'audio :

  encodeur SIMPLE (shallow, params minimes)
  → itérer le SpectralCoreBlock N FOIS (depth = 8, 16, 32, 64...)
  → chaque itération = raisonnement plus profond sur les patterns spectraux
  → à sufficient depth, le FFT GROK l'invariant phonétique

C'est L3 : depth_max = 1/(1-per_step_acc). Plus de depth → plus de compréhension.
C'est L4 : récurrence découple profondeur de params (params FIXES).
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


class GrokDepthModel(nn.Module):
    """Encodeur SIMPLE + SpectralCoreBlock itéré N FOIS (grok depth).
    Params FIXES. La profondeur (itération) fait le grokking."""
    def __init__(self, grok_depth=16, d_model=D_MODEL):
        super().__init__()
        # encodeur SIMPLE (shallow — juste Mel + 1 proj)
        from ocm26400.multimodal_encoders import AudioEncoder
        self.audio_enc = AudioEncoder(out_dim=d_model)  # shallow, params minimes
        # UN SpectralCoreBlock (itéré N fois, pas N blocks)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=1)
        self.head = nn.Linear(d_model, PART)
        self.grok_depth = grok_depth  # combien de fois on itère le Block

    def forward_audio(self, wav):
        """Audio → encodeur shallow → itérer Block N fois → ent (PART).
        Chaque itération = raisonnement plus profond."""
        h = self.audio_enc(wav)  # (B, d_model) features shallow
        for _ in range(self.grok_depth):
            h = self.core(h.unsqueeze(1)).squeeze(1)  # itérer le Block
        return self.head(h)  # (B, PART) après grok depth

    def forward_view(self, feat, proj, grok_depth=None):
        """Texte/phonétique → proj → itérer Block → ent."""
        d = grok_depth if grok_depth is not None else self.grok_depth
        h = proj(feat)
        for _ in range(d):
            h = self.core(h.unsqueeze(1)).squeeze(1)
        return self.head(h)


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
    print(f"[GROK DEPTH] {NW} mots — encoder shallow + Block itéré profond", flush=True)

    # données
    audio_by_word = {}
    for wi, w in enumerate(words):
        wavs = [load_wav(p) for p in glob.glob(os.path.join(SC, w, "*.wav"))[:80]]
        audio_by_word[wi] = torch.stack(wavs).to(device)
    text_all = torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all = torch.tensor([phon_feat(w) for w in words]).to(device)

    cv = LearnedVocab(n=NW, dim=PART, init="ortho" if NW <= PART else "random", seed=0)
    cv.freeze(); canon = cv._matrix().to(device)

    # splits
    audio_tr, audio_te = {}, {}
    for wi in range(NW):
        n = len(audio_by_word[wi]); p = torch.randperm(n); n_te = max(1, n // 5)
        audio_te[wi] = p[:n_te]; audio_tr[wi] = p[n_te:]

    # TESTER PLUSIEURS PROFONDEURS DE GROK
    for DEPTH in [1, 4, 8, 16, 32]:
        print(f"\n{'='*50}")
        print(f"  GROK DEPTH = {DEPTH} (Block itéré {DEPTH} fois, params FIXES)")
        print(f"{'='*50}")

        model = GrokDepthModel(grok_depth=DEPTH).to(device)
        text_proj = nn.Linear(PART, D_MODEL).to(device)
        phon_proj = nn.Linear(PART, D_MODEL).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(text_proj.parameters())
                                + list(phon_proj.parameters()), lr=3e-3)

        t0 = time.time()
        for step in range(10000):
            wi_batch = torch.randint(0, NW, (32,))
            tgt = canon[wi_batch]
            # text + phonetic (shallow depth — ils grokkent vite)
            out_t = model.forward_view(text_all[wi_batch], text_proj, grok_depth=4)
            out_p = model.forward_view(phon_all[wi_batch], phon_proj, grok_depth=4)
            # audio (DEPTH — la profondeur de grok est le levier)
            wavs = torch.stack([audio_by_word[wi.item()][audio_tr[wi.item()][torch.randint(0, len(audio_tr[wi.item()]), (1,)).item()]]
                                for wi in wi_batch])
            out_a = model.forward_audio(wavs)
            # 1-cos loss
            loss = ((1 - F.cosine_similarity(out_t, tgt).clamp(-1, 1)).mean() +
                    (1 - F.cosine_similarity(out_p, tgt).clamp(-1, 1)).mean() +
                    (1 - F.cosine_similarity(out_a, tgt).clamp(-1, 1)).mean())
            opt.zero_grad(); loss.backward(); opt.step()

            if step % 5000 == 0:
                model.eval()
                with torch.no_grad():
                    ok = sum(1 for wi in range(NW) for j in audio_te[wi][:1]
                             if (model.forward_audio(audio_by_word[wi][j:j+1]) @ canon.t()).argmax(1).item() == wi)
                print(f"  step {step} loss={loss.item():.3f} test={ok}/{NW} t={time.time()-t0:.0f}s", flush=True)
                model.train()

        # eval finale
        model.eval()
        with torch.no_grad():
            ok = sum(1 for wi in range(NW) for j in audio_te[wi]
                     if (model.forward_audio(audio_by_word[wi][j:j+1]) @ canon.t()).argmax(1).item() == wi)
            tot = sum(len(audio_te[wi]) for wi in range(NW))
        acc = ok / max(tot, 1)
        print(f"  DEPTH={DEPTH}: TEST = {ok}/{tot} = {acc*100:.1f}% (hasard {100/NW:.1f}%) t={time.time()-t0:.0f}s")

    print(f"\n{'='*60}")
    print(f"Si la accuracy AUGMENTE avec DEPTH → le grok depth marche !")
    print(f"Si elle PLAFONNE → le SpectralCoreBlock (params fixes) a sa limite.")
    print(f"L3: depth_max = 1/(1-per_step) → plus de depth = plus de compréhension")


if __name__ == "__main__":
    print("="*60)
    print("GROK DEPTH — itérer le Block profondément (params FIXES)")
    print("L4: récurrence ⊥ params. Profondeur = raisonnement, pas masse.")
    print("="*60)
    train()
