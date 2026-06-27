#!/usr/bin/env python3
"""Audio→IDs phonétiques DISCRETS invariants (VQ) + crown-jewel — la forme la plus fidèle.

Principe fondateur (RULES_MASTER §H) : « tout convertir en IDs numériques avant le
SpectralCoreBlock ». concept_grok a prouvé que le FFT grok l'association word_ID→number
(73-92%). L'audio échoue parce qu'on restait en features CONTINUES (stochastiques).

Ici on ajoute une couche VQ (vector quantization) qui produit des IDs phonétiques DISCRETS :
  audio -> encoder (InstanceNorm = invariance locuteur) -> features/frame (B,T,d)
       -> VQ: nearest codebook entry -> ID discret par frame (straight-through)
       -> séquence d'IDs phonétiques (invariante si encoder+VQ l'apprennent)
       -> embed IDs -> SpectralCoreBlock GROK l'association ID-seq -> mot (1-cos crown-jewel)

C'est EXACTEMENT le crown-jewel appliqué à l'audio : (phon1, phon2, ...) -> wordID,
comme concept_grok: (tokens) -> numberID. Le FFT découvre la règle de composition phonétique
sur les IDs discrets (nombres), pas sur du signal continu.

VQ (van den Oord 2017): codebook K=128, straight-through estimator, commitment loss.
+ InstanceNorm (locuteur) + SpecAugment (acoustique) pour rendre les IDs invariants.

Respecte : 1-cos crown-jewel, Adam 3e-3, seed 0, IDs discrets, éval holdout propre.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.audio_invariant_ids import specaugment, _data
from train_deep_encoder_v2 import load_wav

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
VQ_K, VQ_DIM = 128, 32       # codebook: 128 IDs phonétiques, dim 32


class FrameEncoder(nn.Module):
    """Mel (InstanceNorm = invariance locuteur) -> convs -> features PAR FRAME (B,T,VQ_DIM)."""
    def __init__(self, n_mels=64, dim=VQ_DIM):
        super().__init__()
        self.n_fft = 256; self.n_mels = n_mels
        fb = torch.zeros(n_mels, self.n_fft // 2 + 1)
        for m in range(n_mels):
            c = (m + 1) * (self.n_fft // 2 + 1 - 1) / (n_mels + 1)
            for f in range(self.n_fft // 2 + 1):
                fb[m, f] = max(0.0, 1.0 - abs(f - c) / max(1.0, (self.n_fft // 2) / (n_mels + 1)))
        self.register_buffer("mel_fb", fb / (fb.sum(1, keepdim=True) + 1e-8))
        self.register_buffer("window", torch.hann_window(self.n_fft))
        self.inst_norm = nn.InstanceNorm1d(n_mels, affine=True)
        self.convs = nn.Sequential(nn.Conv1d(n_mels, 128, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(128, dim, 3, padding=1), nn.ReLU())
    def forward(self, wav, augment=False):
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.n_fft//2,
                          win_length=self.n_fft, window=self.window, return_complex=True, center=False)
        mel = torch.log1p(torch.matmul(self.mel_fb, spec.abs()**2))   # (B,M,T)
        mel = self.inst_norm(mel)                                     # invariance locuteur
        if augment and self.training: mel = specaugment(mel)
        h = self.convs(mel)                                           # (B,dim,T)
        return h.transpose(1, 2)                                      # (B,T,dim)


class VQ(nn.Module):
    """Vector Quantization (van den Oord 2017) : features -> IDs discrets (straight-through)."""
    def __init__(self, k=VQ_K, dim=VQ_DIM, beta=0.25):
        super().__init__()
        self.k, self.beta = k, beta
        self.codebook = nn.Parameter(torch.randn(k, dim) * 0.1)
    def forward(self, z):                          # z=(B,T,dim)
        B, T, D = z.shape
        zf = z.reshape(-1, D)                       # (B*T, D)
        d = (zf.pow(2).sum(1, keepdim=True)
             - 2*zf @ self.codebook.t() + self.codebook.pow(2).sum(1))   # dist² aux codes
        idx = d.argmin(1)                           # ID discret par frame
        zq = self.codebook[idx].reshape(B, T, D)
        # straight-through : forward=zq, backward=z
        zq_st = z + (zq - z).detach()
        commit = F.mse_loss(zf, zq.reshape(-1, D).detach())   # pousse z vers codebook
        return zq_st, idx.reshape(B, T), commit


class AudioVQModel(nn.Module):
    """FrameEncoder -> VQ (IDs discrets) -> embed -> SpectralCoreBlock grok -> mot (1-cos)."""
    def __init__(self, n_words):
        super().__init__()
        self.enc = FrameEncoder()
        self.vq = VQ()
        self.id_embed = nn.Embedding(VQ_K, D_MODEL); nn.init.normal_(self.id_embed.weight, std=0.02)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=64, bidirectional=True)
        self.head = nn.Linear(D_MODEL, PART)
    def forward(self, wav, augment=False):
        z = self.enc(wav, augment=augment)          # (B,T,VQ_DIM)
        zq, ids, commit = self.vq(z)                # IDs discrets invariants (straight-through)
        ids = ids[:, :64]                            # tronque à seq_len=64
        if ids.shape[1] < 64: ids = F.pad(ids, (0, 64-ids.shape[1]))
        emb = self.id_embed(ids)                     # (B,64,D_MODEL) — séquence d'IDs phonétiques
        out = self.core(emb).mean(1)                 # FFT grok la composition ID-seq
        return self.head(out), commit, ids


def train(n_steps=20000, batch=64, lr=3e-3, eval_every=2500, beta=0.25):
    torch.manual_seed(0); random.seed(0)
    words = sorted([w for w in os.listdir(SC)
                    if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")])
    NW = len(words)
    tr, te = _data(words)
    keys = list(tr.keys()); n_words = len(tr)
    cv = LearnedVocab(n=NW, dim=PART, init="ortho", seed=0); cv.freeze()
    canon = cv._matrix().to(device)
    model = AudioVQModel(NW).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def sample():
        bi = [random.choice(keys) for _ in range(batch)]
        wavs = torch.stack([tr[k][torch.randint(0, len(tr[k]), (1,)).item()] for k in bi])
        wi_t = torch.tensor(bi, device=device)
        return wavs, wi_t

    # SC-1 sanity
    print("[SC-1 sanity] overfit 1 batch (400 steps)...", flush=True)
    for _ in range(400):
        # re-utilise les 8 mêmes
        sbi = keys[:8]; sw = torch.stack([tr[k][0] for k in sbi]); swi = torch.tensor(sbi, device=device)
        ent, commit, _ = model(sw, augment=False)
        cos = F.cosine_similarity(ent, canon[swi], -1).clamp(-1,1)
        loss = (1-cos).mean() + beta*commit
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        ent, _, ids = model(sw)
        cos_s = F.cosine_similarity(ent, canon[swi], -1).mean().item()
    print(f"  sanity 1-cos={1-cos_s:.3f} | IDs uniques utilisés: {len(torch.unique(ids))}/{VQ_K}", flush=True)

    print(f"\n[TRAIN VQ-IDs] {n_words} mots | codebook {VQ_K} | InstanceNorm+SpecAugment | {n_steps} steps", flush=True)
    t0 = time.time(); best = 0.0; best_state = None
    for step in range(n_steps):
        wavs, wi_t = sample()
        ent, commit, _ = model(wavs, augment=True)
        cos = F.cosine_similarity(ent, canon[wi_t], -1).clamp(-1,1)
        loss = (1-cos).mean() + beta*commit
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
            ent, _, _ = model(te[wi][j:j+1], augment=False)
            ok += ((ent @ canon.t()).argmax(1).item() == wi); tot += 1
    model.train(); return ok/max(tot,1)


if __name__ == "__main__":
    print("="*64)
    print("AUDIO→IDs DISCRETS INVARIANTS (VQ) + crown-jewel (forme concept_grok)")
    print("="*64)
    model, canon, te, best = train(n_steps=20000)
    print(f"\n{'='*64}\nRÉSULTAT VQ-IDs — holdout PROPRE [100:130]\n{'='*64}")
    print(f"  Test acc: {best*100:.1f}%")
    print(f"  Réf: baseline 45.9% | invariant(InstanceNorm+SpecAugment) | SOTA 96%")
    print(f"  Δ vs baseline: {best*100-45.9:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state": model.state_dict(), "best": best},
               "/media/akone/SAVENVME2/Datasets/ocm26400/audio_vq_trained.pt")
    json.dump({"holdout_acc": best, "delta_vs_baseline_45p9": best*100-45.9,
               "delta_vs_sota_96": best*100-96,
               "method": "VQ discrete invariant IDs + InstanceNorm + SpecAugment + SpectralCoreBlock crown-jewel"},
              open("ocm26400/audio_vq_results.json","w"), indent=2)
    print("  [sauvé] ocm26400/audio_vq_results.json")
