#!/usr/bin/env python3
"""Approche CONTRASTIVE pour l'audio — grokker l'invariant perceptuel.

PRINCIPE : compréhension > mémoire. Pour l'audio, "compréhension" = comprendre
l'INVARIANT du mot (ce qui fait que "cat" = "cat" malgré le locuteur, le bruit,
la vitesse). C'est un invariant STATISTIQUE, pas algébrique.

Méthode : SupCon (Supervised Contrastive Loss, Khosla 2020).
- POSITIFS : autres utterances du MÊME mot → rapprocher (le mot-invariant)
- NÉGATIFS : utterances de mots DIFFÉRENTS → éloigner
- Le SpectralCoreBlock apprend à mapper même-mot → embeddings proches
- = compréhension de l'invariant phonétique du mot (pas mémorisation d'instances)

Après grokking de l'invariant : probe linéaire → classifie depuis la compréhension.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time
import soundfile as sf
from ocm26400.multimodal_encoders import AudioEncoder
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def supcon_loss(emb, labels, tau=0.1):
    """Supervised Contrastive Loss : même mot → proche, mot différent → loin.
    Le modèle grokke l'INVARIANT du mot (compréhension perceptuelle)."""
    sim = emb @ emb.T / tau                       # (B, B) similarités
    B = len(labels)
    mask_self = torch.eye(B, device=emb.device).bool()
    sim = sim.masked_fill(mask_self, -1e9)        # exclure self
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~mask_self
    pos_count = pos_mask.float().sum(1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(1) / pos_count
    return loss.mean()


def augment_wav(wav):
    """Augmentation légère : simule la variation de locuteur/environnement.
    Crée des VUES différentes du même mot → le modèle doit être INVARIANT."""
    out = wav.clone()
    # bruit gaussien (simule environnement)
    out = out + torch.randn_like(out) * 0.005
    # shift temporel aléatoire (simule onset différent)
    shift = torch.randint(0, 200, (1,)).item()
    out = torch.roll(out, shift)
    # variation de volume (simule distance micro)
    out = out * (0.8 + 0.4 * torch.rand(1, device=wav.device))
    return out


class ContrastiveAudioModel(nn.Module):
    """AudioEncoder + SpectralCoreBlock → embedding (compréhension du mot).
    Le noyau spectral FFT grok la structure phonétique invariante."""
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.enc = AudioEncoder(out_dim=d_model)
        self.core = SpectralCoreBlock(d_model=d_model, seq_len=1)

    def forward(self, wav):
        feat = self.enc(wav)          # (B, d_model) features spectrales
        emb = self.core(feat.unsqueeze(1)).squeeze(1)  # noyau spectral grok
        return F.normalize(emb, dim=-1)  # L2-norm pour contrastive


def train_contrastive_audio():
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    WORDS = [w for w in os.listdir(SC)
             if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")][:15]
    T = 8000

    def load_wav(p):
        y, sr = sf.read(p); y = y.astype(np.float32)
        if y.ndim > 1: y = y.mean(1)
        if len(y) < T: y = np.pad(y, (0, T - len(y)))
        else: y = y[:T]
        return torch.tensor(y)

    # charger données
    auds, labs = [], []
    for wi, w in enumerate(WORDS):
        for p in glob.glob(os.path.join(SC, w, "*.wav"))[:100]:
            auds.append(load_wav(p)); labs.append(wi)
    wav_t = torch.stack(auds).to(device)
    lab_t = torch.tensor(labs).to(device)
    N = len(auds)
    idx = torch.randperm(N); ntr = int(N * 0.85)
    tr_i, te_i = idx[:ntr], idx[ntr:]
    print(f"[contrastive audio] {N} samples réels, {len(WORDS)} mots", flush=True)

    model = ContrastiveAudioModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # PHASE 1 : grokker l'invariant du mot (SupCon)
    print(f"\n[PHASE 1] SupCon — grok l'invariant phonétique (compréhension)", flush=True)
    t0 = time.time()
    for step in range(3000):
        bi = tr_i[torch.randint(0, len(tr_i), (64,))]
        wav_batch = wav_t[bi]
        lab_batch = lab_t[bi]
        # créer 2 vues augmentées par audio → batch de 128
        view1 = torch.stack([augment_wav(w) for w in wav_batch])
        view2 = torch.stack([augment_wav(w) for w in wav_batch])
        # encoder les deux vues
        emb1 = model(view1)
        emb2 = model(view2)
        # concaténer : les 64 premiers et 64 seconds ont les MÊMES labels
        emb_all = torch.cat([emb1, emb2], dim=0)
        lab_all = torch.cat([lab_batch, lab_batch], dim=0)
        loss = supcon_loss(emb_all, lab_all, tau=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step} supcon_loss={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)

    # PHASE 2 : probe linéaire sur l'invariant grokké
    print(f"\n[PHASE 2] probe linéaire sur invariant grokké (compréhension → classe)", flush=True)
    model.eval()
    with torch.no_grad():
        emb_tr = model(wav_t[tr_i])
        emb_te = model(wav_t[te_i])
    probe = nn.Linear(D_MODEL, len(WORDS)).to(device)
    opt_p = torch.optim.Adam(probe.parameters(), lr=3e-3)
    for step in range(1000):
        bi = torch.randint(0, len(tr_i), (48,))
        logits = probe(emb_tr[bi].detach())
        loss = F.cross_entropy(logits, lab_t[tr_i[bi]])
        opt_p.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(emb_te).argmax(1) == lab_t[te_i]).float().mean().item()
    print(f"\n=== AUDIO CONTRASTIF (compréhension de l'invariant) ===")
    print(f"  classification TEST (OOD, {len(te_i)} samples, {len(WORDS)} mots): {acc*100:.1f}%")
    print(f"  hasard: {100/len(WORDS):.0f}%")
    print(f"  temps: {time.time()-t0:.0f}s")
    print(f"  méthode: SupCon (grok invariant phonétique) + probe (classify depuis compréhension)")
    return acc


if __name__ == "__main__":
    print("="*60)
    print("AUDIO CONTRASTIF — grokker l'invariant perceptuel du mot")
    print("PRINCIPE : compréhension de l'invariant > mémorisation d'instances")
    print("="*60)
    acc = train_contrastive_audio()
    print(f"\nRésultat: {acc*100:.1f}%")
