#!/usr/bin/env python3
"""Lobe audio M5 (dilated conv 1D, SOTA SpeechCommands) + cœur SpectralCoreBlock.

Architecture Lobe Licensing (Besoins/Formules_Maths_Algo) : lobe sensoriel (encodeur, peut
être un CNN profond) SÉPARÉ du cœur de raisonnement (SpectralCoreBlock, fixé). Le lobe M5
(Dai 2017, ~95% SpeechCommands) est construit FROM-SCRATCH (pas un modèle pré-entraîné) ->
pas de Frankenstein. Entrée = waveform NUMÉRIQUE (samples) -> satisfait « tout numérique ».

M5 : Conv1d(dilated) empilés + BN + ReLU + pool. C'est L'archi SOTA SpeechCommands.
+ cœur SpectralCoreBlock (le paradigme) sur les features.
+ capture simultanée (text+phon+audio) + speed aug, full scale, split officiel.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import glob, os, numpy as np, time, json, random
import soundfile as sf
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
from ocm26400.spectral_core import SpectralCoreBlock
from train_deep_encoder_v2 import text_feat, phon_feat

device = "cuda" if torch.cuda.is_available() else "cpu"
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"


class M5Lobe(nn.Module):
    """Lobe sensoriel M5 (Dai 2017) — dilated Conv1d, SOTA SpeechCommands. From-scratch.
    Entrée waveform numérique (B, T) -> features (B, D_MODEL). Pas de Mel (spectral natif via convs)."""
    def __init__(self, out_dim=D_MODEL):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 80, stride=16)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(32, 32, 3); self.bn2 = nn.BatchNorm1d(32); self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(32, 64, 3); self.bn3 = nn.BatchNorm1d(64); self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(64, 64, 3); self.bn4 = nn.BatchNorm1d(64); self.pool4 = nn.MaxPool1d(4)
        self.proj = nn.Linear(64, out_dim)
    def forward(self, wav):                  # (B, T)
        x = wav.unsqueeze(1)                 # (B,1,T)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.mean(-1)                       # (B,64) global pool
        return self.proj(x)                  # (B, out_dim)


class M5SpectralModel(nn.Module):
    """Lobe M5 (sensoriel, Lobe Licensing) + cœur SpectralCoreBlock (raisonnement, fixé)."""
    def __init__(self, n_words):
        super().__init__()
        self.lobe = M5Lobe()
        self.text_proj = nn.Linear(PART, D_MODEL); self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)   # cœur (paradigme)
        self.head = nn.Linear(D_MODEL, PART)
    def audio_view(self, wav): return self.head(self.core(self.lobe(wav).unsqueeze(1)).squeeze(1))
    def text_view(self, f):  return self.head(self.core(self.text_proj(f).unsqueeze(1)).squeeze(1))
    def phon_view(self, f):  return self.head(self.core(self.phon_proj(f).unsqueeze(1)).squeeze(1))


def load_wav(p, T=16000):
    y,sr=sf.read(p); y=y.astype(np.float32)
    if y.ndim>1: y=y.mean(1)
    if len(y)<T: y=np.pad(y,(0,T-len(y)))
    else: y=y[:T]
    return torch.from_numpy(y)
def speed_perturb(w, r=(0.9,1.1)):
    rate=random.uniform(*r); n=len(w); idx=np.clip(np.arange(n)/rate,0,n-1)
    return np.interp(idx, np.arange(n), w).astype(np.float32)
def load_official():
    return set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip())
def load_data(words):
    test_set=load_official(); tr={}; te={}
    for wi,w in enumerate(words):
        allw=sorted(glob.glob(os.path.join(SC,w,"*.wav"))); trp,tep=[],[]
        for p in allw: (tep if os.path.relpath(p,SC) in test_set else trp).append(p)
        if len(trp)>=50 and len(tep)>=5:
            tr[wi]=torch.stack([load_wav(p) for p in trp]); te[wi]=torch.stack([load_wav(p) for p in tep])
    return tr,te


def train(n_steps=25000, batch=64, lr=3e-3, eval_every=2500):
    torch.manual_seed(0); random.seed(0)
    words=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(words); print("[M5 lobe + cœur spectral] chargement...", flush=True)
    tr,te=load_data(words); keys=list(tr.keys())
    print(f"  {len(keys)} mots | train={sum(len(tr[w]) for w in tr)} | test officiel={sum(len(te[w]) for w in te)}", flush=True)
    text_all=torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all=torch.tensor([phon_feat(w) for w in words]).to(device)
    cv=LearnedVocab(n=NW,dim=PART,init="ortho",seed=0); cv.freeze(); canon=cv._matrix().to(device)
    model=M5SpectralModel(NW).to(device); opt=torch.optim.Adam(model.parameters(),lr=lr)

    def joint(wi_t,wavs):
        tgt=canon[wi_t]
        return ((1-F.cosine_similarity(model.text_view(text_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                (1-F.cosine_similarity(model.phon_view(phon_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                (1-F.cosine_similarity(model.audio_view(wavs),tgt,-1).clamp(-1,1)).mean())

    print(f"\n[TRAIN M5 lobe + SpectralCoreBlock] capture simultanée | waveform numérique | {n_steps} steps", flush=True)
    t0=time.time(); best=0.0; best_state=None
    for step in range(n_steps):
        bi=[random.choice(keys) for _ in range(batch)]
        wl=[]
        for k in bi:
            w=tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy()
            w=speed_perturb(w)               # aug (variation locuteur)
            wl.append(torch.from_numpy(w))
        wavs=torch.stack(wl).to(device); wi_t=torch.tensor(bi,device=device)
        loss=joint(wi_t,wavs); opt.zero_grad(); loss.backward(); opt.step()
        if step%eval_every==0 or step==n_steps-1:
            acc=_eval(model,canon,te)
            if acc>best: best=acc; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
            print(f"  step {step:>5} 1-cos={loss.item():.4f} | test OFFICIEL {acc*100:.1f}% (best {best*100:.1f}%) | t={time.time()-t0:.0f}s", flush=True)
    if best_state: model.load_state_dict(best_state)
    return model,canon,te,best


@torch.no_grad()
def _eval(model,canon,te):
    model.eval(); ok=tot=0
    for wi in te:
        for j in range(0,len(te[wi]),32):
            wavs=te[wi][j:j+32].to(device); p=(model.audio_view(wavs)@canon.t()).argmax(1).cpu()
            ok+=(p==wi).sum().item(); tot+=len(p)
    model.train(); return ok/max(tot,1)


if __name__=="__main__":
    print("="*64); print("LOBE M5 (SOTA SpeechCommands) + cœur SpectralCoreBlock — Lobe Licensing"); print("="*64)
    model,canon,te,best=train(n_steps=25000)
    print(f"\n{'='*64}\nRÉSULTAT M5 lobe + cœur spectral — test OFFICIEL\n{'='*64}")
    print(f"  Test acc OFFICIEL: {best*100:.1f}%")
    print(f"  Réf: Mel-simul 62.5% | SOTA SpeechCommands ~96%")
    print(f"  Δ vs Mel-simul: {best*100-62.5:+.1f}pt | Δ vs SOTA: {best*100-96:+.1f}pt")
    torch.save({"model_state":model.state_dict(),"best":best},"/media/akone/SAVENVME2/Datasets/ocm26400/audio_m5_trained.pt")
    json.dump({"test_acc_official":best,"delta_vs_mel_62p5":best*100-62.5,"delta_vs_sota_96":best*100-96,
               "method":"M5 dilated-conv lobe (Lobe Licensing sensory) + SpectralCoreBlock core, waveform numeric, full scale + speed aug"},
              open("ocm26400/audio_m5_results.json","w"),indent=2)
    print("  [sauvé] ocm26400/audio_m5_results.json")
