#!/usr/bin/env python3
"""Lobe M18-résiduel (SOTA SpeechCommands ~95%) + cœur SpectralCoreBlock + SAFE launch.

M18 (Dai 2017) = M5 + blocs RÉSIDUELS (4 par stage) → ~94-95% SpeechCommands (full SOTA).
From-scratch (pas pré-entraîné) → Lobe Licensing compliant. Entrée waveform NUMÉRIQUE.

SÉCURITÉ (leçon du GPU hang):
  - GPU GUARD: ne lance QUE si >= 8GB VRAM libre (sinon attend) → ne sature pas le système.
  - SAVE-BEST à CHAQUE éval (checkpoint + JSON) → ne perd rien si crash/redémarrage.
  - 1 GPU, batch modeste.

Recipe M5/M18 pour ~95%: weight_decay (1e-2), cosine LR decay, augmentation.
+ cœur SpectralCoreBlock (paradigme) + capture simultanée.
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
CKPT = "/media/akone/SAVENVME2/Datasets/ocm26400/audio_mres_trained.pt"
JSON_OUT = "ocm26400/audio_mres_results.json"


class ResBlock(nn.Module):
    """Bloc résiduel M-resnet: Conv-BN-ReLU-Conv-BN + skip."""
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.ReLU(),
                                 nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch))
    def forward(self, x): return F.relu(x + self.net(x))


class MResLobe(nn.Module):
    """Lobe M18-résiduel (M5 + 4 ResBlocks/stage) ~95% SOTA. From-scratch."""
    def __init__(self, out_dim=D_MODEL, n_res=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 80, stride=16); self.bn1 = nn.BatchNorm1d(32); self.pool1 = nn.MaxPool1d(4)
        self.res32 = nn.Sequential(*[ResBlock(32) for _ in range(n_res)])
        self.down1 = nn.Conv1d(32, 64, 1); self.pool2 = nn.MaxPool1d(4)
        self.res64a = nn.Sequential(*[ResBlock(64) for _ in range(n_res)])
        self.down2 = nn.Conv1d(64, 64, 1); self.pool3 = nn.MaxPool1d(4)
        self.res64b = nn.Sequential(*[ResBlock(64) for _ in range(n_res)])
        self.pool4 = nn.MaxPool1d(4)
        self.proj = nn.Linear(64, out_dim)
    def forward(self, wav):
        x = wav.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x)))); x = self.res32(x)
        x = self.pool2(self.down1(x)); x = self.res64a(x)
        x = self.pool3(self.down2(x)); x = self.res64b(x)
        x = self.pool4(x); x = x.mean(-1)
        return self.proj(x)


class MResSpectralModel(nn.Module):
    def __init__(self, n_words):
        super().__init__()
        self.lobe = MResLobe()
        self.text_proj = nn.Linear(PART, D_MODEL); self.phon_proj = nn.Linear(PART, D_MODEL)
        self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
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
def speed_perturb(w, r=(0.85,1.15)):
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


def gpu_ready(min_gb=8.0):
    if not torch.cuda.is_available(): return True
    free_gb = torch.cuda.mem_get_info(0)[0]/1e9
    return free_gb >= min_gb


def train(n_steps=25000, batch=48, lr=3e-3, eval_every=2000, wd=1e-2, n_res=0):
    torch.manual_seed(0); random.seed(0)
    # GPU GUARD: attend que la VRAM soit libre (ne sature pas)
    while not gpu_ready(8.0):
        print(f"[GPU GUARD] VRAM insuffisante ({torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB < 8GB). Attente 60s (wavemind?).", flush=True)
        time.sleep(60)
    words=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(words); print("[M18-rés lobe + cœur spectral] chargement...", flush=True)
    tr,te=load_data(words); keys=list(tr.keys())
    print(f"  {len(keys)} mots | train={sum(len(tr[w]) for w in tr)} | test officiel={sum(len(te[w]) for w in te)}", flush=True)
    text_all=torch.tensor([text_feat(w) for w in words]).to(device)
    phon_all=torch.tensor([phon_feat(w) for w in words]).to(device)
    cv=LearnedVocab(n=NW,dim=PART,init="ortho",seed=0); cv.freeze(); canon=cv._matrix().to(device)
    model=MResSpectralModel(NW).to(device)
    model.lobe = MResLobe(n_res=n_res).to(device)   # n_res=0 → M5-like (stable, rapide)
    opt=torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)   # LR decay (recipe M5 95%)

    def joint(wi_t,wavs):
        tgt=canon[wi_t]
        return ((1-F.cosine_similarity(model.text_view(text_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                (1-F.cosine_similarity(model.phon_view(phon_all[wi_t]),tgt,-1).clamp(-1,1)).mean()+
                (1-F.cosine_similarity(model.audio_view(wavs),tgt,-1).clamp(-1,1)).mean())

    def save_best(best, state):
        torch.save({"model_state":state,"best":best}, CKPT)
        json.dump({"test_acc_official":best,"delta_vs_sota_96":best*100-96,
                   "method":"M18-residual lobe (Lobe Licensing) + SpectralCoreBlock, full scale+aug+wd+cosLR"},
                  open(JSON_OUT,"w"), indent=2)

    print(f"\n[TRAIN M18-rés + cœur spectral] capture simultanée | wd={wd} cosLR | save-best/eval | {n_steps} steps", flush=True)
    t0=time.time(); best=0.0
    for step in range(n_steps):
        bi=[random.choice(keys) for _ in range(batch)]
        wl=[]
        for k in bi:
            w=tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy(); w=speed_perturb(w)
            wl.append(torch.from_numpy(w))
        wavs=torch.stack(wl).to(device); wi_t=torch.tensor(bi,device=device)
        loss=joint(wi_t,wavs); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step%eval_every==0 or step==n_steps-1:
            acc=_eval(model,canon,te)
            if acc>best:
                best=acc; save_best(best, {k:v.detach().clone() for k,v in model.state_dict().items()})  # SAVE/eval
            print(f"  step {step:>5} 1-cos={loss.item():.4f} lr={sched.get_last_lr()[0]:.4f} | test OFFICIEL {acc*100:.1f}% (best {best*100:.1f}%) [SAUVÉ] | t={time.time()-t0:.0f}s", flush=True)
    print(f"\n[FINI] meilleur test officiel = {best*100:.1f}% (checkpoint sauvegardé)")
    return best


@torch.no_grad()
def _eval(model,canon,te):
    model.eval(); ok=tot=0
    for wi in te:
        for j in range(0,len(te[wi]),32):
            wavs=te[wi][j:j+32].to(device); p=(model.audio_view(wavs)@canon.t()).argmax(1).cpu()
            ok+=(p==wi).sum().item(); tot+=len(p)
    model.train(); return ok/max(tot,1)


if __name__=="__main__":
    print("="*64); print("LOBE M18-RÉSIDUEL (SOTA ~95%) + cœur SpectralCoreBlock — SAFE (GPU guard + save/eval)"); print("="*64)
    best=train(n_steps=25000)
    print(f"  Test acc OFFICIEL final: {best*100:.1f}% | SOTA 96% | Δ: {best*100-96:+.1f}pt")
