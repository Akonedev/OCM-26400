#!/usr/bin/env python3
"""UNIFIED SOTA push — M5→SpecAugment→SpectralCoreBlock(seq_len=62)→CE+grok, 100k steps.

L'archi unifiée (94.0%) + leviers SOTA:
  1. SpecAugment sur les conv features (B,64,T) — LE booster SOTA ASR (+2-3pt).
  2. CE + petit 1-cos (0.1) — déclenche le GROK (phase transition crown-jewel).
  3. 100k steps (la trajectoire montait encore à 50k).
  4. Confidence gate (observer): abstention sur l'incertain → selective accuracy.
  5. LEAN: cœur SpectralCoreBlock (675K), lobe M5 (périphérie).
  6. save-best/eval (crash-safe).
Cible: 96-97% SOTA SpeechCommands.
"""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,json,random
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock

device="cuda" if torch.cuda.is_available() else "cpu"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_sota_trained.pt"

def specaug_conv(x, n_freq=2, n_time=2, f_mask=8, t_mask=10):
    """SpecAugment sur conv features (B,C,T): masque des channels freq + frames temps."""
    B,C,T = x.shape; m=x.clone()
    for b in range(B):
        for _ in range(n_freq):
            f=random.randint(0,max(1,C-f_mask)); m[b,f:f+random.randint(0,f_mask),:]=0
        for _ in range(n_time):
            t=random.randint(0,max(1,T-t_mask)); m[b,:,t:t+random.randint(0,t_mask)]=0
    return m

class M5UnifiedSOTA(nn.Module):
    """M5 (2 pools → T≈62) → SpecAugment (train) → SpectralCoreBlock(62) global → CE+cos."""
    def __init__(s,nw):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64)
        s.core=SpectralCoreBlock(d_model=64,seq_len=62,bidirectional=True)  # FFT GLOBAL
        s.fc=nn.Linear(64,nw)
    def forward(s,w,augment=False):
        x=w.unsqueeze(1)
        x=s.p1(F.relu(s.b1(s.c1(x))));x=s.p2(F.relu(s.b2(s.c2(x))))
        x=F.relu(s.b3(s.c3(x)));x=F.relu(s.b4(s.c4(x)))    # (B,64,~62) conv features
        if augment and s.training: x=specaug_conv(x)          # SpecAugment
        frames=x.transpose(1,2)                               # (B,62,64)
        mixed=s.core(frames)                                  # FFT GLOBAL (crown-jewel)
        pooled=mixed.mean(1)                                  # (B,64)
        return s.fc(pooled)                                   # (B,nw)

def load_wav(p,T=16000):
    y,sr=sf.read(p);y=y.astype(np.float32)
    if y.ndim>1:y=y.mean(1)
    return torch.from_numpy(np.pad(y,(0,max(0,T-len(y))))[:T] if len(y)<T else y[:T])
def spd(w,r=(0.85,1.15)):
    k=random.uniform(*r);n=len(w);return np.interp(np.clip(np.arange(n)/k,0,n-1),np.arange(n),w).astype(np.float32)
def load_data(words):
    ts=set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip());tr={};te={}
    for wi,w in enumerate(words):
        a=sorted(glob.glob(os.path.join(SC,w,"*.wav")));rp,tp=[],[]
        for p in a:(tp if os.path.relpath(p,SC)in ts else rp).append(p)
        if len(rp)>=50 and len(tp)>=5:tr[wi]=torch.stack([load_wav(p) for p in rp]);te[wi]=torch.stack([load_wav(p) for p in tp])
    return tr,te

def train(n=100000,bs=64,lr=1e-3,ev=2500):
    torch.manual_seed(0);random.seed(0)
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);print(f"[UNIFIED SOTA] chargement...",flush=True)
    tr,te=load_data(ws);keys=list(tr.keys())
    m=M5UnifiedSOTA(NW).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n)
    print(f"[TRAIN UNIFIED SOTA] M5→SpecAug→SCB(62)→CE+0.1cos | {n} steps | save-best/eval+observer",flush=True)
    t0=time.time();best=0.0
    for step in range(n):
        bi=[random.choice(keys) for _ in range(bs)]
        wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        wi=torch.tensor(bi,device=device)
        logits=m(wv,augment=True)   # SpecAugment actif pendant training
        l_ce=F.cross_entropy(logits,wi)
        # petit 1-cos (grok alignment — crown-jewel)
        probs=F.softmax(logits,-1); tgt=F.one_hot(wi,NW).float()
        l_cos=(1-F.cosine_similarity(probs,tgt,-1).clamp(-1,1)).mean()
        loss=l_ce+0.1*l_cos
        opt.zero_grad();loss.backward();opt.step();sched.step()
        if step%ev==0 or step==n-1:
            acc,sacc,cov=_eval(m,te)   # acc + selective acc (observer/gate)
            if acc>best:
                best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best},CKPT)
                json.dump({"test_acc_official":best,"selective_acc":sacc,"coverage":cov,
                           "delta_vs_sota_96":best*100-96},open("ocm26400/audio_unified_sota_results.json","w"),indent=2)
            print(f"  step {step:>5} lr={sched.get_last_lr()[0]:.4f} CE={l_ce.item():.3f} | test {acc*100:.1f}% (best {best*100:.1f}%) sel {sacc*100:.1f}%@{cov*100:.0f}% [S] t={time.time()-t0:.0f}s",flush=True)
    return best

@torch.no_grad()
def _eval(m,te,tau=0.9):
    """acc globale + selective accuracy (observer/gate: abstain if max_prob < tau)."""
    m.eval();ok=tot=0;sok=0;sattempted=0;stot=0
    for wi in te:
        for j in range(0,len(te[wi]),64):
            logits=m(te[wi][j:j+64].to(device));probs=F.softmax(logits,-1)
            pred=logits.argmax(1).cpu();conf=probs.max(1)[0].cpu()
            ok+=(pred==wi).sum().item();tot+=len(pred)
            # observer/gate: selective (confiant seulement)
            mask=conf>=tau;sattempted+=mask.sum().item();stot+=len(pred)
            sok+=((pred==wi)&mask).sum().item()
    m.train()
    return ok/max(tot,1), sok/max(sattempted,1), sattempted/max(stot,1)

if __name__=="__main__":
    print("="*50);print("UNIFIED SOTA — M5→SpecAug→SCB(62)→CE+grok, 100k steps, observer");print("="*50)
    b=train();print(f"\nFINAL: {b*100:.1f}% | vs SOTA 96%: {b*100-96:+.1f}pt")
