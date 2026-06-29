#!/usr/bin/env python3
"""M11 résiduel (M5 + 1 ResBlock) CE loss — diversité d'architecture pour l'ensemble.

M11 = M5 + 1 bloc résiduel → features plus profondes → erreurs différentes du M5 simple.
CE loss (pas 1-cos — le 1-cos déstabilisait le résiduel). 100k steps, official split.
Sauvegarde: audio_m11_trained.pt (pour l'ensemble 4-modèles).
"""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,json,random
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock

device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_m11_trained.pt"

class ResBlock(nn.Module):
    def __init__(s,c):
        super().__init__();s.net=nn.Sequential(nn.Conv1d(c,c,3,padding=1),nn.BatchNorm1d(c),nn.ReLU(),nn.Conv1d(c,c,3,padding=1),nn.BatchNorm1d(c))
    def forward(s,x):return F.relu(x+s.net(x))

class M11Unified(nn.Module):
    """M5 + 1 ResBlock après conv2 → SpectralCoreBlock(seq_len=62) → CE. Plus profond que M5."""
    def __init__(s,nw):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.res=ResBlock(32)  # +1 bloc résiduel (diversité vs M5)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64)
        s.core=SpectralCoreBlock(d_model=64,seq_len=62,bidirectional=True)
        s.fc=nn.Linear(64,nw)
    def forward(s,w):
        x=w.unsqueeze(1)
        x=s.p1(F.relu(s.b1(s.c1(x))));x=s.p2(F.relu(s.b2(s.c2(x))))
        x=s.res(x)  # bloc résiduel (différence vs M5)
        x=F.relu(s.b3(s.c3(x)));x=F.relu(s.b4(s.c4(x)))
        frames=x.transpose(1,2);mixed=s.core(frames)
        return s.fc(mixed.mean(1))

def load_wav(p,T=16000):
    y,sr=sf.read(p);y=y.astype(np.float32)
    if y.ndim>1:y=y.mean(1)
    return torch.from_numpy(np.pad(y,(0,max(0,T-len(y))))[:T] if len(y)<T else y[:T])
def spd(w,r=(0.9,1.1)):
    k=random.uniform(*r);n=len(w);return np.interp(np.clip(np.arange(n)/k,0,n-1),np.arange(n),w).astype(np.float32)
def load_data(words):
    ts=set(l.strip() for l in open(os.path.join(SC,"testing_list.txt")) if l.strip());tr={};te={}
    for wi,w in enumerate(words):
        a=sorted(glob.glob(os.path.join(SC,w,"*.wav")));rp,tp=[],[]
        for p in a:(tp if os.path.relpath(p,SC)in ts else rp).append(p)
        if len(rp)>=50 and len(tp)>=5:tr[wi]=torch.stack([load_wav(p) for p in rp]);te[wi]=torch.stack([load_wav(p) for p in tp])
    return tr,te

def train(n=100000,bs=64,lr=1e-3,ev=5000):
    torch.manual_seed(42);random.seed(42)  # seed différent pour diversité
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);print(f"[M11 résiduel CE] chargement...",flush=True)
    tr,te=load_data(ws);keys=list(tr.keys())
    m=M11Unified(NW).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n)
    print(f"[TRAIN M11] M5+ResBlock→SCB(62)→CE | seed42 | {n} steps",flush=True)
    t0=time.time();best=0.0
    for step in range(n):
        bi=[random.choice(keys) for _ in range(bs)]
        wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        wi=torch.tensor(bi,device=device)
        logits=m(wv);loss=F.cross_entropy(logits,wi)
        opt.zero_grad();loss.backward();opt.step();sched.step()
        if step%ev==0 or step==n-1:
            m.eval();ok=tot=0
            with torch.no_grad():
                for w in te:
                    for j in range(0,len(te[w]),64):
                        p=m(te[w][j:j+64].to(device)).argmax(1).cpu();ok+=(p==w).sum().item();tot+=len(p)
            m.train();acc=ok/max(tot,1)
            if acc>best:
                best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best,"arch":"M11"},CKPT)
            print(f"  step {step:>5} lr={sched.get_last_lr()[0]:.4f} CE={loss.item():.3f} | test {acc*100:.1f}% (best {best*100:.1f}%) [S] t={time.time()-t0:.0f}s",flush=True)
    return best

if __name__=="__main__":
    print("="*50);print("M11 RÉSIDUEL (M5+ResBlock→SCB→CE) — diversité ensemble");print("="*50)
    b=train();print(f"\nFINAL: {b*100:.1f}%")
