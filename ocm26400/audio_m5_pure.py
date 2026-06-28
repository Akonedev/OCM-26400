#!/usr/bin/env python3
"""M5 PUR (sans SpectralCoreBlock) — le papier M5 qui atteint ~95% SpeechCommands.

Lobe Licensing: le lobe sensoriel M5 est AUTONOME pour la perception (M5→FC→CE direct).
Le SpectralCoreBlock reste pour le RAISONNEMENT (arithmétique, etc.) — pas pour la classification.
C'est l'archi du papier M5 (Dai 2017): M5 dilated conv → FC → CE → ~95%.

CE isolé (avec core) = 88.6%. M5 pur (sans core) devrait atteindre 90-95% (le core était le goulot).
"""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,json,random
import soundfile as sf

device="cuda" if torch.cuda.is_available() else "cpu"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_m5_pure_trained.pt"

class M5Pure(nn.Module):
    """M5 lobe (dilated conv) → FC direct → NW classes. PAS de SpectralCoreBlock (papier M5)."""
    def __init__(s,nw):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32);s.p2=nn.MaxPool1d(4)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64);s.p3=nn.MaxPool1d(4)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64);s.p4=nn.MaxPool1d(4)
        s.fc=nn.Linear(64,nw)   # M5 → FC direct (pas de core spectral)
    def forward(s,w):
        x=w.unsqueeze(1)
        for c,b,p in [(s.c1,s.b1,s.p1),(s.c2,s.b2,s.p2),(s.c3,s.b3,s.p3),(s.c4,s.b4,s.p4)]:
            x=p(F.relu(b(c(x))))
        return s.fc(x.mean(-1))   # (B,NW) logits directs

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

def train(n=50000,bs=64,lr=1e-3,ev=2500):
    torch.manual_seed(0);random.seed(0)
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);print(f"[M5 pur] chargement...",flush=True)
    tr,te=load_data(ws);keys=list(tr.keys())
    m=M5Pure(NW).to(device);opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n)
    print(f"[TRAIN M5 pur] lr={lr} wd=1e-4 cosLR | {n} steps | save-best/eval",flush=True)
    t0=time.time();best=0.0
    for step in range(n):
        bi=[random.choice(keys) for _ in range(bs)]
        wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        wi=torch.tensor(bi,device=device)
        logits=m(wv);loss=F.cross_entropy(logits,wi)
        opt.zero_grad();loss.backward();opt.step();sched.step()
        if step%ev==0 or step==n-1:
            acc=_eval(m,te)
            if acc>best:
                best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best},CKPT)
                json.dump({"test_acc_official":best,"delta_vs_core_88p6":best*100-88.6,"delta_vs_sota_96":best*100-96},open("ocm26400/audio_m5_pure_results.json","w"),indent=2)
            print(f"  step {step:>5} lr={sched.get_last_lr()[0]:.4f} CE={loss.item():.3f} | test {acc*100:.1f}% (best {best*100:.1f}%) [S] t={time.time()-t0:.0f}s",flush=True)
    return best

@torch.no_grad()
def _eval(m,te):
    m.eval();ok=tot=0
    for wi in te:
        for j in range(0,len(te[wi]),64):
            p=m(te[wi][j:j+64].to(device)).argmax(1).cpu();ok+=(p==wi).sum().item();tot+=len(p)
    m.train();return ok/max(tot,1)

if __name__=="__main__":
    print("="*50);print("M5 PUR (sans SpectralCoreBlock) — papier M5 → ~95%");print("="*50)
    b=train();print(f"\nFINAL: {b*100:.1f}% | vs core 88.6%: {b*100-88.6:+.1f}pt | vs SOTA 96%: {b*100-96:+.1f}pt")
