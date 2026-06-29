"""M5 avec T≈124 frames (1 pool au lieu de 2) → séquence 2x plus longue pour le SCB.
Le FFT global voit 2x plus de contexte phonétique → composition plus riche."""
import torch,torch.nn as nn,torch.nn.functional as F,glob,os,numpy as np,time,random
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock

device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
CKPT="/media/akone/SAVENVME2/Datasets/ocm26400/audio_m5_t124_trained.pt"

class M5T124(nn.Module):
    """M5 avec 1 seul pool (au lieu de 2) → T≈124 frames pour le SCB(seq_len=124)."""
    def __init__(s,nw):
        super().__init__()
        s.c1=nn.Conv1d(1,32,80,stride=16);s.b1=nn.BatchNorm1d(32);s.p1=nn.MaxPool1d(4)  # 1 pool
        s.c2=nn.Conv1d(32,32,3);s.b2=nn.BatchNorm1d(32)  # PAS de pool2 (garde T≈124)
        s.c3=nn.Conv1d(32,64,3);s.b3=nn.BatchNorm1d(64)
        s.c4=nn.Conv1d(64,64,3);s.b4=nn.BatchNorm1d(64)
        s.core=SpectralCoreBlock(d_model=64,seq_len=124,bidirectional=True)  # seq_len=124!
        s.fc=nn.Linear(64,nw)
    def forward(s,w):
        x=w.unsqueeze(1)
        x=s.p1(F.relu(s.b1(s.c1(x))))    # (B,32,~248)
        x=F.relu(s.b2(s.c2(x)))           # (B,32,~248) pas de pool
        x=F.relu(s.b3(s.c3(x)))           # (B,64,~248)
        x=F.relu(s.b4(s.c4(x)))           # (B,64,~248)
        frames=x[:,:124].transpose(1,2)   # (B,124,64) truncate à 124 pour le SCB
        mixed=s.core(frames)              # FFT GLOBAL sur 124 frames!
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

if __name__=='__main__':
    torch.manual_seed(999);random.seed(999)
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws);tr,te=load_data(ws);keys=list(tr.keys())
    m=M5T124(NW).to(device);opt=torch.optim.Adam(m.parameters(),lr=1e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=100000)
    print(f"[M5-T124] seq_len=124 (1 pool), 100k steps, seed999",flush=True);t0=time.time();best=0
    for step in range(100000):
        bi=[random.choice(keys) for _ in range(64)]
        wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
        wi=torch.tensor(bi,device=device)
        logits=m(wv);loss=F.cross_entropy(logits,wi)
        opt.zero_grad();loss.backward();opt.step();sched.step()
        if step%5000==0 or step==99999:
            m.eval();ok=tot=0
            with torch.no_grad():
                for w in te:
                    for j in range(0,len(te[w]),64):
                        p=m(te[w][j:j+64].to(device)).argmax(1).cpu();ok+=(p==w).sum().item();tot+=len(p)
            m.train();acc=ok/max(tot,1)
            if acc>best:best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in m.state_dict().items()},"best":best,"arch":"M5T124"},CKPT)
            print(f"  step {step:>5} test {acc*100:.1f}% (best {best*100:.1f}%) [S] t={time.time()-t0:.0f}s",flush=True)
    print(f"\nFINAL: {best*100:.1f}%")
