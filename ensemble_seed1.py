import sys, os
seed = int(sys.argv[1]) if len(sys.argv)>1 else 1
os.environ["ENSEMBLE_SEED"] = str(seed)
# patch le seed dans le module unified avant import
import ocm26400.audio_unified_m5scb as m
original_train = m.train
def train_with_seed(n=100000, bs=64, lr=1e-3, ev=2500):
    import torch, random
    torch.manual_seed(seed); random.seed(seed)
    return original_train(n, bs, lr, ev)
# juste lance le training avec le bon seed et un checkpoint différent
import torch, random, json, glob, numpy as np, time
from ocm26400.audio_unified_m5scb import M5Unified, load_wav, spd, load_data, _eval
import torch.nn.functional as F
torch.manual_seed(seed); random.seed(seed)
device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
NW=len(ws); tr,te=load_data(ws); keys=list(tr.keys())
mdl=M5Unified(NW).to(device); opt=torch.optim.Adam(mdl.parameters(),lr=1e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=50000)
CKPT=f"/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed{seed}.pt"
print(f"[ENSEMBLE seed{seed}] 50k steps",flush=True);t0=time.time();best=0
for step in range(50000):
    bi=[random.choice(keys) for _ in range(64)]
    wv=torch.stack([torch.from_numpy(spd(tr[k][torch.randint(0,len(tr[k]),(1,)).item()].numpy())) for k in bi]).to(device)
    wi=torch.tensor(bi,device=device)
    logits=mdl(wv);loss=F.cross_entropy(logits,wi)
    opt.zero_grad();loss.backward();opt.step();sched.step()
    if step%5000==0 or step==49999:
        acc=_eval(mdl,te)
        if acc>best:best=acc;torch.save({"model_state":{k:v.detach().clone() for k,v in mdl.state_dict().items()},"best":best,"seed":seed},CKPT)
        print(f"  s{seed} step {step:>5} test {acc*100:.1f}% (best {best*100:.1f}%) t={time.time()-t0:.0f}s",flush=True)
print(f"[ENSEMBLE seed{seed}] FINAL best={best*100:.1f}%",flush=True)
