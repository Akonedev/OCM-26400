"""Encodeur audio ResNet-style (PROFONDEUR via résiduels) + cross-modal simultané.

Principe: profondeur > params. ResNet = profondeur SANS explosion de params
(connections résiduelles). 8 conv layers avec résiduels = profondeur réelle.

Le SpectralCoreBlock grok sur les features profondes de l'encodeur ResNet.
Cross-modal: texte + phonétique ancre l'apprentissage. 1-cos loss (crown-jewel).
"""
import torch, torch.nn as nn, torch.nn.functional as F, glob, os, numpy as np, time
import soundfile as sf
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab
device="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0)
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
T=8000

class ResNetAudio(nn.Module):
    """Encodeur audio ResNet-style: Mel → 8 conv résiduels → invariant.
    Profondeur via résiduels (pas plus de params — L4)."""
    def __init__(self,out_dim=D_MODEL,n_mels=64):
        super().__init__(); self.n_fft=256; self.n_mels=n_mels
        fb=torch.zeros(n_mels,self.n_fft//2+1)
        for m in range(n_mels):
            c=(m+1)*(self.n_fft//2)/(n_mels+1)
            for f in range(self.n_fft//2+1): fb[m,f]=max(0,1-abs(f-c)/max(1,self.n_fft//2/(n_mels+1)))
        self.register_buffer("mel_fb",fb/(fb.sum(1,keepdim=True)+1e-8))
        self.register_buffer("win",torch.hann_window(self.n_fft))
        # 8 conv layers avec résiduels (ResNet-style profondeur)
        self.conv1=nn.Conv1d(n_mels,128,3,padding=1); self.bn1=nn.BatchNorm1d(128)
        self.conv2=nn.Conv1d(128,128,3,padding=1); self.bn2=nn.BatchNorm1d(128)
        self.conv3=nn.Conv1d(128,128,3,padding=1); self.bn3=nn.BatchNorm1d(128)
        self.conv4=nn.Conv1d(128,128,3,padding=1); self.bn4=nn.BatchNorm1d(128)
        self.conv5=nn.Conv1d(128,64,3,padding=1); self.bn5=nn.BatchNorm1d(64)
        self.conv6=nn.Conv1d(64,64,3,padding=1); self.bn6=nn.BatchNorm1d(64)
        self.conv7=nn.Conv1d(64,64,3,padding=1); self.bn7=nn.BatchNorm1d(64)
        self.conv8=nn.Conv1d(64,32,3,padding=1)
        self.proj=nn.Linear(32,out_dim)
    def forward(self,wav):
        spec=torch.stft(wav,n_fft=self.n_fft,hop_length=self.n_fft//2,
            win_length=self.n_fft,window=self.win,return_complex=True,center=False)
        mel=torch.log1p(self.mel_fb@(spec.abs()**2))
        # ResNet blocks (résiduels = profondeur sans explosion)
        h=F.relu(self.bn1(self.conv1(mel)))
        h=h+F.relu(self.bn2(self.conv2(h)))  # res 1
        h=h+F.relu(self.bn3(self.conv3(h)))  # res 2
        h=h+F.relu(self.bn4(self.conv4(h)))  # res 3
        h2=F.relu(self.bn5(self.conv5(h)))
        h2=h2+F.relu(self.bn6(self.conv6(h2)))  # res 4
        h2=h2+F.relu(self.bn7(self.conv7(h2)))  # res 5 (64ch)
        h3=self.conv8(h2)  # 64→32 (pas de res, changement dim)
        return self.proj(h3.mean(dim=-1))

def text_feat(w):
    v=np.zeros(PART,dtype=np.float32)
    for c in w.lower(): v[(ord(c)*167)%PART]+=1
    return v
def phon_feat(w):
    w=w.lower();vw=sum(1 for c in w if c in"aeiou");cs=len(w)-vw
    p="".join("v" if c in"aeiou" else"c" for c in w)[:8]
    v=np.zeros(PART,dtype=np.float32)
    for c in p:v[(ord(c)*167)%PART]+=1
    v[(vw*7)%PART]+=1;v[(cs*11+PART//2)%PART]+=1
    return v
def load_wav(p):
    y,sr=sf.read(p);y=y.astype(np.float32)
    if y.ndim>1:y=y.mean(1)
    if len(y)<T:y=np.pad(y,(0,T-len(y)))
    else:y=y[:T]
    return torch.tensor(y)

words=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
NW=len(words)
print(f"[ResNet-8 + cross-modal] {NW} mots, 150 samples/mot, 30000 steps",flush=True)
audio_by_word={}
for wi,w in enumerate(words):
    wavs=[load_wav(p) for p in glob.glob(os.path.join(SC,w,"*.wav"))[:150]]
    audio_by_word[wi]=torch.stack(wavs).to(device)
text_all=torch.tensor([text_feat(w) for w in words]).to(device)
phon_all=torch.tensor([phon_feat(w) for w in words]).to(device)
cv=LearnedVocab(n=NW,dim=PART,init="ortho" if NW<=PART else"random",seed=0);cv.freeze()
canon=cv._matrix().to(device)

class M(nn.Module):
    def __init__(s):
        super().__init__();s.enc=ResNetAudio();s.tp=nn.Linear(PART,D_MODEL)
        s.pp=nn.Linear(PART,D_MODEL);s.core=SpectralCoreBlock(d_model=D_MODEL,seq_len=1)
        s.head=nn.Linear(D_MODEL,PART)
    def fwd_view(s,f,p):
        return s.head(s.core(p(f).unsqueeze(1)).squeeze(1))
    def fwd_audio(s,w):
        return s.head(s.core(s.enc(w).unsqueeze(1)).squeeze(1))

m=M().to(device);opt=torch.optim.Adam(m.parameters(),lr=3e-3)
a_tr,a_te={},{}
for wi in range(NW):
    n=len(audio_by_word[wi]);p=torch.randperm(n);ne=max(1,n//5)
    a_te[wi]=p[:ne];a_tr[wi]=p[ne:]
t0=time.time()
for step in range(30000):
    wb=torch.randint(0,NW,(32,));tgt=canon[wb]
    ot=m.fwd_view(text_all[wb],m.tp);op=m.fwd_view(phon_all[wb],m.pp)
    wavs=torch.stack([audio_by_word[wi.item()][a_tr[wi.item()][torch.randint(0,len(a_tr[wi.item()]),(1,)).item()]] for wi in wb])
    oa=m.fwd_audio(wavs)
    loss=((1-F.cosine_similarity(ot,tgt).clamp(-1,1)).mean()+(1-F.cosine_similarity(op,tgt).clamp(-1,1)).mean()+(1-F.cosine_similarity(oa,tgt).clamp(-1,1)).mean())
    opt.zero_grad();loss.backward();opt.step()
    if step%5000==0:
        m.eval()
        with torch.no_grad():
            ok=sum(1 for wi in range(NW) for j in a_te[wi][:2] if(m.fwd_audio(audio_by_word[wi][j:j+1])@canon.t()).argmax(1).item()==wi)
            tot=sum(min(2,len(a_te[wi])) for wi in range(NW))
        print(f"  step {step:>5} loss={loss.item():.4f} test={ok}/{tot}({ok/max(tot,1)*100:.1f}%) t={time.time()-t0:.0f}s",flush=True)
        m.train()
m.eval()
with torch.no_grad():
    ok=sum(1 for wi in range(NW) for j in a_te[wi] if(m.fwd_audio(audio_by_word[wi][j:j+1])@canon.t()).argmax(1).item()==wi)
    tot=sum(len(a_te[wi]) for wi in range(NW))
acc=ok/max(tot,1)
print(f"\n=== ResNet-8 audio + cross-modal ===")
print(f"  TEST(OOD): {ok}/{tot}={acc*100:.1f}% (hasard {100/NW:.1f}%)")
print(f"  SOTA ~96% | gap {acc*100-96:+.1f}pt | temps {time.time()-t0:.0f}s")
torch.save({"model_state":m.state_dict(),"acc":acc,"words":words},
    "/media/akone/SAVENVME2/Datasets/ocm26400/resnet8_trained.pt")
print(f"  [SAUVÉ] resnet8_trained.pt")
