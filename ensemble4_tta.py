#!/usr/bin/env python3
"""Ensemble 4-modèles (3×M5 + 1×M11 résiduel) + TTA → dépasser SOTA 96%.

Diversité d'ARCHITECTURE (M5 simple + M11 résiduel) → erreurs plus différentes →
gain ensemble plus grand que seeds seuls. + TTA (3 speed rates).
"""
import torch,torch.nn.functional as F,glob,os,numpy as np,json,random
from ocm26400.audio_unified_m5scb import M5Unified, load_wav, load_data, spd
from ocm26400.audio_m11_ce import M11Unified

device="cuda"
SC="/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
MODELS=[
    ("M5_s0","/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_trained.pt",M5Unified),
    ("M5_s1","/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed1.pt",M5Unified),
    ("M5_s2","/media/akone/SAVENVME2/Datasets/ocm26400/audio_unified_seed2.pt",M5Unified),
    ("M11","/media/akone/SAVENVME2/Datasets/ocm26400/audio_m11_trained.pt",M11Unified),
]
TTA_RATES=[1.0,0.95,1.05]

@torch.no_grad()
def main():
    ws=sorted([w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC,w)) and not w.startswith("_")])
    NW=len(ws); tr,te=load_data(ws)
    models=[]
    for name,ckpt,cls in MODELS:
        if os.path.exists(ckpt):
            m=cls(NW).to(device)
            ck=torch.load(ckpt,map_location=device,weights_only=True)
            m.load_state_dict(ck["model_state"]);m.eval()
            models.append((name,m))
            print(f"  {name}: loaded (best was {ck.get('best',0)*100:.1f}%)",flush=True)
    print(f"\n[ENSEMBLE 4-arch] {len(models)} modèles × {len(TTA_RATES)} TTA",flush=True)
    ok_ens=tot=0; ok_tta=0; ok_indiv={n:0 for n,_ in models}
    for wi in te:
        for j in range(0,len(te[wi]),32):
            batch=te[wi][j:j+32];B=len(batch)
            probs_ens=torch.zeros(B,NW,device=device);probs_tta=torch.zeros(B,NW,device=device)
            for name,m in models:
                wavs=batch.to(device)
                logits=m(wavs)
                probs_ens+=F.softmax(logits,-1);probs_tta+=F.softmax(logits,-1)
                ok_indiv[name]+=(logits.argmax(1).cpu()==wi).sum().item()
                for rate in TTA_RATES[1:]:
                    wv_aug=torch.stack([torch.from_numpy(spd(w.numpy(),(rate-0.05,rate+0.05))) for w in batch]).to(device)
                    probs_tta+=F.softmax(m(wv_aug),-1)
            ok_ens+=(probs_ens.argmax(1).cpu()==wi).sum().item()
            ok_tta+=(probs_tta.argmax(1).cpu()==wi).sum().item();tot+=B
    acc_ens=ok_ens/max(tot,1);acc_tta=ok_tta/max(tot,1)
    print(f"\n{'='*60}")
    print(f"ENSEMBLE 4-ARCHITECTURES + TTA — test officiel")
    print(f"{'='*60}")
    for n,_ in models:print(f"  {n} individuel: {ok_indiv[n]/max(tot,1)*100:.1f}%")
    print(f"  Ensemble simple ({len(models)} modèles):  {acc_ens*100:.1f}%")
    print(f"  Ensemble + TTA ({len(TTA_RATES)} rates):       {acc_tta*100:.1f}%")
    best_single=max(ok_indiv.values())/max(tot,1)
    print(f"  Δ ensemble+TTA vs best single:          {(acc_tta-best_single)*100:+.1f}pt")
    print(f"  SOTA SpeechCommands:                    ~96%")
    print(f"  Δ vs SOTA:                              {acc_tta*100-96:+.1f}pt")
    json.dump({"ensemble_tta_acc":acc_tta,"ensemble_acc":acc_ens,"n_models":len(models),
               "individual":{n:ok_indiv[n]/max(tot,1) for n,_ in models},"delta_vs_sota_96":acc_tta*100-96},
              open("ocm26400/audio_ensemble4_tta_results.json","w"),indent=2)
    print("  [sauvé] ocm26400/audio_ensemble4_tta_results.json")

if __name__=="__main__":main()
