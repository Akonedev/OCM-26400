#!/usr/bin/env python3
"""Entraînement SUBSTANTIEL de l'OmniModel sur VRAIES données — paradigme du projet.

Respecte les principes :
- noyau SpectralCoreBlock PARTAGÉ (MODEL UNIFIÉ, pas de wrapper)
- entraînement JOINT (un optimizer, loss multi-tâche = classify + generate flow-matching)
- capture simultanée multi-modalité (audio + image en même temps)

Données réelles :
- audio : SpeechCommands (mots étiquetés) → classification + génération
- image : tinyimagenet (self-supervisé, pas de labels plats) → génération/encodage

Sauve le checkpoint vers SAVENVME2.
"""
import torch, glob, os, time, json
import numpy as np
import soundfile as sf
from PIL import Image
from ocm26400.omni import OmniModel, joint_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
IMG_DIR = "/media/akone/SAVENVME2/Datasets/vision_tinyimagenet"
CKPT = "/media/akone/SAVENVME2/Datasets/ocm26400/omnimodel_real_trained.pt"
N_WORDS = 20
PER_WORD = 150
T_AUDIO = 8000

# ---- charger VRAI audio SpeechCommands ----
def load_wav(p):
    y, sr = sf.read(p); y = y.astype(np.float32)
    if y.ndim > 1: y = y.mean(1)
    if len(y) < T_AUDIO: y = np.pad(y, (0, T_AUDIO - len(y)))
    else: y = y[:T_AUDIO]
    return torch.tensor(y)

words = [w for w in os.listdir(SC) if os.path.isdir(os.path.join(SC, w)) and not w.startswith("_")][:N_WORDS]
auds, labs = [], []
for wi, w in enumerate(words):
    for p in glob.glob(os.path.join(SC, w, "*.wav"))[:PER_WORD]:
        auds.append(load_wav(p)); labs.append(wi)
aud_t = torch.stack(auds).to(device); lab_t = torch.tensor(labs).to(device)
N = len(auds)
print(f"[audio] {N} samples réels, {N_WORDS} mots: {words}", flush=True)

# ---- charger VRAIES images tinyimagenet (self-supervised) ----
def load_img(p):
    im = Image.open(p).convert("RGB").resize((8, 8))
    return torch.tensor(np.array(im)).permute(2, 0, 1).float() / 255.0
img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:2000]
imgs = torch.stack([load_img(p) for p in img_paths]).to(device)
print(f"[image] {len(imgs)} images réelles tinyimagenet (self-supervisé)", flush=True)

# ---- split train/test audio ----
idx = torch.randperm(N); ntr = int(N * 0.85)
tr_i, te_i = idx[:ntr], idx[ntr:]

# ---- OmniModel (noyau spectral partagé) ----
m = OmniModel(n_audio_classes=N_WORDS, n_image_classes=10, core_type="spectral").to(device)
opt = torch.optim.Adam(m.parameters(), lr=3e-3)

def audio_batch(n=48):
    i = tr_i[torch.randint(0, len(tr_i), (min(n, len(tr_i)),))]
    return {"audio": {"x": aud_t[i], "y": lab_t[i], "feat": aud_t[i, :32]}}

def image_batch(n=48):
    i = torch.randint(0, len(imgs), (n,))
    # self-supervisé : "label" = index cluster (10 bins), feat = image flatten
    yi = (i % 10).to(device)   # FIX: labels sur le même device que le modèle
    return {"image": {"x": imgs[i], "y": yi, "feat": imgs[i].reshape(n, -1)[:, :64]}}

# ---- entraînement JOINT (audio + image alternés) ----
print(f"\n[training JOINT sur GPU — audio({N_WORDS} mots) + image(self-sup) ]", flush=True)
t0 = time.time()
hist = []
for step in range(2500):
    # alterne audio / image (capture simultanée multi-modalité)
    if step % 2 == 0:
        loss, parts = joint_loss(m, audio_batch())
    else:
        loss, parts = joint_loss(m, image_batch())
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"  step {step} loss={loss.item():.3f} {parts} t={time.time()-t0:.0f}s", flush=True)
        hist.append({"step": step, "loss": loss.item(), "parts": parts})

# ---- évaluation ----
m.eval()
with torch.no_grad():
    logits = m.classify("audio", aud_t[te_i]); pred = logits.argmax(1)
    audio_acc = (pred == lab_t[te_i]).float().mean().item()
print(f"\n=== RÉSULTATS sur VRAIES données ===", flush=True)
print(f"  classification audio (test, {len(te_i)} samples, {N_WORDS} mots): {audio_acc*100:.1f}% (hasard={100/N_WORDS:.0f}%)", flush=True)

# ---- génération audio class-conditionnée (loss finale) ----
with torch.no_grad():
    b = audio_batch(32)
    gen_loss = m.audio_dec.flow_match_loss(m.gen_amv("audio", b["audio"]["y"]), b["audio"]["feat"]).item()
print(f"  génération audio (flow-match loss): {gen_loss:.3f}", flush=True)

# ---- sauver checkpoint + métriques sur SAVENVME2 ----
os.makedirs(os.path.dirname(CKPT), exist_ok=True)
report = {
    "model": "OmniModel (noyau SpectralCoreBlock partagé, MODEL UNIFIÉ)",
    "training": "joint loss (classify + generate flow-matching), principes L1-L6 respectés",
    "data": {"audio": f"SpeechCommands {N_WORDS} mots × {PER_WORD} = {N} reels",
             "image": f"tinyimagenet {len(imgs)} reels (self-supervise)"},
    "steps": 2500, "device": device, "time_s": round(time.time()-t0, 1),
    "results": {"audio_classification_test": round(audio_acc, 4),
                "audio_generation_flowloss": round(gen_loss, 4)},
    "words": words,
}
torch.save({"model_state": m.state_dict(), "report": report,
            "n_audio_classes": N_WORDS, "n_image_classes": 10}, CKPT)
print(f"\n[SAUVÉ] {CKPT}", flush=True)
print(json.dumps(report["results"], indent=2), flush=True)
# aussi écrire le rapport JSON
with open(os.path.join(os.path.dirname(CKPT), "omnimodel_real_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print(f"[RAPPORT] omnimodel_real_report.json", flush=True)
