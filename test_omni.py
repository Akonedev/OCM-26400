#!/usr/bin/env python3
"""Interface de test de l'OmniModel entraîné sur vraies données.

Usage:
  python test_omni.py audio_classify <wav_file>     # classer un wav réel
  python test_omni.py audio_generate <label 0-19>   # générer un waveform
  python test_omni.py image_classify <png_file>     # classer une image réelle
  python test_omni.py reason "<question>"           # raisonnement textuel
  python test_omni.py bench                          # benchmark complet (tous modes)
  python test_omni.py words                          # liste les mots appris

Charge le checkpoint SAVENVME2/ocm26400/omnimodel_real_trained.pt
"""
import sys, os, torch, numpy as np
CKPT = "/media/akone/SAVENVME2/Datasets/ocm26400/omnimodel_real_trained.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    from ocm26400.omni import OmniModel
    ck = torch.load(CKPT, map_location=device, weights_only=True)
    m = OmniModel(n_audio_classes=ck["n_audio_classes"], n_image_classes=ck["n_image_classes"],
                  core_type="spectral").to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, ck["report"]


def load_wav(p, T=8000):
    import soundfile as sf
    y, sr = sf.read(p); y = y.astype(np.float32)
    if y.ndim > 1: y = y.mean(1)
    if len(y) < T: y = np.pad(y, (0, T - len(y)))
    else: y = y[:T]
    return torch.tensor(y).unsqueeze(0).to(device)


def load_img(p):
    from PIL import Image
    im = Image.open(p).convert("RGB").resize((8, 8))
    return torch.tensor(np.array(im)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def cmd_words(rep):
    print("Mots appris:", rep["words"])


def cmd_audio_classify(path, m, rep):
    wav = load_wav(path)
    with torch.no_grad():
        logits = m.classify("audio", wav); pred = logits.argmax(1).item()
    print(f"[audio_classify] {path}")
    print(f"  -> mot prédit: '{rep['words'][pred]}' (label {pred}/{len(rep['words'])})")


def cmd_audio_generate(label, m, rep):
    lab = torch.tensor([int(label)]).to(device)
    with torch.no_grad():
        wav = m.generate("audio", lab, steps=12)
    print(f"[audio_generate] label {label} ('{rep['words'][int(label)]}')")
    print(f"  -> waveform généré {tuple(wav.shape)}, énergie {wav.norm():.2f}")


def cmd_image_classify(path, m, rep):
    img = load_img(path)
    with torch.no_grad():
        logits = m.classify("image", img); pred = logits.argmax(1).item()
    print(f"[image_classify] {path}")
    print(f"  -> classe prédite: {pred}/10")


def cmd_reason(question, m, rep):
    from ocm26400.language_cascade_grok import solve_gsm8k_grokked
    pred, trace = solve_gsm8k_grokked(question)
    print(f"[reason] Q: {question}")
    print(f"  -> réponse: {pred}")
    print(f"  trace: {trace[:5]}")


def cmd_bench(m, rep):
    """Benchmark complet de tous les modes."""
    import glob
    print(f"=== BENCHMARK OmniModel (entraîné sur vraies données) ===")
    print(f"  rapport: {rep['results']}\n")
    # audio classification sur quelques wavs réels
    SC = "/media/akone/SAVENVME2/Datasets/_speechcommands_cache/SpeechCommands/speech_commands_v0.02"
    ok = 0; tot = 0
    for wi, w in enumerate(rep["words"][:10]):
        fs = glob.glob(os.path.join(SC, w, "*.wav"))[100:103]  # samples hors-training
        for p in fs:
            wav = load_wav(p)
            with torch.no_grad():
                pred = m.classify("audio", wav).argmax(1).item()
            ok += (pred == wi); tot += 1
    print(f"  audio classification (30 samples, 10 mots): {ok}/{tot} = {ok/max(tot,1)*100:.0f}%")
    # génération audio
    with torch.no_grad():
        wav = m.generate("audio", torch.tensor([0]).to(device), steps=12)
    print(f"  audio génération (label 0): waveform {tuple(wav.shape)}, énergie {wav.norm():.2f}")
    # image classification
    IMG = "/media/akone/SAVENVME2/Datasets/vision_tinyimagenet"
    p = sorted(glob.glob(os.path.join(IMG, "*.png")))[2050]
    img = load_img(p)
    with torch.no_grad():
        ipred = m.classify("image", img).argmax(1).item()
    print(f"  image classification ({os.path.basename(p)}): classe {ipred}/10")
    # raisonnement
    cmd_reason("3 + 4 x 2", m, rep)


if __name__ == "__main__":
    if not os.path.exists(CKPT):
        print(f"Checkpoint introuvable: {CKPT}\nLance d'abord: python train_real_full.py")
        sys.exit(1)
    m, rep = load_model()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "bench"
    if cmd == "words":
        cmd_words(rep)
    elif cmd == "audio_classify":
        cmd_audio_classify(sys.argv[2], m, rep)
    elif cmd == "audio_generate":
        cmd_audio_generate(sys.argv[2], m, rep)
    elif cmd == "image_classify":
        cmd_image_classify(sys.argv[2], m, rep)
    elif cmd == "reason":
        cmd_reason(sys.argv[2], m, rep)
    elif cmd == "bench":
        cmd_bench(m, rep)
    else:
        print(__doc__)
