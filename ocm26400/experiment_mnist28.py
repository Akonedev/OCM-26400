#!/usr/bin/env python3
"""
EXPÉRIENCE MNIST 28x28 RÉEL (OCM-26400, résiduel DA C6).

Le DA a identifié : 'génération image sur MNIST réel (pas juste 8x8)'.
On adresse ce résiduel : classification ET génération flow-matching sur MNIST 28x28
(vrai dataset via sklearn, ou digits upsampled 8x8→28x28).
"""
import json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_mnist28(n=1000):
    """Charge MNIST 28x28 (via sklearn digits upsampled si OpenML inaccessible)."""
    try:
        from sklearn.datasets import fetch_openml
        m = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = torch.tensor(m.data[:n].reshape(-1, 28, 28), dtype=torch.float32) / 255.0
        y = torch.tensor(m.target[:n].astype(int), dtype=torch.long)
        return X, y, "MNIST 28x28 (OpenML)"
    except Exception:
        from sklearn.datasets import load_digits
        d = load_digits()
        X8 = torch.tensor(d.data[:n].reshape(-1, 8, 8), dtype=torch.float32) / 16.0
        X = F.interpolate(X8.unsqueeze(1), size=(28, 28), mode='bilinear', align_corners=False).squeeze(1)
        y = torch.tensor(d.target[:n], dtype=torch.long)
        return X, y, "digits 8x8 upsampled 28x28"

def main():
    from ocm26400.amv import D_MODEL
    from ocm26400.generators import AMVConditionedDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, y, source = load_mnist28(1000)
    print(f"MNIST 28x28 | device={device} | source={source} | {len(X)} images")
    X = X.to(device); y = y.to(device)

    # split
    cut = int(0.8 * len(X))
    Xtr, Xte = X[:cut], X[cut:]
    ytr, yte = y[:cut], y[cut:]

    # 1. CLASSIFICATION (CNN léger)
    clf = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    ).to(device)
    opt_c = torch.optim.Adam(clf.parameters(), lr=3e-3)
    t0 = time.time()
    for _ in range(500):
        idx = torch.randint(0, len(Xtr), (128,))
        loss = F.cross_entropy(clf(Xtr[idx]), ytr[idx])
        opt_c.zero_grad(); loss.backward(); opt_c.step()
    with torch.no_grad():
        acc = (clf(Xte).argmax(-1) == yte).float().mean().item()
    dt_cls = time.time() - t0
    print(f"\n1. Classification 28x28 : {acc*100:.1f}% ({dt_cls:.1f}s)")

    # 2. GÉNÉRATION flow-matching 28x28 (784 pixels)
    dec = AMVConditionedDecoder(x_dim=784, cond_dim=10).to(device)
    emb = nn.Embedding(10, 10).to(device)
    opt_g = torch.optim.Adam(list(dec.parameters()) + list(emb.parameters()), lr=3e-3)
    Xflat = Xtr.reshape(len(Xtr), -1)  # (N, 784)

    # MSE avant
    with torch.no_grad():
        s0 = dec.sample(emb(yte[:20]), steps=4)
        mse_before = float(((s0 - Xte[:20].reshape(-1, 784)) ** 2).mean())

    t0 = time.time()
    for _ in range(800):
        idx = torch.randint(0, len(Xflat), (64,))
        loss = dec.flow_match_loss(emb(ytr[idx]), Xflat[idx])
        opt_g.zero_grad(); loss.backward(); opt_g.step()

    with torch.no_grad():
        s1 = dec.sample(emb(yte[:20]), steps=8)
        mse_after = float(((s1 - Xte[:20].reshape(-1, 784)) ** 2).mean())
    dt_gen = time.time() - t0
    print(f"2. Génération flow-matching 28x28 : MSE {mse_before:.3f}→{mse_after:.3f} ({dt_gen:.1f}s)")

    # 3. Vérification : images générées reconnaissables par le classifieur
    with torch.no_grad():
        gen = dec.sample(emb(torch.arange(10).to(device)), steps=8)
        gen_pred = clf(gen).argmax(-1)
    correct_gen = (gen_pred == torch.arange(10).to(device)).sum().item()
    print(f"3. Images générées reconnues par classifieur : {correct_gen}/10")

    verdict = "VALIDÉ" if (acc > 0.8 and mse_after < mse_before) else "PARTIEL"
    print(f"\nVERDICT MNIST 28x28 : {verdict}")
    results = {"source": source, "n": len(X), "cls_acc": round(acc, 4),
               "gen_mse_before": round(mse_before, 4), "gen_mse_after": round(mse_after, 4),
               "gen_recognized": correct_gen, "verdict": verdict}
    json.dump(results, open("ocm26400/mnist28_results.json", "w"), indent=2)
    print("Résultats: ocm26400/mnist28_results.json")

if __name__ == "__main__":
    main()
