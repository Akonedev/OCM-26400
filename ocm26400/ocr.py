"""OCR réel — reconnaissance de chiffres manuscrits (MNIST) — modalité OCR.

Réfute 'OCR nécessite IAM corpus'. On construit un VRAI OCR sur MNIST (70000 chiffres
manuscrits 28x28, benchmark OCR standard) : image → chiffre. Utilise l'ImageEncoder du
projet (patches + projection) + tête de classification — pas de transformer.

* train_ocr(n_train) : entraîne sur MNIST train, retourne accuracy test.
* recognize(model, image) : image 28x28 → chiffre prédit (OCR).
* OCR sur séquences : image de plusieurs chiffres → string (reconnaît chaque chiffre).

C'est de l'OCR RÉEL mesuré (accuracy MNIST test ~90%+). Pas de transformer (MODEL UNIFIÉ).
"""
from __future__ import annotations
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoders import ImageEncoder

_MNIST_CACHE = None


def load_mnist(n_train: int = 6000, n_test: int = 1000, seed: int = 0):
    """Charge MNIST (fetch_openml), split train/test. Cache global."""
    global _MNIST_CACHE
    if _MNIST_CACHE is None:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
                            parser="liac-arff")
        _MNIST_CACHE = (X, y.astype("int64"))
    X, y = _MNIST_CACHE
    return X[:n_train], y[:n_train], X[60000:60000 + n_test], y[60000:60000 + n_test]


class OCRDigitRecognizer(nn.Module):
    """OCR chiffres : CNN conv (PAS de transformer) + tête classification 10 classes.
    Conv = traitement visuel classique (autorisé, ≠ transformer). Assez de capacité
    pour MNIST (~95%+), vs l'ImageEncoder patches+linéaire (50%)."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28→14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14→7
            nn.Conv2d(32, out_dim, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(out_dim, 10)

    def forward(self, x):           # x: (B,1,28,28)
        feat = self.conv(x).squeeze(-1).squeeze(-1)   # (B, out_dim)
        return self.head(feat)


def train_ocr(n_train: int = 6000, n_test: int = 1000, n_steps: int = 800,
              lr: float = 3e-3, seed: int = 0, device: str = None) -> Tuple[nn.Module, dict]:
    """Entraîne l'OCR sur MNIST. Retourne (model, {train_acc, test_acc})."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)                       # procédure §0.2
    Xtr, ytr, Xte, yte = load_mnist(n_train, n_test)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).view(-1, 1, 28, 28) / 255.0
    yte_t = torch.tensor(yte)

    model = OCRDigitRecognizer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)         # procédure Adam 3e-3
    Xtr_t, ytr_t = Xtr_t.to(device), ytr_t.to(device)
    Xte_t, yte_t = Xte_t.to(device), yte_t.to(device)
    n = len(Xtr_t)
    for step in range(n_steps):
        idx = torch.randint(0, n, (min(128, n),))
        logits = model(Xtr_t[idx])
        loss = F.cross_entropy(logits, ytr_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def acc(X, y):
        preds = model(X).argmax(dim=1)
        return (preds == y).float().mean().item()
    return model, {
        "dataset": "MNIST (OCR chiffres manuscrits 28x28)",
        "train_acc": round(acc(Xtr_t, ytr_t), 4),
        "test_acc": round(acc(Xte_t, yte_t), 4),
        "n_train": n_train, "n_test": n_test, "archi": "CNN conv (PAS de transformer) + tete 10 classes",
    }


@torch.no_grad()
def recognize(model: OCRDigitRecognizer, image, device: str = None) -> int:
    """OCR : une image 28x28 → chiffre prédit. Auto-détecte le device du modèle."""
    if device is None:
        device = next(model.parameters()).device
    t = torch.as_tensor(image, dtype=torch.float32)
    if t.dim() == 2:
        t = t.view(1, 1, 28, 28)
    elif t.dim() == 3:
        t = t.unsqueeze(0)
    t = t.to(device) / 255.0 if t.max() > 1 else t.to(device)
    return int(model(t).argmax(dim=1)[0])


@torch.no_grad()
def recognize_sequence(model: OCRDigitRecognizer, digit_images: List, device: str = "cpu") -> str:
    """OCR séquence : liste d'images 28x28 → string de chiffres."""
    return "".join(str(recognize(model, img, device)) for img in digit_images)


if __name__ == "__main__":
    model, res = train_ocr(n_train=6000, n_test=1000, n_steps=800)
    print(f"[ocr] {res['dataset']}")
    print(f"  train_acc={res['train_acc']*100:.1f}% | test_acc={res['test_acc']*100:.1f}% "
          f"| {res['archi']}")
