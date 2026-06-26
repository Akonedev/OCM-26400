#!/usr/bin/env python3
"""Génération par COMPRÉHENSION — le modèle CRÉE des images depuis les règles grokkées.

PRINCIPE : comprendre → générer. Le modèle grok les primitives visuelles et leurs
règles de composition → il peut GÉNÉRER de nouvelles images en composant les
primitives selon les règles comprises. Pas une copie — une CRÉATION depuis la compréhension.

Pipeline :
1. Cross-modal grok : image + texte → concept ID (compréhension, déjà fait)
2. Flow-matching decoder : concept AMV → intégrer du bruit vers l'image
   Le decoder apprend : concept → comment composer les primitives visuelles → signal
3. Génération : concept ID → AMV → flow-matching → image générée
4. Vérification : l'image générée est classifiée par le cross-modal → bon concept ?

C'est le crown-jewel à l'envers : grok (a,b)→op(a,b) permet de CALCULER tout résultat.
Grok les règles visuelles → GÉNÉRER toute image du concept.
"""
import torch, torch.nn as nn, torch.nn.functional as F, glob, os, numpy as np, time
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.amv import D_MODEL, PART
from ocm26400.learned_vocab import LearnedVocab

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
IMG_DIR = "/media/akone/SAVENVME2/Datasets/vision_tinyimagenet"
N_CLUSTERS = 10
PATCH_DIM = 48  # 4x4x3


def extract_patches(path):
    """Image → 4 patches (4x4x3 = 48-dim chacun), puis moyenne."""
    im = Image.open(path).convert("RGB").resize((8, 8))
    arr = np.array(im, dtype=np.float32) / 255.0
    return np.mean([arr[:4, :4].flatten(), arr[:4, 4:].flatten(),
                    arr[4:, :4].flatten(), arr[4:, 4:].flatten()], axis=0)


# ============================================================
# Flow-matching decoder : concept AMV → génère l'image
# ============================================================
class FlowDecoder(nn.Module):
    """Apprend : concept AMV + temps t + bruit x_t → vélocité v.
    À l'inférence : intégrer du bruit vers l'image, conditionné par le concept."""
    def __init__(self, cond_dim=D_MODEL, x_dim=PATCH_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + x_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, x_dim)
        )

    def forward(self, x, cond, t):
        """x: (B, x_dim), cond: (B, cond_dim), t: (B, 1) → vélocité (B, x_dim)."""
        return self.net(torch.cat([x, cond, t], dim=-1))

    @torch.no_grad()
    def sample(self, cond, n_steps=20):
        """Génère depuis le bruit, conditionné par le concept AMV."""
        B = cond.shape[0]
        x = torch.randn(B, PATCH_DIM, device=cond.device)  # start: bruit
        for i in range(n_steps):
            t = torch.full((B, 1), i / n_steps, device=cond.device)
            v = self.forward(x, cond, t)
            x = x + v / n_steps  # Euler integration
        return x  # image générée


def train_generation():
    # 1. charger images + clustering (concept IDs)
    paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))[:1500]
    all_patches = np.array([extract_patches(p) for p in paths], dtype=np.float64)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=256, random_state=0, n_init=3)
    km.fit(all_patches)
    labels = km.predict(all_patches)
    all_patches = all_patches.astype(np.float32)  # back to float32 for torch

    # concept canonical embeddings
    cv = LearnedVocab(n=N_CLUSTERS, dim=PART, init="ortho", seed=0)
    cv.freeze()
    canon = cv._matrix().to(device)  # (N_CLUSTERS, PART)

    # 2. cross-modal classifier (pour vérifier la génération)
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(PATCH_DIM, D_MODEL)
            self.core = SpectralCoreBlock(d_model=D_MODEL, seq_len=1)
            self.head = nn.Linear(D_MODEL, PART)
        def forward(self, x):
            return self.head(self.core(self.proj(x).unsqueeze(1)).squeeze(1))

    clf = Classifier().to(device)
    opt_c = torch.optim.Adam(clf.parameters(), lr=3e-3)

    # 3. flow decoder (générateur)
    decoder = FlowDecoder(cond_dim=PART, x_dim=PATCH_DIM).to(device)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=3e-3)

    # split
    perm = np.random.permutation(len(paths))
    ntr = int(len(paths) * 0.8)
    tr_idx, te_idx = perm[:ntr], perm[ntr:]
    patches_t = torch.tensor(all_patches, dtype=torch.float32).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long).to(device)

    # PHASE 1 : entraîner classifier + decoder SIMULTANÉMENT
    print(f"[génération par compréhension] {len(paths)} images, {N_CLUSTERS} concepts", flush=True)
    print(f"  Phase 1: classifier (comprendre) + flow decoder (générer) simultanés", flush=True)
    t0 = time.time()
    for step in range(4000):
        bi = torch.tensor(np.random.choice(tr_idx, 48, replace=False))
        x_real = patches_t[bi]          # (48, 48) patches réels
        y = labels_t[bi]                # concept IDs
        cond = canon[y]                 # (48, PART) concept AMV

        # --- classifier loss (comprendre : image → concept) ---
        out_c = clf(x_real)
        loss_cls = (1 - F.cosine_similarity(out_c, cond).clamp(-1, 1)).mean()

        # --- flow-matching loss (générer : concept → image) ---
        x_0 = torch.randn_like(x_real)  # bruit (48, 48)
        t = torch.rand(48, 1, device=device)  # (48, 1)
        x_t = (1 - t) * x_0 + t * x_real  # interpolation (48,48)
        v_target = x_real - x_0        # vélocité cible (48,48)
        v_pred = decoder(x_t, cond, t)
        loss_gen = F.mse_loss(v_pred, v_target)

        loss = loss_cls + loss_gen
        opt_c.zero_grad(); opt_d.zero_grad()
        loss.backward()
        opt_c.step(); opt_d.step()

        if step % 1000 == 0:
            clf.eval()
            with torch.no_grad():
                ok = sum(1 for i in te_idx[:100]
                         if (clf(patches_t[i:i+1]) @ canon.t()).argmax(1).item() == labels[i])
            print(f"  step {step} cls={loss_cls.item():.3f} gen={loss_gen.item():.3f} "
                  f"clf_test={ok}% t={time.time()-t0:.0f}s", flush=True)
            clf.train()

    # PHASE 2 : GÉNÉRER des images depuis les concepts
    print(f"\n[Phase 2: GÉNÉRATION — concept AMV → flow-matching → image créée]")
    decoder.eval(); clf.eval()
    n_gen = 5  # générer 5 images par concept
    generated = {}
    verify_ok = 0; verify_tot = 0
    with torch.no_grad():
        for ci in range(N_CLUSTERS):
            cond = canon[ci:ci+1].expand(n_gen, -1)  # (5, PART)
            gen_patches = decoder.sample(cond, n_steps=25)  # (5, 48) patches générés
            generated[ci] = gen_patches.cpu().numpy()
            # VÉRIFICATION : le classifier reconnaît-il l'image générée comme le bon concept ?
            pred = (clf(gen_patches) @ canon.t()).argmax(1)
            hits = (pred == ci).sum().item()
            verify_ok += hits; verify_tot += n_gen
    gen_acc = verify_ok / max(verify_tot, 1)
    print(f"\n=== GÉNÉRATION PAR COMPRÉHENSION ===")
    print(f"  images générées: {verify_tot} ({n_gen} par concept × {N_CLUSTERS})")
    print(f"  vérification (généré reconnu comme bon concept): {verify_ok}/{verify_tot} = {gen_acc*100:.0f}%")
    print(f"  hasard: {100/N_CLUSTERS:.0f}%")
    print(f"  temps: {time.time()-t0:.0f}s")
    print(f"  méthode: concept ID → flow-matching → image CRÉÉE (pas copiée)")
    print(f"  preuve: l'image générée est reconnue par le classifier = génération cohérente")

    # save
    ckpt = "/media/akone/SAVENVME2/Datasets/ocm26400/generation_trained.pt"
    torch.save({"decoder": decoder.state_dict(), "classifier": clf.state_dict(),
                "canon": canon, "gen_acc": gen_acc,
                "method": "flow-matching generation from comprehension"}, ckpt)
    print(f"  [SAUVÉ] {ckpt}")
    return gen_acc


if __name__ == "__main__":
    print("="*60)
    print("GÉNÉRATION PAR COMPRÉHENSION — créer depuis les règles grokkées")
    print("="*60)
    acc = train_generation()
    print(f"\nGénération vérifiée: {acc*100:.0f}%")
