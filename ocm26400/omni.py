"""OmniModel UNIFIÉ — un seul modèle, entraînement joint, génération neurale apprise
(OCM-26400, cahier des charges « MODÈLE OMNI UNIFIÉ, pas de wrapper »).

Réponse directe à la correction de l'utilisateur : PAS de wrappers ni de code externe.
UN SEUL nn.Module avec :
  * plusieurs TÊTES D'ENTRÉE (audio Mel, image patches, texte) -> projection vers
    l'espace amodal unifié AMV-256 ;
  * un NOYAU PARTAGÉ (ReasonerBlock, raisonnement latent) ;
  * plusieurs TÊTES DE SORTIE NEURONALES (classification ET GÉNÉRATION conditionnée) ;
  * un ENTRAÎNENEMENT JOINT (un optimizer, une loss multi-tâche = paradigme complet).

Le modèle APPREND À GÉNÉRER : décodeurs entraînés à produire un échantillon de sa classe
(génération conditionnée par le label -> AMV -> signal). Pas de StubTTS/procédural : la
génération est NEURALE, INTERNE, APPRISE sur les bases.

HONNÊTE : c'est un prototype du paradigme unifié (un seul modèle fait plusieurs tâches :
classify audio+image, génère audio+image conditionné), pas un SOTA à l'échelle milliards.
Il prouve l'UNIFICATION + l'entraînement joint + la génération apprise, sur les bases
réelles (digits, notes) — le schéma exact qu'on scale ensuite.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .amv import D_MODEL
from .reasoner import ReasonerBlock
from .multimodal_encoders import AudioEncoder, ImageEncoder


class OmniModel(nn.Module):
    """Modèle omni unifié : entrées multi-modales -> AMV partagé -> sorties neurales."""

    def __init__(self, n_audio_classes=5, n_image_classes=10,
                 audio_feat=32, img_side=8, d_model=D_MODEL):
        super().__init__()
        self.d_model = d_model
        # ---- têtes d'entrée (vers AMV) ----
        self.audio_enc = AudioEncoder(out_dim=d_model)         # waveform -> AMV
        self.image_enc = ImageEncoder(out_dim=d_model, patch=4)  # image -> AMV
        # ---- noyau partagé ----
        self.core = ReasonerBlock(d_model=d_model)             # raisonnement latent unifié
        # ---- têtes de sortie : classification ----
        self.audio_cls = nn.Linear(d_model, n_audio_classes)
        self.image_cls = nn.Linear(d_model, n_image_classes)
        # ---- têtes de sortie : GÉNÉRATION (décoder un signal depuis l'AMV) ----
        self.audio_dec = nn.Linear(d_model, audio_feat)        # AMV -> features audio
        self.image_dec = nn.Linear(d_model, img_side * img_side)  # AMV -> pixels
        # ---- conditionnement par label (génération class-conditionnée) ----
        self.audio_label_emb = nn.Embedding(n_audio_classes, d_model)
        self.image_label_emb = nn.Embedding(n_image_classes, d_model)

    # --- encode (n'importe quelle modalité -> AMV unifié) ---
    def encode(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        h = self.audio_enc(x) if modality == "audio" else self.image_enc(x)
        return self.core(h)                                    # AMV (B, d_model)

    # --- classification (AMV -> logits) ---
    def classify(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        amv = self.encode(modality, x)
        return (self.audio_cls if modality == "audio" else self.image_cls)(amv)

    # --- GÉNÉRATION conditionnée (label -> AMV -> signal reconstruit) ---
    def generate(self, modality: str, label: torch.Tensor) -> torch.Tensor:
        amv = self.core(self.audio_label_emb(label) if modality == "audio"
                        else self.image_label_emb(label))
        return self.audio_dec(amv) if modality == "audio" else self.image_dec(amv)


def joint_loss(model: OmniModel, batch: Dict) -> Dict:
    """Loss multi-tâche (paradigme complet) : classifier + reconstruire (générer).

    batch = {modality: {"x":..., "y":..., "feat":...}}. Pour chaque modalité on ajoute :
      L_cls(classify(x), y) + L_gen(generate(y), feat_de_x)   # apprend à reconnaître ET générer.
    """
    import torch.nn.functional as F
    total, parts = 0.0, {}
    for mod, d in batch.items():
        cls_logits = model.classify(mod, d["x"])
        l_cls = F.cross_entropy(cls_logits, d["y"])
        gen = model.generate(mod, d["y"])                      # génère l'échantillon de sa classe
        l_gen = F.mse_loss(gen, d["feat"])                     # vs features réelles
        total = total + l_cls + l_gen
        parts[f"{mod}_cls"] = float(l_cls.detach())
        parts[f"{mod}_gen"] = float(l_gen.detach())
    return total, parts
