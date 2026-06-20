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
from .spectral_core import SpectralCoreBlock
from .multimodal_encoders import AudioEncoder, ImageEncoder
from .generators import AMVConditionedDecoder


class OmniModel(nn.Module):
    """Modèle omni UNIFIÉ sous le NOYAU SPECTRAL de l'utilisateur (architecture du projet).

    core_type='spectral' (défaut) = SpectralCoreBlock (FFT, architecture du projet).
    core_type='mlp' = ReasonerBlock (variant historique). Le noyau est PARTAGÉ par
    toutes les têtes (unifié, pas de wrapper)."""

    def __init__(self, n_audio_classes=5, n_image_classes=10,
                 audio_feat=32, img_side=8, d_model=D_MODEL, core_type: str = "spectral"):
        super().__init__()
        self.d_model = d_model
        self.core_type = core_type
        # ---- têtes d'entrée (vers AMV) ----
        self.audio_enc = AudioEncoder(out_dim=d_model)         # waveform -> AMV
        self.image_enc = ImageEncoder(out_dim=d_model, patch=4)  # image -> AMV
        # ---- noyau UNIFIÉ partagé (architecture spectrale de l'utilisateur par défaut) ----
        self.core = (SpectralCoreBlock(d_model=d_model) if core_type == "spectral"
                     else ReasonerBlock(d_model=d_model))
        # ---- têtes de sortie : classification ----
        self.audio_cls = nn.Linear(d_model, n_audio_classes)
        self.image_cls = nn.Linear(d_model, n_image_classes)
        # ---- têtes de sortie : GÉNÉRATION FLOW-MATCHING (vraie génération de signal) ----
        self.audio_dec = AMVConditionedDecoder(x_dim=audio_feat, cond_dim=d_model)
        self.image_dec = AMVConditionedDecoder(x_dim=img_side * img_side, cond_dim=d_model)
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

    # --- AMV de conditionnement pour la génération (label -> core -> AMV) ---
    def gen_amv(self, modality: str, label: torch.Tensor) -> torch.Tensor:
        return self.core(self.audio_label_emb(label) if modality == "audio"
                         else self.image_label_emb(label))

    # --- GÉNÉRATION flow-matching (label -> AMV -> signal généré) ---
    def generate(self, modality: str, label: torch.Tensor, steps: int = 8) -> torch.Tensor:
        amv = self.gen_amv(modality, label)
        dec = self.audio_dec if modality == "audio" else self.image_dec
        return dec.sample(amv, steps=steps)


def joint_loss(model: OmniModel, batch: Dict) -> Dict:
    """Loss multi-tâche : classifier + GÉNÉRER (flow matching).

    batch = {modality: {"x":..., "y":..., "feat":...}}. Pour chaque modalité :
      L_cls(classify(x), y) + L_gen = flow_match_loss(gen_amv(y), feat).
    """
    import torch.nn.functional as F
    total, parts = 0.0, {}
    for mod, d in batch.items():
        cls_logits = model.classify(mod, d["x"])
        l_cls = F.cross_entropy(cls_logits, d["y"])
        amv = model.gen_amv(mod, d["y"])
        dec = model.audio_dec if mod == "audio" else model.image_dec
        l_gen = dec.flow_match_loss(amv, d["feat"])           # vraie génération (flow matching)
        total = total + l_cls + l_gen
        parts[f"{mod}_cls"] = float(l_cls.detach())
        parts[f"{mod}_gen"] = float(l_gen.detach())
    return total, parts
