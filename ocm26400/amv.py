"""AMV-256 — Amodal Mentalese Vector (OCM-26400, spec Besoins_Maths.md §1).

  v = [ v_ent(64) || v_prop(64) || v_op(64) || v_meta(64) ] in R^256

Partitionnement dur (QPLS) : les 4 roles sont des tranches contigües fixées,
pas une distribution continue. Les partitions sont des VUES sur le tenseur sous-jacent
(modifications in-place visibles, gradients propagés).
"""
import torch

D_MODEL = 256
PART = 64  # 4 partitions x 64 = 256


class AMVVector:
    """Wrapper léger autour d'un tenseur (256,) exposant les 4 partitions.

    Les attributs ent/prop/op/meta sont des vues (slices) -> toute écriture in-place
    sur une partition se reflète sur .tensor (requis par les tests et par la loss ACSP).
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor):
        assert tensor.shape[-1] == D_MODEL, f"AMV attendu {D_MODEL}, got {tensor.shape}"
        self.tensor = tensor

    # ── fabriques ──
    @classmethod
    def zeros(cls, device=None) -> "AMVVector":
        return cls(torch.zeros(D_MODEL, device=device))

    @classmethod
    def randn(cls, device=None) -> "AMVVector":
        return cls(torch.randn(D_MODEL, device=device))

    # ── partitions (vues) ──
    @property
    def ent(self) -> torch.Tensor:
        return self.tensor[0:64]

    @property
    def prop(self) -> torch.Tensor:
        return self.tensor[64:128]

    @property
    def op(self) -> torch.Tensor:
        return self.tensor[128:192]

    @property
    def meta(self) -> torch.Tensor:
        return self.tensor[192:256]

    # ── méta helpers ──
    # Partition meta(64) (juge 19/06) : évite la contention.
    #   meta[0] = confidence LSRA (reasoner.py)
    #   meta[1] = confidence source/bridge (intégration v6)
    #   meta[2] = score de consistance cross-modale (L_consist)
    def confidence(self) -> torch.Tensor:
        """Confidence LSRA c in [0,1] = sigmoid(meta[0]). Spec §1."""
        return torch.sigmoid(self.meta[0])

    def source_confidence(self) -> torch.Tensor:
        """Confidence de la source/bridge = sigmoid(meta[1]). (juge: rôle dédié)."""
        return torch.sigmoid(self.meta[1])

    def consist_score(self) -> torch.Tensor:
        """Score de consistance cross-modale brut = meta[2]. (juge: rôle dédié)."""
        return self.meta[2]
