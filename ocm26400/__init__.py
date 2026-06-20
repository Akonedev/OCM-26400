"""OCM-26400 — Omni-Cognitive Mentalese architecture.

Noyau du joyau spec (Besoins_Maths.md): AMV-256 (vecteur 4-partitions) +
ACSP loss + LSRA (gate de vérification symbolique). Construit en TDD.
"""
from .infonce import info_nce, info_nce_symmetric, multimodal_l_consist
from .learned_vocab import LearnedVocab
from .concept_amodal import ModalityEncoder, amodal_align_loss
from .morphology import MorphologyVerifier, CONJUGATE_PAST, CONJUGATE_GERUND, CONJUGATE_THIRD
from .knowledge_base import KnowledgeBase
from .cognitive_agent import CognitiveAgent
from .compositional_vocab import CompositionalVocabulary
from .sleep import extract_rule, consolidate
from .real_linguistic import RealViewEncoder, load_real_words, MODALITIES as LING_MODALITIES
from .multimodal_encoders import AudioEncoder, ImageEncoder, VideoEncoder, ThreeDEncoder
from .tools import LearningAgent, StaticTool, Tool
from .self_correction import self_correct, self_improve, self_consistency_confidence
from .web_tools import WebFetchTool, URLMemory, fetch_url
from .computer_use import ShellTool, safe_default_allowlist

__all__ = [
    "info_nce",
    "info_nce_symmetric",
    "multimodal_l_consist",
    "LearnedVocab",
    "ModalityEncoder",
    "amodal_align_loss",
    "MorphologyVerifier",
    "CONJUGATE_PAST",
    "CONJUGATE_GERUND",
    "CONJUGATE_THIRD",
    "KnowledgeBase",
    "CognitiveAgent",
    "CompositionalVocabulary",
    "extract_rule",
    "consolidate",
    "RealViewEncoder",
    "load_real_words",
    "AudioEncoder",
    "ImageEncoder",
    "VideoEncoder",
    "ThreeDEncoder",
    "LearningAgent",
    "StaticTool",
    "self_correct",
    "self_improve",
    "WebFetchTool",
    "URLMemory",
    "ShellTool",
]

