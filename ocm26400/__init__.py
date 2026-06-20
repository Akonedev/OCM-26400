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
from .computer_use import ShellTool, safe_default_allowlist, GUITool
from .orchestrator import Orchestrator, ExpertAgent, DevAdvocate, Judge, MoERouter
from .omni import OmniModel
from .rules import Rule, RuleLibrary
from .agents_tools import Skill, Toolkit, Mission, execute_mission
from .omni_rules import RULES, train_omni_rules, comprehend, generate_chain, inter_rule_gt
from .bench import run_bench, level
from .diff_decode import decode_gumbel, l_step_diff, acsp_loss_diff, train_with_acsp

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
    "GUITool",
    "Orchestrator",
    "MoERouter",
    "OmniModel",
    "RuleLibrary",
    "Toolkit",
    "Mission",
    "train_omni_rules",
    "run_bench",
    "decode_gumbel",
    "acsp_loss_diff",
]

