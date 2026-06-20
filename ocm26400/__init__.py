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
from .spectral_core import SpectralCoreBlock
from .product_key_vocab import ProductKeyVocab
from .tool_policy import ToolPolicy, TaskEncoder, ToolUsingAgent
from .generators import AMVConditionedDecoder
from .skills_system import ExpertSkill, SkillCreator, ExpertSkillRegistry, production_skills
from .expert_agents import ExpertAgentWithSkills, EXPERT_PROMPTS, extended_production_skills
from .agent_swarm import SwarmAgent, SwarmOrchestrator, SwarmConfig, AgentMemory
from .meta_controller import MetaController
from .curriculum import Curriculum
from .phrase import PhraseComposer, phrase_similarity
from .mcp_adapter import McpAdapter, McpTool, default_adapter, adapter_security_audit
from .train import run_pipeline, stage0_build
from .domain_trainer import evaluate_all_domains, cross_domain_chains, reasoning_bench_aime
from .bench_runner import (
    bench_agentic, bench_reasoning, bench_qcm, bench_terminal, run_all_benchmarks,
)
from .eval_harness import (
    BenchmarkItem, ItemResult, EvalReport, BenchmarkRunner,
    compare_to_baselines, random_baseline, total_abstention,
    load_jsonl, synthetic_aime_demo,
)

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
    "SpectralCoreBlock",
    "ProductKeyVocab",
    "ToolPolicy",
    "AMVConditionedDecoder",
    "ExpertSkill",
    "ExpertSkillRegistry",
    "ExpertAgentWithSkills",
    "EXPERT_PROMPTS",
    "SwarmAgent",
    "SwarmOrchestrator",
    "MetaController",
    "Curriculum",
    "PhraseComposer",
    "McpAdapter",
    "McpTool",
    "default_adapter",
    "adapter_security_audit",
    "BenchmarkItem",
    "EvalReport",
    "BenchmarkRunner",
    "compare_to_baselines",
]

