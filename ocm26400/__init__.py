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
from .symbolic_math import symbolic_math_rules, poly_deriv, gcd, modexp, factorize
from .document_learner import DocumentLearner, text_embedding
from .neural_multihop import neural_holdout_eval, neural_multihop_eval
from .morphology_fr import conjugate, fr_conjugation_rules
from .equation_solver import solve_equation, derivative, integrate, simplify
from .text_decoder import CharGenerator, train_char_generator, reconstruct
from .cot_arithmetic import eval_expr, step, ReasoningStep, CotTrace, solve_word_problem
from .physics_units import Quantity, newton_second, kinetic_energy, ohms_law, verify_law
from .sleep_phases import light_sleep, deep_sleep, paradoxal_sleep, full_night
from .semantic_embeddings import SemanticEmbeddings
from .code_generator import generate, verify_code, generate_and_verify, coverage
from .browser_tool import InteractiveBrowser
from .in_context import predict_from_context, learn_rule_from_context, in_context_accuracy
from .artefact_generator import generate_chart, generate_slides, available_generators
from .procedural_memory import ProceduralMemory, Procedure, default_procedures
from .youtube_learner import fetch_transcript, learn_from_youtube, get_metadata
from .owasp_scanner import scan_code, scan_report
from .cognition import TheoryOfMind, evaluate_morality, SpatialReasoner, structure_mapping
from .commonsense import CommonSense, default_commonsense
from .stream import stream_chars, stream_cot, TokenStream
from .chemistry import molar_mass, balance_simple, is_balanced, parse_formula
from .genetics import punnett_square, phenotype_ratios, mendelian_cross
from .finance import compound_interest, loan_payment, total_paid
from .logic_engine import evaluate, is_tautology, valid_argument, truth_table
from .nlp_tools import translate, sentiment, summarize
from .graph_algorithms import dijkstra, astar, bfs, dfs
from .hypercomplex import Quaternion, hamilton_identity, rotate_vector
from .calibration import brier_score, expected_calibration_error, confidence_summary
from .continual_learning import EWCCallback, demo_ewc
from .language_primitives import lemmatize_fr, lemmatize_en, inflect_adjective
from .linguistics import capture_all, phonemes, morphemes_of
from .unified_capture import UnifiedCapture, ConceptCapture
from .bpe_tokenizer import BPETokenizer, train_default
from .syntax_parser import parse, pos_tag, SyntacticStructure
from .mcts_planner import MCTS, MCTSNode
from .explainer import explain_deduction, explain_abstention, Explanation
from .curiosity import CuriosityDrive, select_curious
from .multihop_rag import multi_hop_retrieve, rerank
from .phonology import to_ipa, classify_sounds, elision, liaison
from .world_model import WorldModel, train_world_model, rollout
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

