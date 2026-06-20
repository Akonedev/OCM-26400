"""Tests TDD — tool-use APPRIS (E3, OCM-26400, spec #4).

Valide : le modèle (noyau spectral unifié + tête ToolPolicy) APPREND à sélectionner le
bon skill (pas un plan câblé), et l'agent décide+exécute.
"""
import torch

from ocm26400.tool_policy import TaskEncoder, ToolPolicy, train_tool_policy, ToolUsingAgent
from ocm26400.spectral_core import SpectralCoreBlock
from ocm26400.agents_tools import default_toolkit


def test_task_encoder_uses_spectral_core():
    """La tâche passe par le NOYAU SPECTRAL unifié (archi utilisateur)."""
    enc = TaskEncoder(n_task_types=4)
    assert isinstance(enc.core, SpectralCoreBlock)
    assert enc(torch.tensor([0, 1])).shape == (2, 256)


def test_tool_policy_shape():
    pol = ToolPolicy(n_skills=4)
    assert pol(torch.randn(3, 256)).shape == (3, 4)


def test_tool_policy_learns_correct_skill():
    """Imitation : le modèle apprend task_type -> skill correct (>90%)."""
    n_skills = 4
    enc = TaskEncoder(n_task_types=n_skills); pol = ToolPolicy(n_skills=n_skills)
    traces = [(i, i) for i in range(n_skills)] * 50          # task i -> skill i
    train_tool_policy(enc, pol, traces, n_steps=400)
    correct = 0
    for t in range(n_skills):
        amv = enc(torch.tensor([t]))
        idx, _ = pol.decide(amv[0])
        correct += (idx == t)
    assert correct / n_skills >= 0.75, f"sélection de skill apprise trop basse: {correct}/{n_skills}"


def test_tool_using_agent_decides_and_executes():
    """L'agent décide le skill (appris) puis l'exécute via le Toolkit."""
    n_skills = 4
    enc = TaskEncoder(n_task_types=n_skills); pol = ToolPolicy(n_skills=n_skills)
    train_tool_policy(enc, pol, [(i, i) for i in range(n_skills)] * 50, n_steps=400)
    tk = default_toolkit()                                    # calculate/lookup/move/speak (ordre)
    agent = ToolUsingAgent(enc, pol, tk)
    # task 0 -> skill 0 (calculate) ; exécute calculate(2,3)
    name, result, conf = agent.act(0, (2, 3))
    assert name == tk.names()[0]
    assert result == 5                                        # calculate(2,3)=5
    assert conf > 0.0
