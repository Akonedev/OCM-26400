"""Tests TDD — agents experts + prompts + skills production-grade (OCM-26400).

Valide : prompts système experts par domaine, agent résout avec skill + quality_check,
skills étendus (cybersécurité, UX, game-dev, recherche), production-grade.
"""
from ocm26400.expert_agents import (
    EXPERT_PROMPTS, ExpertAgentWithSkills, extended_production_skills,
)


def test_expert_prompts_cover_domains():
    """Les prompts système couvrent les domaines clés (dev, security, UX, recherche, game)."""
    for d in ["development", "cybersecurity", "ux_design", "research", "game_dev",
              "writing", "marketing", "video_production", "chemistry", "biology", "economics"]:
        assert d in EXPERT_PROMPTS and len(EXPERT_PROMPTS[d]) > 50


def test_expert_agent_solve_production_grade():
    """L'agent résout une tâche avec un skill + quality_check = production-grade."""
    agent = ExpertAgentWithSkills(domain="development")
    result = agent.solve("code review task")
    assert "result" in result and result["quality"] == "production-grade"
    assert len(result["best_practices"]) >= 2
    assert "prompt" in result


def test_expert_agent_uses_domain_skill():
    """L'agent en cybersécurité utilise le skill security_audit."""
    reg = extended_production_skills()
    agent = ExpertAgentWithSkills(domain="cybersecurity", registry=reg)
    result = agent.solve("audit target", skill_name="security_audit")
    assert result["skill"] == "security_audit"
    assert "OWASP" in str(result["best_practices"])


def test_extended_skills_cover_user_repos():
    """Skills étendus couvrent les domaines des repos cités (security, UX, game, recherche)."""
    reg = extended_production_skills()
    domains = reg.domains()
    assert "cybersecurity" in domains
    assert "ux_design" in domains
    assert "game_dev" in domains
    assert "research" in domains
    assert len(reg.names()) >= 11          # 4 base + 7 étendus (incluant pentest/marketing/level)


def test_expert_agent_creates_skill_when_missing():
    """Skill manquant -> le modèle en CRÉE un nouveau (adaptatif, pas erreur)."""
    agent = ExpertAgentWithSkills(domain="development")
    result = agent.solve("besoin_spécial", skill_name="inexistant")
    assert "skill" in result and "result" in result    # skill créé (pas erreur)
    assert "auto-créé" in result["result"]
