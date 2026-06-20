"""Tests TDD — système de skills experts (OCM-26400).

Valide : ExpertSkill (execute + quality_check), SkillCreator (compose règles -> nouveau
skill), ExpertSkillRegistry (register/get/has/domains), production_skills.
"""
import pytest
from ocm26400.skills_system import (
    ExpertSkill, QualityError, SkillCreator, ExpertSkillRegistry, production_skills,
)
from ocm26400.rules import RuleLibrary


def test_expert_skill_execute_and_quality():
    """Le skill exécute ET vérifie les meilleures pratiques sur le résultat."""
    s = ExpertSkill("test", "test skill", ["non-vide"], fn=lambda x: f"résultat {x}")
    assert s.execute("hello") == "résultat hello"
    assert s.quality_check("ok") is True


def test_expert_skill_quality_error_on_bad_output():
    """Résultat vide/erreur -> QualityError (meilleure pratique non respectée)."""
    s = ExpertSkill("bad", "skill défaillant", ["non-vide"], fn=lambda: None)
    with pytest.raises(QualityError):
        s.execute()


def test_registry_register_get_has_domains():
    reg = ExpertSkillRegistry()
    reg.register(ExpertSkill("calc", "calcul expert", [], fn=lambda: "ok", domain="math"))
    assert reg.get("calc") is not None
    assert reg.has("calc") is True
    assert reg.has("inexistant") is False
    assert "math" in reg.domains()
    assert len(reg.by_domain("math")) == 1


def test_skill_creator_composes_rules():
    """SkillCreator compose les règles en un nouveau ExpertSkill."""
    lib = RuleLibrary.default(n=11)
    creator = SkillCreator(lib)
    skill = creator.create_skill("add_then_mul", [("add", (3,)), ("mul", (2,))], domain="math")
    result = skill.execute(4)                    # 4 -> add(4,3)=7 -> mul(7,2)=3 (mod 11)
    assert result == 3
    assert "add" in skill.description and "mul" in skill.description
    assert len(skill.best_practices) >= 2


def test_skill_creation_when_missing():
    """Le modèle crée un nouveau skill quand le registry n'en a pas pour le besoin."""
    reg = production_skills()
    assert not reg.has("composé_spécial")
    lib = RuleLibrary.default()
    creator = SkillCreator(lib)
    new_skill = creator.create_skill("composé_spécial", [("linop", (5,))])
    assert new_skill.quality_check(new_skill.execute(3)) is True


def test_production_skills_are_expert_grade():
    """Les skills production ont des meilleures pratiques (expert, pas amateur)."""
    reg = production_skills()
    assert len(reg.names()) >= 4
    for name in reg.names():
        skill = reg.get(name)
        assert len(skill.best_practices) >= 2     # chaque skill a des règles de qualité
        assert skill.description != ""
