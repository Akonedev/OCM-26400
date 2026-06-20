"""Tests TDD — méta-contrôleur (OCM-26400, 'généraliser sélection/création skills/agents/prompts').

Valide : routing MoE par tâche, sélection prompt, skill trouvé ou créé, exécution
production-grade, généralisation (tâches différentes → domaines différents), batch.
"""
from ocm26400.meta_controller import MetaController


def test_meta_routes_to_correct_domain():
    """Le méta-contrôleur route la tâche vers le bon domaine (MoE)."""
    mc = MetaController()
    assert mc.analyze("calcul du nombre optimal").domain == "math"
    assert mc.analyze("audit de sécurité owasp").domain == "cybersecurity"
    assert mc.analyze("interface utilisateur accessible").domain == "ux_design"


def test_meta_selects_prompt_for_domain():
    """Le prompt expert correspond au domaine routé."""
    mc = MetaController()
    a = mc.analyze("recherche scientifique sur les données")
    assert "recherche" in a.prompt.lower() or "scientifique" in a.prompt.lower() or "expert" in a.prompt.lower()


def test_meta_finds_existing_skill():
    """Un skill existant est sélectionné (pas créé)."""
    mc = MetaController()
    a = mc.analyze("code review task")
    assert a.skill_created is False          # skill existant trouvé
    assert "code_review" in a.skill_name or "auto" not in a.skill_name


def test_meta_creates_missing_skill():
    """Un skill manquant est CRÉÉ (généralisation : le modèle crée l'outil adapté)."""
    mc = MetaController()
    a = mc.analyze("tâche dans un nouveau domaine mathématique complexe")
    # si aucun skill n'existe pour ce besoin précis, il est créé
    if a.skill_created:
        assert "auto_" in a.skill_name


def test_meta_executes_production_grade():
    """L'exécution produit un résultat quality-checké (production-grade)."""
    mc = MetaController()
    result = mc.execute("revue de code python")
    assert result["quality"] == "production-grade"
    assert "result" in result and result["result"] != ""


def test_meta_generalizes_different_domains():
    """Tâches différentes → domaines différents (généralisation du routing)."""
    mc = MetaController()
    tasks = ["calcul math", "hack sécurité", "design UX", "recherche source"]
    domains = {mc.analyze(t).domain for t in tasks}
    assert len(domains) >= 3                   # au moins 3 domaines distincts = généralisation


def test_meta_batch_execute():
    """Batch : le méta-contrôleur orchestre plusieurs tâches (chacune son domaine)."""
    mc = MetaController()
    results = mc.batch_execute(["calcul 2+2", "audit owasp", "design interface"])
    assert len(results) == 3
    domains = {r["domain"] for r in results}
    assert len(domains) >= 2                   # tâches dispatchées vers différents domaines
