"""Tests mémoire procédurale (OCM-26400) — audit H12."""
from ocm26400.procedural_memory import Procedure, ProceduralMemory, default_procedures


def test_procedure_steps():
    p = Procedure(name="x", steps=["a", "b", "c"])
    assert p.n_steps() == 3


def test_learn_and_get():
    pm = ProceduralMemory()
    pm.learn(Procedure(name="f", steps=["1", "2"]))
    assert pm.size() == 1
    assert pm.get("f") is not None
    assert pm.get("inconnu") is None


def test_how_to_finds_matching():
    pm = default_procedures()
    p = pm.how_to("comment faire du thé ?")
    assert p is not None and p.name == "faire_du_thé"


def test_how_to_abstains_on_unknown():
    pm = default_procedures()
    assert pm.how_to("comment piloter un avion ?") is None   # abstention


def test_replay_dry_run():
    pm = default_procedures()
    out = pm.replay("trier_liste")
    assert out is not None and len(out) == 4       # 4 étapes
    assert all(s is None for _, s in out)          # dry-run (no executor)


def test_replay_with_executor():
    pm = default_procedures()
    out = pm.replay("trier_liste", executor=lambda s: len(s))
    assert all(isinstance(r, int) for _, r in out)


def test_generalize_creates_template():
    """Abstraction : procédure spécifique → template paramétré."""
    pm = ProceduralMemory()
    pm.learn(Procedure(name="faire_café", steps=["moudre grains", "verser eau"],
                       effects=["café prêt"]))
    tmpl = pm.generalize("faire_café", "faire_boisson_chaude", ["ingrédient"])
    assert tmpl is not None
    assert pm.get("faire_boisson_chaude") is not None


def test_default_procedures_count():
    pm = default_procedures()
    assert pm.size() >= 4
    assert "faire_du_thé" in pm.names()


def test_procedural_distinct_from_semantic():
    """La mémoire procédurale stocke du SAVOIR-FAIRE (étapes), pas des faits."""
    pm = default_procedures()
    proc = pm.get("résoudre_équation_2nd_degré")
    # c'est une séquence d'actions, pas un fait isolé
    assert proc.n_steps() >= 3
    assert any("calculer" in s or "identifier" in s for s in proc.steps)
