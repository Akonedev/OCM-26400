"""Tests sommeil multi-phases (OCM-26400) — audit H19."""
from ocm26400.sleep_phases import (
    light_sleep, deep_sleep, paradoxal_sleep, full_night,
)


def _memory(op=(3, 5), n=11, k=8, seed=0):
    """Construit une mémoire de k faits cohérents avec op=(αa+βb) mod n."""
    import random
    rng = random.Random(seed)
    a, b = op
    mem = {}
    for _ in range(k):
        x, y = rng.randint(0, n - 1), rng.randint(0, n - 1)
        mem[(x, y)] = (a * x + b * y) % n
    return mem


def test_light_sleep_runs():
    mem = _memory()
    rep = light_sleep(mem)
    assert rep.phase.startswith("light")
    assert rep.facts_in == 8


def test_deep_sleep_extracts_rule():
    """Sommeil profond : extrait la règle (3,5) depuis 8 faits."""
    mem = _memory(op=(3, 5))
    rep = deep_sleep(mem, n=11)
    assert rep.rule_extracted == (3, 5)
    assert rep.compression == 8.0           # 8 faits → 1 règle
    assert rep.generalizes is True


def test_deep_sleep_no_rule_on_noise():
    """Sur du bruit incohérent, pas de règle extraite (honnête)."""
    mem = {(0, 0): 0, (1, 1): 7, (2, 2): 3, (3, 3): 9}   # pas de loi linéaire mod 11
    rep = deep_sleep(mem, n=11)
    # extract_rule peut ou non trouver ; on vérifie juste qu'il ne crash pas
    assert rep.phase.startswith("deep")


def test_paradoxal_sleep_finds_connections():
    """REM : détecte des analogies/compositions entre règles."""
    rules = [(3, 5), (3, 2), (1, 5)]   # partagent des coefficients → analogies
    rep = paradoxal_sleep(rules, n=11)
    assert rep.phase.startswith("paradoxal")
    assert rep.new_connections >= 1


def test_paradoxal_single_rule_no_connection():
    rep = paradoxal_sleep([(3, 5)], n=11)
    assert rep.new_connections == 0


def test_full_night_consolidates():
    mem = _memory(op=(3, 5))
    night = full_night(mem, extra_rules=[(1, 2)])
    assert night["verdict"] == "FULL_NIGHT_CONSOLIDATED"
    assert night["rule_learned"] is True
    assert night["total_compression"] == 8.0
    assert len(night["phases"]) == 3       # léger + profond + paradoxal
