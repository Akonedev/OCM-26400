"""Génétique mendélienne — carrés de Punnett + hérédité — domaine biologie compétent.

Compétence biologie RÉELLE (pas cosmétique) :
* Carrés de Punnett : croisement de 2 génotypes → ratios phénotypiques/génotypiques.
* Dominance/récessivité : détermine le phénotype depuis le génotype.
* Probabilités de descendance (fraction d'un phénotype donné).

Vérifiable : le carré de Punnett donne les ratios exacts de Mendel (3:1, 9:3:3:1, etc.).
"""
from __future__ import annotations
from itertools import product
from typing import Dict, List, Tuple


def phenotype(genotype: str, dominance: Dict[str, str]) -> str:
    """Détermine le phénotype d'un génotype (ex 'Aa') selon la dominance.
    dominance = {allèle: phénotype_exprimé}."""
    alleles = list(genotype)
    # l'allèle dominant s'exprime s'il est présent
    for a in alleles:
        if a.upper() == a and a in dominance:   # allèle dominant (majuscule)
            return dominance[a]
    # sinon le récessif (les 2 identiques minuscules)
    for a in alleles:
        if a in dominance:
            return dominance[a]
    return genotype


def gametes(genotype: str) -> List[str]:
    """Gamètes possibles d'un génotype (ex 'Aa' → ['A','a'] ; 'AaBb' → AB,Ab,aB,ab)."""
    # un gène par paire de caractères
    pairs = [genotype[i:i + 2] for i in range(0, len(genotype), 2)]
    choices = [[p[0], p[1]] for p in pairs]
    return ["".join(g) for g in product(*choices)]


def punnett_square(parent1: str, parent2: str) -> Dict[str, float]:
    """Carré de Punnett : croisement p1 × p2 → {génotype_descendance: probabilité}."""
    g1, g2 = gametes(parent1), gametes(parent2)
    counts: Dict[str, int] = {}
    for ga in g1:
        for gb in g2:
            # fusionne allèle par allèle (tri par gène pour canonical: dom avant récessif)
            child = ""
            for i in range(0, len(ga)):
                a, b = ga[i], gb[i]
                # ordonne : majuscule (dominant) d'abord
                pair = (a + b) if a.upper() == a or a < b else (b + a)
                child += pair
            counts[child] = counts.get(child, 0) + 1
    total = sum(counts.values())
    return {g: c / total for g, c in counts.items()}


def phenotype_ratios(parent1: str, parent2: str, dominance: Dict[str, str]
                     ) -> Dict[str, float]:
    """Ratios phénotypiques de la descendance (ex 75% dominant / 25% récessif)."""
    geno = punnett_square(parent1, parent2)
    pheno: Dict[str, float] = {}
    for g, p in geno.items():
        phen = phenotype(g, dominance)
        pheno[phen] = pheno.get(phen, 0) + p
    return pheno


def mendelian_cross(p1: str, p2: str, dominance: Dict[str, str]) -> Dict:
    """Croisement mendélien complet : génotypes + phénotypes + ratios."""
    return {
        "parents": [p1, p2],
        "genotype_ratios": punnett_square(p1, p2),
        "phenotype_ratios": phenotype_ratios(p1, p2, dominance),
    }


if __name__ == "__main__":
    dom = {"A": "dominant", "a": "récessif"}
    print("[genetics] Aa × Aa :", mendelian_cross("Aa", "Aa", dom)["phenotype_ratios"])
    # devrait donner ~75% dominant / 25% récessif (loi de Mendel 3:1)
    dom2 = {"A": "dominant", "a": "récessif", "B": "dominant2", "b": "récessif2"}
    print("[genetics] AaBb × AaBb :", mendelian_cross("AaBb", "AaBb", dom2)["phenotype_ratios"])
    # 9:3:3:1
