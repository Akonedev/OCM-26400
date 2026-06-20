"""Vérifieur morphologique multi-opérations (OCM-26400, spec cahier des charges).

Étend le Verifier de base (verifier.py) avec le DISPATCH PAR op_id : plusieurs
règles morphologiques (conjugaison : PAST/GERUND/THIRD, accord syntaxique) coexistent
dans un MÊME vérifieur, sélectionnées par op_id. C'est la pièce « conjugaison complète
+ règles syntaxiques vérifiables » du cahier des charges, et l'utilisation effective
du dispatch op_id (préparé par les contrats partagés mais non exploité jusqu'ici).

HONNÊTE / ANTI-FRANKENSTEIN :
* NE MODIFIE PAS la base Verifier.compose (le chemin compose_fn ignore toujours op_id,
  contract test_contracts.py préservé). MorphologyVerifier OVERRIDE compose(a,b,op_id)
  pour dispatcher — additive, pas de collage.
* Les règles sont des tables EXPLICITES (verbe×temps -> forme), codées déterministement.
  C'est l'opposé d'un LLM qui devine : la règle est vérifiable.

Id dispatch (verdict) :
    CONJUGATE_PAST   = 0   walk + PAST   -> walked
    CONJUGATE_GERUND = 1   walk + GERUND -> walking
    CONJUGATE_THIRD  = 2   walk + THIRD  -> walks
"""
from .verifier import Verifier, SymbolicDict

# op_id réservés (conjugaison)
CONJUGATE_PAST = 0
CONJUGATE_GERUND = 1
CONJUGATE_THIRD = 2


class MorphologyVerifier(Verifier):
    """Verifieur morphologique : compose(a, b, op_id) dispatch par op_id.

    rules: liste/dict op_id -> callable(a, b) -> forme_id. n_ops = len(rules).
    Override compose() pour utiliser op_id (la base l'ignore sur le chemin compose_fn).
    """

    def __init__(self, dictionary: SymbolicDict, rules, n_ops: int = None):
        if isinstance(rules, dict):
            rules_list = [rules[k] for k in sorted(rules.keys())]
        else:
            rules_list = list(rules)
        super().__init__(dictionary, compose_fn=None, n_ops=n_ops or len(rules_list))
        self.rules = rules_list

    def compose(self, a: int, b: int, op_id: int = 0) -> int:
        if 0 <= op_id < len(self.rules):
            return self.rules[op_id](a, b)
        return a  # no-op hors range (rétrocompatible)

    def is_valid_intermediate(self, a: int, b: int, m: int, op_id: int = 0) -> bool:
        return m == self.compose(a, b, op_id=op_id)
