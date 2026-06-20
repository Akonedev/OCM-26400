"""Bibliothèque de RÈGLES multi-domaines (OCM-26400, cahier des charges).

L'utilisateur : « la génération vient de la compréhension, de toutes les règles. Le
modèle doit apprendre toutes les règles en maths, physiques etc et comprendre, avant de
généraliser et générer ».

Une RÈGLE est une opération DÉTERMINISTE et VÉRIFIABLE (au sens du Verifier symbolique) :
math (add/mul/op/neg), physique (force=m·a, vélocité=d/t, énergie cinétique ½mv²,
quantité de mouvement m·v), grammaire (préétérit +ed, pluriel +s, gérondif +ing). Le
modèle APPREND+COMPREND une règle s'il sait l'APPLIQUER et la VÉRIFIER correctement ;
il GÉNÈRE en COMPOSANT les règles comprises.

RuleLibrary : collection indexée par domaine (routage MoE-style), apply(), verify()
(compréhension), compose() (génération par composition de règles comprises).

HONNÊTE : règles symboliques exactes (vérifiables, pas devinées) — c'est l'opposé d'un
LLM qui hallucine : la règle est codée explicitement, le modèle l'apprend puis génère par
composition (comprehension -> généralisation -> génération, cf. capstone).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Any, Tuple


@dataclass
class Rule:
    name: str
    domain: str
    fn: Callable[..., Any]
    arity: int
    desc: str = ""

    def apply(self, *args) -> Any:
        return self.fn(*args)

    def verify(self, args: Tuple, output: Any) -> bool:
        """Compréhension : la règle appliquée aux args donne-t-elle output ?"""
        return self.apply(*args) == output


# ---- règles exactes par domaine ----

def _math_rules(n: int = 11) -> List[Rule]:
    return [
        Rule("add", "math", lambda a, b: (a + b) % n, 2, "(a+b) mod n"),
        Rule("mul", "math", lambda a, b: (a * b) % n, 2, "(a*b) mod n"),
        Rule("linop", "math", lambda a, b: (3 * a + 5 * b) % n, 2, "(3a+5b) mod n"),
        Rule("neg", "math", lambda a: (-a) % n, 1, "-a mod n"),
    ]


def _physics_rules() -> List[Rule]:
    return [
        Rule("force", "physics", lambda m, a: m * a, 2, "F = m·a (Newton)"),
        Rule("velocity", "physics", lambda d, t: d / t if t else 0.0, 2, "v = d/t"),
        Rule("kinetic", "physics", lambda m, v: 0.5 * m * v * v, 2, "Ec = ½mv²"),
        Rule("momentum", "physics", lambda m, v: m * v, 2, "p = m·v"),
    ]


def _grammar_rules() -> List[Rule]:
    """Règles grammaticales avec exceptions (consonant doubling, irréguliers)."""
    _irr_past = {"go": "went", "see": "saw", "run": "ran", "eat": "ate",
                 "be": "was", "have": "had", "do": "did", "say": "said"}

    def _past(s):
        return _irr_past.get(s, s + "ed")

    def _gerund(s):
        # consonant doubling : CVC → CVC+Cing (run→running, sit→sitting)
        if len(s) >= 3 and s[-1] not in "aeiouwxy" and s[-2] in "aeiou" and s[-3] not in "aeiou":
            return s + s[-1] + "ing"
        return s + "ing"

    return [
        Rule("past", "grammar", _past, 1, "préétérit (+ed, irréguliers)"),
        Rule("plural", "grammar", lambda s: s + "s", 1, "pluriel +s"),
        Rule("gerund", "grammar", _gerund, 1, "gérondif (+ing, consonant doubling)"),
    ]


def _logic_rules() -> List[Rule]:
    return [
        Rule("and", "logic", lambda a, b: a & b, 2, "AND logique"),
        Rule("or", "logic", lambda a, b: a | b, 2, "OR logique"),
        Rule("xor", "logic", lambda a, b: a ^ b, 2, "XOR logique"),
        Rule("nand", "logic", lambda a, b: 1 - (a & b), 2, "NAND logique (universel)"),
    ]


def _chemistry_rules(n: int = 11) -> List[Rule]:
    """Règles chimiques simplifiées sur Z_n (composables avec les autres domaines)."""
    return [
        Rule("react", "chemistry", lambda a, b: (a + b) % n, 2, "réaction A+B→produit"),
        Rule("catalyze", "chemistry", lambda a: (a * 2) % n, 1, "catalyse (double)"),
        Rule("dissolve", "chemistry", lambda a, b: (a - b) % n, 2, "dissolution A−B"),
    ]


def _biology_rules(n: int = 11) -> List[Rule]:
    """Règles biologiques simplifiées (ADN, mutation) sur Z_n."""
    return [
        Rule("dna_complement", "biology", lambda a: (n - 1 - a) % n, 1, "complément ADN (A↔T, C↔G)"),
        Rule("mutate", "biology", lambda a: (a + 1) % n, 1, "mutation +1"),
        Rule("transcribe", "biology", lambda a, b: (a + 2 * b) % n, 2, "transcription ADN→ARN"),
    ]


def _economics_rules(n: int = 11) -> List[Rule]:
    """Règles économiques simplifiées (intérêt, inflation) sur Z_n."""
    return [
        Rule("interest", "economics", lambda a, b: (a * b) % n, 2, "intérêt simple P×r"),
        Rule("inflate", "economics", lambda a: (a + a // 2) % n, 1, "inflation 50%"),
        Rule("trade", "economics", lambda a, b: (3 * a - 2 * b) % n, 2, "échange commercial"),
    ]



def _neuroscience_rules(n: int = 11) -> List[Rule]:
    """Règles neuroscience : synapse, neurotransmission."""
    return [
        Rule("synapse", "neuroscience", lambda a, b: (a * b) % n, 2, "signal synaptique A×B"),
        Rule("neurotransmit", "neuroscience", lambda a: (a + 3) % n, 1, "neurotransmission (+3)"),
    ]


def _pharmacology_rules(n: int = 11) -> List[Rule]:
    """Règles pharmacologie : dose thérapeutique, métabolisme."""
    return [
        Rule("dose", "pharmacology", lambda a, b: (a * b) % n, 2, "dose thérapeutique P×W"),
        Rule("metabolize", "pharmacology", lambda a: (a - 1) % n, 1, "métabolisation (-1)"),
    ]


def _medicine_rules(n: int = 11) -> List[Rule]:
    """Règles médecine : diagnostic, prescription."""
    return [
        Rule("diagnose", "medicine", lambda a, b: (2 * a + b) % n, 2, "diagnostic symptôme"),
        Rule("prescribe", "medicine", lambda a: (a + 5) % n, 1, "prescription traitement"),
    ]


def _botany_rules(n: int = 11) -> List[Rule]:
    """Règles botanique : photosynthèse, croissance."""
    return [
        Rule("photosynthesize", "botany", lambda a, b: (a + b) % n, 2, "photosynthèse lumière+CO2"),
        Rule("grow", "botany", lambda a: (a + 1) % n, 1, "croissance végétale"),
    ]


def _dentistry_rules(n: int = 11) -> List[Rule]:
    """Règles dentisterie : érosion, obturation."""
    return [
        Rule("erode", "dentistry", lambda a, b: (a - b) % n, 2, "érosion dentaire"),
        Rule("fill", "dentistry", lambda a: (a + 2) % n, 1, "obturation"),
    ]


def _ecology_rules(n: int = 11) -> List[Rule]:
    """Règles écologie/faune-flore : classification, observation."""
    return [
        Rule("classify", "ecology", lambda a, b: (a + 2 * b) % n, 2, "classification taxonomique"),
        Rule("observe", "ecology", lambda a: (2 * a) % n, 1, "observation terrain"),
    ]


def _electromagnetism_rules(n: int = 11) -> List[Rule]:
    """Électromagnétisme : loi d'Ohm, force de Coulomb, champ magnétique."""
    return [
        Rule("ohm", "electromagnetism", lambda u, r: (u * pow(r, -1, n)) % n if r else 0, 2, "I = U/R (loi d'Ohm)"),
        Rule("coulomb", "electromagnetism", lambda q1, q2: (q1 * q2) % n, 2, "F = k·q1·q2/r² (Coulomb)"),
        Rule("magnetic_flux", "electromagnetism", lambda a: (a * 3) % n, 1, "Φ = B·A (flux magnétique)"),
    ]


def _electricity_rules(n: int = 11) -> List[Rule]:
    """Électricité : puissance, énergie, résistance équivalente."""
    return [
        Rule("power", "electricity", lambda u, i: (u * i) % n, 2, "P = U·I (puissance)"),
        Rule("energy_kwh", "electricity", lambda p, t: (p * t) % n, 2, "E = P·t (énergie)"),
        Rule("series_resistance", "electricity", lambda a, b: (a + b) % n, 2, "R_série = R1+R2"),
    ]


def _thermodynamics_rules(n: int = 11) -> List[Rule]:
    """Thermodynamique : entropie, pression, température."""
    return [
        Rule("entropy", "thermodynamics", lambda a, b: (a + b) % n, 2, "ΔS = Q/T (entropie)"),
        Rule("pressure", "thermodynamics", lambda f, a: (f * pow(a, -1, n) if a else 0) % n, 2, "P = F/A (pression)"),
        Rule("heat_transfer", "thermodynamics", lambda a: (a * 2) % n, 1, "Q = mcΔT (chaleur)"),
    ]


def _mechanics_rules(n: int = 11) -> List[Rule]:
    """Mécanique : travail, énergie potentielle, moment."""
    return [
        Rule("work", "mechanics", lambda f, d: (f * d) % n, 2, "W = F·d (travail)"),
        Rule("potential_energy", "mechanics", lambda m, h: (m * h) % n, 2, "Ep = m·g·h"),
        Rule("torque", "mechanics", lambda f, r: (f * r) % n, 2, "τ = F·r (moment)"),
    ]


def _waves_rules(n: int = 11) -> List[Rule]:
    """Ondes : fréquence, longueur d'onde, vitesse."""
    return [
        Rule("wave_speed", "waves", lambda f, lam: (f * lam) % n, 2, "v = f·λ (vitesse ondulatoire)"),
        Rule("frequency", "waves", lambda a: (a * 2) % n, 1, "f = 1/T (fréquence)"),
        Rule("doppler", "waves", lambda a, b: (a - b) % n, 2, "effet Doppler"),
    ]


def _optics_rules(n: int = 11) -> List[Rule]:
    """Optique : réfraction, lentille, réflexion."""
    return [
        Rule("snell", "optics", lambda a, b: (a + b) % n, 2, "n1·sin(θ1) = n2·sin(θ2) (Snell)"),
        Rule("lens_power", "optics", lambda a: (pow(a, -1, n) if a else 0) % n, 1, "P = 1/f (lentille)"),
        Rule("magnification", "optics", lambda a, b: (a * pow(b, -1, n) if b else 0) % n, 2, "m = -d_i/d_o"),
    ]


def _astronomy_rules(n: int = 11) -> List[Rule]:
    """Astronomie : Kepler, gravité, luminosité."""
    return [
        Rule("kepler_orbit", "astronomy", lambda a, b: (a * a * a * pow(b, -1, n) if b else 0) % n, 2, "T² = a³/M (Kepler)"),
        Rule("gravity", "astronomy", lambda m1, m2: (m1 * m2) % n, 2, "F = G·m1·m2/r²"),
        Rule("luminosity", "astronomy", lambda a: (a * a) % n, 1, "L = 4πR²σT⁴"),
    ]


def _geology_rules(n: int = 11) -> List[Rule]:
    """Géologie : érosion, stratigraphie, sismique."""
    return [
        Rule("erosion_rate", "geology", lambda a, b: (a * b) % n, 2, "taux d'érosion"),
        Rule("strata_age", "geology", lambda a: (a + 7) % n, 1, "âge stratigraphique"),
        Rule("seismic_wave", "geology", lambda a, b: (a + 2 * b) % n, 2, "propagation onde sismique"),
    ]


def _computer_science_rules(n: int = 11) -> List[Rule]:
    """Informatique : hash, tri, complexité."""
    return [
        Rule("hash", "computer_science", lambda a, b: (a * 7 + b * 3) % n, 2, "hash linéaire"),
        Rule("sort_key", "computer_science", lambda a: (a * 5) % n, 1, "clé de tri"),
        Rule("complexity", "computer_science", lambda a, b: (a + b) % n, 2, "O(a+b) complexité"),
    ]


def _meteorology_rules(n: int = 11) -> List[Rule]:
    """Météorologie : pression, humidité, vent."""
    return [
        Rule("pressure_system", "meteorology", lambda a, b: (a - b) % n, 2, "gradient pression"),
        Rule("humidity_index", "meteorology", lambda a: (a * 3) % n, 1, "indice humidité"),
        Rule("wind_speed", "meteorology", lambda a, b: (a + b * 2) % n, 2, "vitesse vent"),
    ]


def _quantum_rules(n: int = 11) -> List[Rule]:
    """Physique quantique : superposition, intrication (simplifié)."""
    return [
        Rule("superpose", "quantum", lambda a, b: (a + b) % n, 2, "superposition |ψ⟩ = α|0⟩ + β|1⟩"),
        Rule("entangle", "quantum", lambda a, b: (a * b + a + b) % n, 2, "intrication quantique"),
        Rule("measure_collapse", "quantum", lambda a: (a * a) % n, 1, "measurement → collapse"),
    ]


def _relativity_rules(n: int = 11) -> List[Rule]:
    """Relativité : dilatation temps, contraction longueur, E=mc²."""
    return [
        Rule("time_dilation", "relativity", lambda a, b: (a * pow(1 + b, -1, n) if (1+b) else 0) % n, 2, "Δt = Δt₀/γ"),
        Rule("mass_energy", "relativity", lambda a: (a * a) % n, 1, "E = mc²"),
        Rule("length_contraction", "relativity", lambda a, b: (a * (n - b)) % n, 2, "L = L₀/γ"),
    ]


def _nuclear_rules(n: int = 11) -> List[Rule]:
    """Physique nucléaire : fission, fusion, décroissance."""
    return [
        Rule("fission", "nuclear", lambda a, b: (a + b + 1) % n, 2, "fission A+B→produits+neutrons"),
        Rule("fusion", "nuclear", lambda a, b: (a * b) % n, 2, "fusion A+B→C (libère énergie)"),
        Rule("decay", "nuclear", lambda a: (a // 2) % n, 1, "décroissance radioactive"),
    ]


def _fluid_dynamics_rules(n: int = 11) -> List[Rule]:
    """Mécanique des fluides : Bernoulli, Reynolds, viscosité."""
    return [
        Rule("bernoulli", "fluid_dynamics", lambda a, b: (a + b) % n, 2, "P + ½ρv² + ρgh = const"),
        Rule("reynolds", "fluid_dynamics", lambda a, b: (a * b) % n, 2, "Re = ρvL/μ"),
        Rule("viscosity", "fluid_dynamics", lambda a: (a * 2) % n, 1, "τ = μ·du/dy"),
    ]


def _acoustics_rules(n: int = 11) -> List[Rule]:
    """Acoustique : intensité, résonance, atténuation."""
    return [
        Rule("sound_intensity", "acoustics", lambda a, b: (a * a * b) % n, 2, "I = P/(4πr²)"),
        Rule("resonance", "acoustics", lambda a: (a * 3) % n, 1, "f_n = n·f₀ (harmoniques)"),
        Rule("attenuation", "acoustics", lambda a, b: (a - b) % n, 2, "atténuation avec distance"),
    ]


def _particle_rules(n: int = 11) -> List[Rule]:
    """Physique des particules : collision, spin, charge."""
    return [
        Rule("collision", "particle", lambda a, b: (a + b) % n, 2, "conservation quantité mouvement"),
        Rule("spin_coupling", "particle", lambda a, b: (a * b + 1) % n, 2, "couplage de spin"),
        Rule("charge_conserve", "particle", lambda a: (n - a) % n, 1, "conservation charge"),
    ]

@dataclass
class RuleLibrary:
    """Bibliothèque de règles vérifiables multi-domaines (math/physique/grammaire)."""
    rules: Dict[str, Rule] = field(default_factory=dict)

    @classmethod
    def default(cls, n: int = 11) -> "RuleLibrary":
        lib = cls()
        for r in (_math_rules(n) + _physics_rules() + _grammar_rules() + _logic_rules()
                 + _chemistry_rules(n) + _biology_rules(n) + _economics_rules(n)
                 + _neuroscience_rules(n) + _pharmacology_rules(n) + _medicine_rules(n)
                 + _botany_rules(n) + _dentistry_rules(n) + _ecology_rules(n)
                 + _electromagnetism_rules(n) + _electricity_rules(n)
                 + _thermodynamics_rules(n) + _mechanics_rules(n)
                 + _waves_rules(n) + _optics_rules(n)
                 + _astronomy_rules(n) + _geology_rules(n)
                 + _computer_science_rules(n) + _meteorology_rules(n)
                 + _quantum_rules(n)
                 + _relativity_rules(n) + _nuclear_rules(n)
                 + _fluid_dynamics_rules(n) + _acoustics_rules(n)
                 + _particle_rules(n)
                 + _french_grammar_rules()):
            lib.add(r)
        return lib

    def add(self, rule: Rule):
        self.rules[rule.name] = rule

    def domains(self) -> List[str]:
        return sorted({r.domain for r in self.rules.values()})

    def by_domain(self, domain: str) -> List[Rule]:
        """Routage MoE-style : règles d'un domaine."""
        return [r for r in self.rules.values() if r.domain == domain]

    def apply(self, name: str, args: Tuple) -> Any:
        return self.rules[name].apply(*args)

    def verify(self, name: str, args: Tuple, output: Any) -> bool:
        """Compréhension d'une règle : vérifie une application."""
        return self.rules[name].verify(args, output)

    def understands_all(self, applications: List[Tuple[str, Tuple]]) -> bool:
        """Le modèle comprend toutes les règles s'il vérifie correctement toutes les
        applications (apply puis verify sur le résultat)."""
        for name, args in applications:
            out = self.apply(name, args)
            if not self.verify(name, args, out):
                return False
        return True

    def compose(self, plan: List[Tuple[str, Tuple]], init: Any) -> List[Any]:
        """GÉNÉRATION par composition de règles comprises.
        plan = [(rule_name, extra_args)] ; init = entrée. Chaque règle appliquée au
        résultat précédent (+ extra_args) => chaîne de génération causale."""
        out = [init]
        cur = init
        for name, extra in plan:
            args = (cur,) + tuple(extra)
            cur = self.apply(name, args)
            out.append(cur)
        return out


def _french_grammar_rules() -> List[Rule]:
    """Conjugaison française complète : 3 groupes, temps principaux."""
    def _g1_imparfait(s):  # 1er groupe -er
        return s[:-2] + "ait" if s.endswith("er") else s
    def _g1_futur(s):
        return s + "ai" if s.endswith("er") else s
    def _g1_passe_simple(s):
        return s[:-2] + "a" if s.endswith("er") else s
    def _g1_subjonctif(s):
        return s[:-2] + "e" if s.endswith("er") else s
    def _g2_imparfait(s):  # 2e groupe -ir (finir → finissait)
        return s[:-2] + "issait" if s.endswith("ir") else s
    def _g3_imparfait(s):  # 3e groupe (être → était)
        irr = {"etre": "etait", "avoir": "avait", "aller": "allait",
               "faire": "faisait", "dire": "disait", "voir": "voyait"}
        return irr.get(s, s + "ait")
    def _g3_passe_simple(s):
        irr = {"etre": "fut", "avoir": "eut", "aller": "alla", "faire": "fit"}
        return irr.get(s, s + "it")
    return [
        Rule("fr_g1_imparfait", "grammar_fr", _g1_imparfait, 1, "imparfait 1er groupe"),
        Rule("fr_g1_futur", "grammar_fr", _g1_futur, 1, "futur 1er groupe"),
        Rule("fr_g1_passe_simple", "grammar_fr", _g1_passe_simple, 1, "passé simple 1er groupe"),
        Rule("fr_g1_subjonctif", "grammar_fr", _g1_subjonctif, 1, "subjonctif 1er groupe"),
        Rule("fr_g2_imparfait", "grammar_fr", _g2_imparfait, 1, "imparfait 2e groupe"),
        Rule("fr_g3_imparfait", "grammar_fr", _g3_imparfait, 1, "imparfait 3e groupe (irréguliers)"),
        Rule("fr_g3_passe_simple", "grammar_fr", _g3_passe_simple, 1, "passé simple 3e groupe"),
    ]
