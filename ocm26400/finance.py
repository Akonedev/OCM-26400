"""Finance RÉELLE — intérêts composés, annuités, prêts — domaine économie compétent.

Compétence économie/finance VÉRIFIABLE réelle :
* Intérêts composés : A = P(1+r/n)^(nt) ou A = P(1+r)^t.
* Annuité / paiement de prêt : PMT = P·r/(1−(1+r)^−n).
* Valeur actuelle / future. Taux effectif.
Tout est VÉRIFIABLE (formules financières standard).
"""
from __future__ import annotations
from typing import Dict


def compound_interest(principal: float, rate: float, time: int,
                      n: int = 1) -> float:
    """A = P(1 + r/n)^(nt). Capital final après 'time' périodes."""
    return round(principal * (1 + rate / n) ** (n * time), 2)


def continuous_compound(principal: float, rate: float, time: float) -> float:
    """A = P·e^(rt) (composition continue)."""
    import math
    return round(principal * math.exp(rate * time), 2)


def loan_payment(principal: float, annual_rate: float, n_payments: int) -> float:
    """Mensualité d'un prêt : PMT = P·r/(1−(1+r)^−n). r = taux périodique (annuel/12)."""
    r = annual_rate / 12
    if r == 0:
        return round(principal / n_payments, 2)
    return round(principal * r / (1 - (1 + r) ** -n_payments), 2)


def total_paid(principal: float, annual_rate: float, n_payments: int) -> Dict:
    """Coût total d'un prêt : mensualité × n + intérêts versés."""
    pmt = loan_payment(principal, annual_rate, n_payments)
    total = pmt * n_payments
    return {"monthly": pmt, "total": round(total, 2),
            "interest_paid": round(total - principal, 2)}


def present_value(future: float, rate: float, time: int) -> float:
    """Valeur actuelle : PV = FV / (1+r)^t."""
    return round(future / (1 + rate) ** time, 2)


def effective_rate(nominal: float, n: int = 12) -> float:
    """Taux effectif annuel : (1+r/n)^n − 1."""
    return round(((1 + nominal / n) ** n - 1) * 100, 4)


if __name__ == "__main__":
    print("[finance] 1000€ à 5% pendant 10 ans :", compound_interest(1000, 0.05, 10), "€")
    print("[finance] prêt 200000€ à 3% sur 25 ans :", total_paid(200000, 0.03, 300))
    print("[finance] valeur actuelle de 1000€ à 5% dans 10 ans :", present_value(1000, 0.05, 10), "€")
    print("[finance] taux effectif de 12% nominal :", effective_rate(0.12), "%")
