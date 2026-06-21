"""Scanner OSINT — extraction/analyse d'indicateurs — réfute audit OSINT (MOYENNE).

EX-T12, B199. OSINT (Open Source Intelligence) : reconnaissance depuis sources ouvertes.
Ici, analyse d'un texte/document pour extraire et évaluer des INDICATEURS :
* Emails, domaines, URLs, IPs, numéros de téléphone.
* Credentials/leaks potentiels (motifs API_KEY=, password=, tokens).
* Risques (credentials en clair = fuite).

Vérifiable : extraction exacte par regex + validation. Honnête : OSINT pattern-based
sur texte fourni (pas de requêtes réseau vers des APIs externes — l'audit note que ça
nécessiterait des sources externes ; on couvre l'analyse d'indicateurs).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class OSINTReport:
    emails: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    ips: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    leaked_credentials: List[str] = field(default_factory=list)
    risk_level: str = "LOW"
    findings: int = 0

    def to_dict(self) -> dict:
        return {**self.__dict__, "findings": sum(len(v) for k, v in self.__dict__.items()
                                                  if isinstance(v, list) and k != "findings")}


_EMAIL = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)
_URL = re.compile(r"https?://[^\s<>\"')]+", re.I)
_DOMAIN = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b", re.I)
_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE = re.compile(r"\b(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,3}\)?[\s.-]?){2,4}\d{2,4}\b")
# credentials fuités : KEY = "value" (longue chaîne)
_CRED = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwd|token|aws[_-]?(access|secret)[_-]?key)"
    r"\s*[=:]\s*['\"]?[A-Za-z0-9+/=_-]{12,}")


def is_valid_ip(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def is_internal_ip(ip: str) -> bool:
    """IP privée/réservée (10.x, 172.16-31.x, 192.168.x, 127.x)."""
    if not is_valid_ip(ip):
        return False
    a, b, _, _ = (int(x) for x in ip.split("."))
    return (a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)
            or a == 127)


def scan(text: str) -> OSINTReport:
    """Scanne un texte → rapport OSINT (indicateurs + credentials + risque)."""
    emails = sorted(set(_EMAIL.findall(text)))
    urls = sorted(set(_URL.findall(text)))
    # domaines : hors emails/urls déjà capturés
    domains = sorted({d for d in _DOMAIN.findall(text)
                      if d not in "".join(emails + urls) and "@" not in d})
    ips = sorted({ip for ip in _IP.findall(text) if is_valid_ip(ip)})
    phones = sorted(set(_PHONE.findall(text)))
    creds = sorted(set(_CRED.findall(text)))

    risk = "LOW"
    if creds:
        risk = "CRITICAL"        # credentials en clair = fuite critique
    elif emails or ips:
        risk = "MEDIUM"
    rep = OSINTReport(emails=emails, domains=domains, urls=urls, ips=ips,
                      phones=phones, leaked_credentials=creds, risk_level=risk)
    rep.findings = sum(len(v) for k, v in rep.__dict__.items()
                       if isinstance(v, list) and k != "findings")
    return rep


if __name__ == "__main__":
    sample = """
    Contact: admin@exemple.com, support@corp.fr
    Site: https://www.exemple.com et http://10.0.0.5:8080
    Serveur DNS: 192.168.1.1, public: 8.8.8.8
    Tel: +33 6 12 34 56 78
    Config: API_KEY = "sk-1234567890abcdefXYZ" et password = "secretpass123"
    """
    rep = scan(sample)
    for k, v in rep.to_dict().items():
        print(f"  {k}: {v}")
