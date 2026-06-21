"""Tests scanner OSINT (OCM-26400) — audit OSINT."""
from ocm26400.osint_scanner import scan, is_valid_ip, is_internal_ip, OSINTReport


def test_extract_emails():
    r = scan("contact: a@b.com et x@y.fr")
    assert "a@b.com" in r.emails and "x@y.fr" in r.emails


def test_extract_urls():
    r = scan("voir https://site.com/page et http://autre.org")
    assert any("site.com" in u for u in r.urls)


def test_extract_ips():
    r = scan("serveurs 192.168.1.1 et 8.8.8.8")
    assert "192.168.1.1" in r.ips and "8.8.8.8" in r.ips


def test_detect_credentials_critical():
    """Credentials en clair → risque CRITICAL."""
    r = scan('API_KEY = "sk-1234567890abcdef"')
    assert len(r.leaked_credentials) >= 1
    assert r.risk_level == "CRITICAL"


def test_no_credential_low_risk():
    r = scan("rien d'intéressant ici")
    assert r.risk_level == "LOW"
    assert r.findings == 0


def test_email_gives_medium_risk():
    r = scan("contact: a@b.com")
    assert r.risk_level == "MEDIUM"


def test_ip_validation():
    assert is_valid_ip("192.168.1.1") is True
    assert is_valid_ip("999.1.1.1") is False
    assert is_valid_ip("1.2.3") is False


def test_internal_ip_detection():
    assert is_internal_ip("192.168.1.1") is True
    assert is_internal_ip("10.0.0.5") is True
    assert is_internal_ip("127.0.0.1") is True
    assert is_internal_ip("8.8.8.8") is False
