"""Cryptographie — primitives réelles — vague 3.

* César / Vigenère (chiffrement symétrique classique).
* RSA (asymétrique : clé publique/privée, chiffrer/déchiffrer avec modexp).
* Hashing (démonstration, pas de sécurité réelle — utiliser hashlib en prod).
* Attaque fréquentielle (casser César).
Vérifiable : chiffrer puis déchiffrer redonne le clair.
"""
from __future__ import annotations
from typing import Tuple
from .symbolic_math import modexp, gcd, is_prime


def caesar_encrypt(text: str, shift: int) -> str:
    out = []
    for c in text:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            out.append(c)
    return "".join(out)


def caesar_decrypt(text: str, shift: int) -> str:
    return caesar_encrypt(text, -shift)


def vigenere_encrypt(text: str, key: str) -> str:
    out, k = [], [ord(c.upper()) - ord("A") for c in key if c.isalpha()]
    if not k:
        return text
    ki = 0
    for c in text:
        if c.isalpha():
            base = ord("A") if c.isupper() else ord("a")
            out.append(chr((ord(c) - base + k[ki % len(k)]) % 26 + base))
            ki += 1
        else:
            out.append(c)
    return "".join(out)


def vigenere_decrypt(text: str, key: str) -> str:
    inv_key = "".join(chr((26 - (ord(c.upper()) - ord("A"))) % 26 + ord("A"))
                      for c in key if c.isalpha())
    return vigenere_encrypt(text, inv_key)


def rsa_keygen(p: int, q: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Génère clés RSA (publique, privée) depuis 2 premiers p, q."""
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537                       # exposant public standard
    if gcd(e, phi) != 1:
        e = 3
    d = _modinv(e, phi)             # exposant privé
    return (n, e), (n, d)


def rsa_encrypt(message: int, pub: Tuple[int, int]) -> int:
    n, e = pub
    return modexp(message, e, n)


def rsa_decrypt(cipher: int, priv: Tuple[int, int]) -> int:
    n, d = priv
    return modexp(cipher, d, n)


def _modinv(a: int, m: int) -> int:
    """Inverse modulaire (étendu d'Euclide)."""
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("pas d'inverse modulaire")
    return x % m


def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = _egcd(b % a, a)
    return g, y - (b // a) * x, x


def frequency_attack(ciphertext: str) -> list:
    """Casse César : essaie les 26 décalages, score par fréquence du 'E' (FR/EN)."""
    results = []
    for shift in range(26):
        dec = caesar_decrypt(ciphertext, shift)
        score = dec.count("e") + dec.count("E") + dec.count("a") + dec.count("s")
        results.append((shift, dec, score))
    return sorted(results, key=lambda x: -x[2])


if __name__ == "__main__":
    msg = "BONJOUR LE MONDE"
    enc = caesar_encrypt(msg, 3)
    print(f"[crypto] César : {msg} → {enc} → {caesar_decrypt(enc, 3)}")
    vig = vigenere_encrypt("ATTAQUEZ", "CLE")
    print(f"[crypto] Vigenère : ATTAQUEZ → {vig} → {vigenere_decrypt(vig, 'CLE')}")
    pub, priv = rsa_keygen(61, 53)
    c = rsa_encrypt(42, pub)
    print(f"[crypto] RSA : 42 → {c} → {rsa_decrypt(c, priv)}")
    print(f"[crypto] casser '{enc}' : meilleur={frequency_attack(enc)[0][:2]}")
