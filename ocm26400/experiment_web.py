#!/usr/bin/env python3
"""
EXPÉRIENCE apprentissage depuis URLs RÉELLES (OCM-26400, cahier des charges).

Démontre le VRAI apprentissage depuis une URL : l'agent fetch une page web réelle
(WebFetchTool, HTTP urllib), l'apprend (stocke son contenu), et la retient (re-demande
-> récupéré sans re-fetch). C'est « je donne une URL au model, il lit la page et
l'apprend » + browser use / RAG web, avec de VRAIES requêtes réseau.

Utilise l'API REST Wikipedia (JSON propre : title + extract) pour des contenus fiables.
"""
import json, time
from ocm26400.web_tools import WebFetchTool, URLMemory

# URLs réelles (API REST Wikipedia -> JSON {title, extract})
URLS = [
    ("https://en.wikipedia.org/api/rest_v1/page/summary/Paris", "Paris"),
    ("https://en.wikipedia.org/api/rest_v1/page/summary/Albert_Einstein", "Einstein"),
    ("https://en.wikipedia.org/api/rest_v1/page/summary/Pythagorean_theorem", "Pythagore"),
]


def main():
    tool = WebFetchTool(timeout=20, max_chars=600)
    mem = URLMemory(tool)
    print("OCM-26400 APPRENTISSAGE DEPUIS URLs RÉELLES (WebFetchTool, vrai HTTP)")
    learned = []
    t0 = time.time()
    for url, label in URLS:
        print(f"\n→ URL : {url}")
        knows_before = mem.knows(url)
        content = mem.learn(url)                 # VRAI fetch + apprentissage
        if content and not content.startswith("[fetch error"):
            snippet = content.replace("\n", " ")[:160]
            print(f" appris '{label}' ({len(content)} chars) : {snippet}...")
            learned.append({"url": url, "label": label,
                            "chars": len(content), "snippet": snippet,
                            "retained": mem.knows(url)})
        else:
            print(f"  fetch échoué/limité : {content}")
            learned.append({"url": url, "label": label, "error": content})

    # rétention : re-demande = récupéré (cache, pas de re-fetch réseau)
    print("\nRétention (re-demande -> récupéré sans re-fetch) :")
    for url, label in URLS:
        if mem.knows(url):
            c = mem.retrieve(url)
            print(f"  '{label}' : retenu ({len(c)} chars) -> {'OK' if c else 'vide'}")

    dt = time.time() - t0
    n_ok = sum(1 for x in learned if "chars" in x)
    verdict = "VALIDÉ" if n_ok >= 2 else "NON VALIDÉ (réseau ?)"
    print(f"\n{n_ok}/{len(URLS)} URLs réelles apprises. VERDICT (apprentissage depuis URL réel) : {verdict}")

    results = {
        "task": "apprentissage depuis URLs réelles (spec 'donner URL -> lire -> apprendre')",
        "n_urls": len(URLS), "n_learned": n_ok,
        "learned": learned, "verdict": verdict, "duration_s": round(dt, 1),
    }
    with open("ocm26400/web_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRésultats: ocm26400/web_results.json")
    return results


if __name__ == "__main__":
    main()
