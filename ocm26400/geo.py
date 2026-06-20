"""Couche géospatial — cartes / globe / street / 3D (OCM-26400, cahier des charges).

Le cahier des charges demande : « lire/générer cartes style Google Earth/Street/OSM,
styles/layers/vues différentes ; vues globe, cartes, street immersive avec reconstruction
des volumes, tuiles, plans ; profondeur jusqu'au sol (3D volumes, bâtiments) ; tout
reconstruit selon le lieu sélectionné ; navigation souris/clavier ; recherches ;
afficher toutes les infos (geo, histoire, eco, santé, politiques...) ; compléter par les
agents (osiris, worldmonitor) ; NE PAS intégrer d'API provider externe ».

On implémente (sans API provider — format OSM / procédural déterministe) :

* Web Mercator RÉEL : lat/lon <-> tuiles (x,y,z). Standard des cartes tuilées (OSM/Google).
* GlobeView : projection lat/lon -> point 3D sur sphère (vue globe).
* InfoDB : infos MULTI-DOMAINES par lieu (liste complète : geo/histoire/eco/santé/
  politique/démo/climat/culture/éducation/transport/environnement/sécurité) — générées
  de façon déterministe depuis le lieu (procédural honnête, structure réelle, sans API).
* StreetView3D : reconstruction 3D volumique PROCÉDURALE (bâtiments/tuiles) selon le
  lieu sélectionné (seed déterministe -> volumes cohérents par lieu, distincts entre lieux).
* GeoMap : navigation (pan/zoom/center), recherche par nom, sélection lieu -> infos+3D,
  layers activables.

HONNÊTE : pas d'API provider externe. Math carte = réel (Web Mercator). Données infos et
volumes 3D = procéduraux déterministes (un vrai fichier OSM .pbf remplacerait la
génération — l'interface est identique). C'est le MOTEUR géospatial, pas un rendu GPU.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import torch

# liste complète des domaines d'info (complétable par agents)
INFO_DOMAINS = [
    "geography", "history", "economy", "health", "politics", "demographics",
    "climate", "culture", "education", "transport", "environment", "security",
]


@dataclass
class GeoPoint:
    lat: float
    lon: float


# ---------------- Web Mercator (réel) ----------------

def latlon_to_tile(lat: float, lon: float, zoom: int) -> Tuple[int, int]:
    """lat/lon -> (x,y) tuile Web Mercator (standard OSM/Google)."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    lat_rad = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * n)
    return x, y


def tile_to_latlon(x: int, y: int, zoom: int) -> Tuple[float, float]:
    n = 2 ** zoom
    lon = x / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return math.degrees(lat_rad), lon


def globe_to_xyz(lat: float, lon: float, r: float = 1.0) -> Tuple[float, float, float]:
    """Projection lat/lon -> point 3D sur sphère (vue globe)."""
    la, lo = math.radians(lat), math.radians(lon)
    return (r * math.cos(la) * math.cos(lo), r * math.sin(la), r * math.cos(la) * math.sin(lo))


# ---------------- Infos multi-domaines par lieu (procédural déterministe) ----------------

def _seed(lat: float, lon: float) -> int:
    return int.from_bytes(
        __import__("hashlib").md5(f"{round(lat, 3)},{round(lon, 3)}".encode()).digest()[:4], "big")


@dataclass
class InfoDB:
    """Infos par lieu, un domaine à la fois ou tous. Procédural déterministe (sans API)."""

    domains: List[str] = field(default_factory=lambda: list(INFO_DOMAINS))

    def info(self, lat: float, lon: float, domain: Optional[str] = None) -> Dict[str, str]:
        rng = random.Random(_seed(lat, lon))
        out = {}
        doms = [domain] if domain else self.domains
        for d in doms:
            # valeurs déterministes plausibles par domaine
            if d == "demographics":
                out[d] = f"population ~{rng.randint(5_000, 9_000_000):,} ({rng.randint(100, 800)}/km²)"
            elif d == "economy":
                out[d] = f"PIB/hab ~{rng.randint(3_000, 70_000):,} USD ; secteur {rng.choice(['agri', 'industrie', 'services'])}"
            elif d == "climate":
                out[d] = f"climat {rng.choice(['océanique', 'continental', 'tropical', 'aride', 'polaire'])} ; {rng.randint(-5, 35)}°C moy"
            elif d == "health":
                out[d] = f"espérance vie ~{rng.randint(60, 85)} ans ; {rng.randint(1, 8)} hôpitaux/100k"
            elif d == "geography":
                out[d] = f"lat {lat:.3f}, lon {lon:.3f} ; altitude ~{rng.randint(0, 3000)}m"
            else:
                out[d] = f"{d} : indice {rng.randint(0, 100)}/100 ({rng.choice(['faible', 'moyen', 'élevé'])})"
        return out


# ---------------- Reconstruction 3D volumique procédurale par lieu ----------------

@dataclass
class StreetView3D:
    """Reconstruction 3D (volumes/tuiles) d'un lieu : city-block procédural déterministe.
    Différents lieux -> différents volumes cohérents (seed par lieu)."""
    grid: int = 16
    max_height: int = 10

    def reconstruct(self, lat: float, lon: float) -> torch.Tensor:
        """Volume 3D (1, D, H, W) : occupation voxel d'un city-block du lieu."""
        rng = random.Random(_seed(lat, lon))
        g = self.grid
        vol = torch.zeros(g, g, g)
        # blocs de bâtiments sur la grille (hauteurs déterministes par lieu)
        for by in range(0, g, 2):
            for bx in range(0, g, 2):
                if rng.random() < 0.75:                       # parcelle bâtie
                    h = rng.randint(1, self.max_height)
                    vol[bx:bx + 1, by:by + 1, :h] = 1.0       # bâtiment (volume)
        # rues = zones non bâties (0) -> profondeur navigable jusqu'au sol
        return vol.unsqueeze(0)                               # (1, D, H, W)


# ---------------- Carte navigable ----------------

@dataclass
class GeoMap:
    center: GeoPoint = field(default_factory=lambda: GeoPoint(48.85, 2.35))   # Paris
    zoom: int = 10
    layers: List[str] = field(default_factory=lambda: list(INFO_DOMAINS))
    active_layers: List[str] = field(default_factory=lambda: ["geography", "demographics"])
    locations: Dict[str, GeoPoint] = field(default_factory=lambda: {
        "paris": GeoPoint(48.85, 2.35), "london": GeoPoint(51.51, -0.13),
        "new york": GeoPoint(40.71, -74.01), "tokyo": GeoPoint(35.68, 139.69),
    })
    info_db: InfoDB = field(default_factory=InfoDB)
    view3d: StreetView3D = field(default_factory=StreetView3D)

    def pan(self, dlat: float, dlon: float):
        self.center = GeoPoint(self.center.lat + dlat, self.center.lon + dlon)

    def zoom_in(self):  self.zoom = min(19, self.zoom + 1)
    def zoom_out(self): self.zoom = max(0, self.zoom - 1)
    def center_on(self, point: GeoPoint): self.center = point

    def tile(self) -> Tuple[int, int]:
        return latlon_to_tile(self.center.lat, self.center.lon, self.zoom)

    def search(self, name: str) -> Optional[GeoPoint]:
        return self.locations.get(name.strip().lower())

    def select(self, point: GeoPoint) -> Dict:
        """Sélectionne un lieu -> infos (layers actifs) + reconstruction 3D."""
        return {
            "point": (point.lat, point.lon),
            "tile": latlon_to_tile(point.lat, point.lon, self.zoom),
            "globe_xyz": globe_to_xyz(point.lat, point.lon),
            "info": self.info_db.info(point.lat, point.lon, domain=None)
            if "all" in self.active_layers else
            {d: self.info_db.info(point.lat, point.lon, d)[d] for d in self.active_layers},
            "view3d_shape": tuple(self.view3d.reconstruct(point.lat, point.lon).shape),
        }
