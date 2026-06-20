"""Tests TDD — couche géospatial (OCM-26400, spec cartes/globe/street/3D).

Valide : Web Mercator (lat/lon<->tuile), infos multi-domaines déterministes par lieu,
projection globe, reconstruction 3D procédurale par lieu, navigation/recherche/sélection.
"""
import math
import torch

from ocm26400.geo import (
    GeoPoint, latlon_to_tile, tile_to_latlon, globe_to_xyz, InfoDB, StreetView3D,
    GeoMap, INFO_DOMAINS,
)


def test_latlon_tile_consistency():
    """lat/lon -> tuile -> lat/lon -> tuile : cohérence Web Mercator."""
    lat, lon, z = 48.85, 2.35, 10
    t = latlon_to_tile(lat, lon, z)
    la2, lo2 = tile_to_latlon(*t, z)
    assert latlon_to_tile(la2, lo2, z) == t


def test_info_domains_complete_and_deterministic():
    """InfoDB couvre tous les domaines et est déterministe par lieu."""
    db = InfoDB()
    info = db.info(48.85, 2.35)
    for d in INFO_DOMAINS:
        assert d in info
    # déterministe : même lieu -> même info
    assert db.info(48.85, 2.35) == info
    # lieu différent -> info différente
    assert db.info(40.71, -74.01) != info


def test_globe_projection_unit_sphere():
    """Projection globe : (0,0) -> (1,0,0) sur la sphère unité."""
    x, y, z = globe_to_xyz(0.0, 0.0, r=1.0)
    assert abs(x - 1.0) < 1e-6 and abs(y) < 1e-6 and abs(z) < 1e-6
    gx, gy, gz = globe_to_xyz(45.0, 90.0)
    assert abs(math.sqrt(gx * gx + gy * gy + gz * gz) - 1.0) < 1e-6


def test_street3d_reconstructs_volume_per_location():
    """Reconstruction 3D : volume (1,g,g,g), distinct par lieu (cohérent par lieu)."""
    v = StreetView3D(grid=12).reconstruct(48.85, 2.35)
    assert v.shape == (1, 12, 12, 12)
    assert float(v.sum()) > 0                                   # des bâtiments
    v2 = StreetView3D(grid=12).reconstruct(40.71, -74.01)
    assert not torch.equal(v, v2)                               # lieu différent -> volume différent
    # cohérent par lieu : même lieu -> même volume
    assert torch.equal(StreetView3D(grid=12).reconstruct(48.85, 2.35), v)


def test_geomap_navigation_and_search():
    """Navigation (pan/zoom/center), recherche par nom -> lieu."""
    m = GeoMap()
    z0 = m.zoom
    m.zoom_in(); assert m.zoom == z0 + 1
    m.zoom_out(); assert m.zoom == z0
    m.pan(1.0, 1.0); assert abs(m.center.lat - 49.85) < 1e-9
    p = m.search("paris")
    assert p is not None and abs(p.lat - 48.85) < 1e-9
    assert m.search("inexistant") is None


def test_geomap_select_returns_info_and_3d():
    """Sélection d'un lieu -> infos (layers actifs) + reconstruction 3D + globe."""
    m = GeoMap(active_layers=["geography", "demographics"])
    res = m.select(GeoPoint(48.85, 2.35))
    assert "geography" in res["info"] and "demographics" in res["info"]
    assert len(res["globe_xyz"]) == 3
    assert res["view3d_shape"][0] == 1                          # volume 3D


def test_layers_activables_filter_info():
    """Les layers actives filtrent les infos affichées."""
    m = GeoMap(active_layers=["climate"])
    res = m.select(GeoPoint(35.68, 139.69))
    assert list(res["info"].keys()) == ["climate"]
