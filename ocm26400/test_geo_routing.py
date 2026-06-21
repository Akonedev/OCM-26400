"""Tests routing géospatial (OCM-26400)."""
from ocm26400.geo import haversine, route


def test_haversine():
    # Paris (48.85, 2.35) → Lyon (45.76, 4.84) ≈ 390 km
    d = haversine(48.85, 2.35, 45.76, 4.84)
    assert 350 < d < 450


def test_haversine_same_point():
    assert haversine(48.85, 2.35, 48.85, 2.35) < 1.0


def test_route_finds_path():
    locations = [("Paris", 48.85, 2.35), ("Lyon", 45.76, 4.84),
                 ("Marseille", 43.30, 5.37), ("Lille", 50.63, 3.06)]
    path, dist = route(locations, "Paris", "Marseille")
    assert path is not None
    assert path[0] == "Paris" and path[-1] == "Marseille"
    assert dist > 0


def test_route_direct():
    locations = [("A", 0, 0), ("B", 1, 0)]
    path, dist = route(locations, "A", "B")
    assert path == ["A", "B"]
    assert dist > 100  # ~111 km par degré
