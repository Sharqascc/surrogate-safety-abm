"""3-way verification of lat/lon → metres + bearing."""
from __future__ import annotations
import math
from pathlib import Path
import networkx as nx
import pytest
from pyproj import Geod
from surrogate_safety_abm.environment.intersection import Intersection
from surrogate_safety_abm.geometry import lat_lon_to_local_metres

REPO_ROOT = Path(__file__).parent.parent.parent
INT_DIR = REPO_ROOT / "data" / "intersections"

def v1_equirect(lat0, lon0, lat1, lon1):
    x, y = lat_lon_to_local_metres(lat0, lon0, lat1, lon1)
    return math.hypot(x, y), math.degrees(math.atan2(x, y)) % 360

def v2_geod(lat0, lon0, lat1, lon1):
    az, _, d = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)
    return d, az % 360

def v3_sphere(lat0, lon0, lat1, lon1):
    """V3: pyproj.Geod on a sphere — genuinely different from V2's ellipsoid.

    pyproj.Geod.inv signature is (lons1, lats1, lons2, lats2) — note
    lon-before-lat ordering, opposite of the (lat, lon) convention used
    elsewhere in this test file.
    """
    az, _, d = Geod(ellps="sphere").inv(lon0, lat0, lon1, lat1)
    return d, az % 360

def _adiff(a, b):
    d = (a - b) % 360
    return min(d, 360 - d)

HAND = [
    (0.0, 0.0, 0.008993, 0.0, 1000.0, 0.0, "1km N equator"),
    (0.0, 0.0, 0.0, 0.008993, 1000.0, 90.0, "1km E equator"),
    (0.0, 0.0, -0.008993, 0.0, 1000.0, 180.0, "1km S equator"),
    (0.0, 0.0, 0.0, -0.008993, 1000.0, 270.0, "1km W equator"),
    (22.28, 73.14, 22.28+0.008993, 73.14, 1000.0, 0.0, "1km N Vadodara"),
    (22.28, 73.14, 22.28, 73.14+0.00973, 1000.0, 90.0, "1km E Vadodara"),
    (-22.5, 73.0, -22.5+0.008993, 73.0, 1000.0, 0.0, "1km N S-hemisphere"),
]

@pytest.mark.parametrize("lat0,lon0,lat1,lon1,ed,eb,lbl", HAND)
def test_hand(lat0, lon0, lat1, lon1, ed, eb, lbl):
    d1, b1 = v1_equirect(lat0, lon0, lat1, lon1)
    d2, b2 = v2_geod(lat0, lon0, lat1, lon1)
    d3, b3 = v3_sphere(lat0, lon0, lat1, lon1)
    for name, d in [("V1", d1), ("V2", d2), ("V3", d3)]:
        assert d == pytest.approx(ed, rel=0.01), f"{name} dist [{lbl}]"
    for name, b in [("V1", b1), ("V2", b2), ("V3", b3)]:
        assert _adiff(b, eb) < 1.0, f"{name} bearing [{lbl}]: {b:.2f} vs {eb}"

def _check_site(gp, nid, lbl):
    G = nx.read_graphml(gp)
    ce = G.nodes[nid]
    cla, clo = float(ce["y"]), float(ce["x"])
    nbs = sorted(set(G.successors(nid)) | set(G.predecessors(nid)))
    assert len(nbs) >= 3
    for nb in nbs:
        n = G.nodes[nb]
        nla, nlo = float(n["y"]), float(n["x"])
        d1, b1 = v1_equirect(cla, clo, nla, nlo)
        d2, b2 = v2_geod(cla, clo, nla, nlo)
        d3, b3 = v3_sphere(cla, clo, nla, nlo)
        assert d1 == pytest.approx(d2, rel=0.005), f"{lbl} {nb} V1/V2"
        assert d1 == pytest.approx(d3, rel=0.005), f"{lbl} {nb} V1/V3"
        assert _adiff(b1, b2) < 0.5, f"{lbl} {nb} V1/V2 bearing"
        assert _adiff(b1, b3) < 0.5, f"{lbl} {nb} V1/V3 bearing"

def test_qismat_3way():
    _check_site(INT_DIR / "qismat_chowck.graphml", "8499269593", "Qismat")

def test_akota_3way():
    _check_site(INT_DIR / "Akota_garden.graphml", "8631923919", "Akota")

def test_from_osm_qismat():
    ic = Intersection.from_osm(INT_DIR / "qismat_chowck.graphml",
                               "8499269593", "qismat_chowck")
    assert len(ic.arms) >= 3
    assert len(ic.conflict_points) >= 3

def test_from_osm_akota():
    ic = Intersection.from_osm(INT_DIR / "Akota_garden.graphml",
                               "8631923919", "akota_garden")
    assert len(ic.arms) >= 3

def test_zero():
    x, y = lat_lon_to_local_metres(22.28, 73.14, 22.28, 73.14)
    assert x == 0.0 and y == 0.0

def test_from_osm_bad_node():
    with pytest.raises(ValueError, match="not found"):
        Intersection.from_osm(INT_DIR / "qismat_chowck.graphml", "9999999999")
