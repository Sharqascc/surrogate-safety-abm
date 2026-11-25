# OSM intersection extracts

Source: OpenStreetMap, downloaded and converted with `osmnx` on 2026-11-25T08:21:28.

| name | display | centroid lat | centroid lon | nodes | edges |
|---|---|---|---|---|---|
| qismat_chowck | Qismat Chowk, Vadodara | 22.28570 | 73.14279 | 74 | 158 |
| Akota_garden | Akota Garden, Vadodara | 22.29226 | 73.17212 | 103 | 194 |

## Files per intersection

- `<name>.osm` — raw OSM XML
- `<name>.graphml` — NetworkX graph
- `<name>.gpkg` — GeoPackage (QGIS)

## Re-fetch

```python
import osmnx as ox
G = ox.graph_from_xml('qismat_chowck.osm', simplify=True)
```
