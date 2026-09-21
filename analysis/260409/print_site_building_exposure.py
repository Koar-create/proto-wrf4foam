"""Report the tallest building within a set of radii around each LiDAR site,
using the Guangzhou LOD1 building footprint dataset (attribute `jzgd` = height, m).

Used to check the qualitative claim (from 2026-07-24 meeting discussion) that
GAW111 (Haixinsha river island) has a much lower building-height ceiling in its
immediate surroundings than GAW103/GAW104.

Requires: pyshp, pyproj; util/lidar_station_info.json (gitignored, not committed).
Usage: python analysis/260409/print_site_building_exposure.py
"""
import json
import math
from pathlib import Path

import pyproj
import shapefile

REPO = Path(__file__).resolve().parents[2]
SHP_PATH = REPO / "data" / "Guangzhou_shp_file" / "project_UTM49" / "Export_Output.shp"
STATION_JSON = REPO / "util" / "lidar_station_info.json"
HEIGHT_FIELD = "jzgd"
RADII_M = [100, 200, 300, 500]
SITE_IDS = ("GAW103", "GAW104", "GAW111")


def load_sites(json_path: Path) -> dict[str, tuple[float, float]]:
    with open(json_path, encoding="utf-8") as f:
        station_data = json.load(f)
    sites: dict[str, tuple[float, float]] = {}
    for site_id in SITE_IDS:
        if site_id not in station_data:
            raise KeyError(f"missing station {site_id!r} in {json_path}")
        info = station_data[site_id]["station_info"]
        sites[site_id] = (info["lon"], info["lat"])
    return sites


def centroid(shape):
    xs = [p[0] for p in shape.points]
    ys = [p[1] for p in shape.points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def main():
    sites = load_sites(STATION_JSON)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32649", always_xy=True)
    sf = shapefile.Reader(str(SHP_PATH), encoding="gbk", encodingErrors="ignore")
    shapes = sf.shapes()
    records = sf.records()
    centroids = [centroid(s) for s in shapes]

    for name, (lon, lat) in sites.items():
        x0, y0 = transformer.transform(lon, lat)
        print(f"{name}:")
        for radius in RADII_M:
            heights = [
                rec[HEIGHT_FIELD]
                for (cx, cy), rec in zip(centroids, records)
                if math.hypot(cx - x0, cy - y0) <= radius and rec[HEIGHT_FIELD] is not None
            ]
            if heights:
                print(
                    f"  r={radius:>4} m: n={len(heights):>3}, "
                    f"max={max(heights):.1f} m, mean={sum(heights) / len(heights):.1f} m"
                )
            else:
                print(f"  r={radius:>4} m: no buildings")


if __name__ == "__main__":
    main()
