#!/usr/bin/env python3
"""Convert between lon/lat and local XY around the Guangzhou building origin.

This utility performs coordinate conversion only. It does not read or write
any project data files.

Examples:
    python util/convert_lonlat_xy_origin.py --lonlat 113.3218197 23.1133057
    python util/convert_lonlat_xy_origin.py --xy 100.0 -50.0
"""

from __future__ import annotations

import argparse

ORIGIN_LON = 113.3218197
ORIGIN_LAT = 23.1133057
WGS84_CRS = "EPSG:4326"
UTM49N_CRS = "EPSG:32649"


def make_transformers():
    """Create forward and inverse WGS84 <-> UTM49N transformers."""
    from pyproj import Transformer

    to_utm = Transformer.from_crs(WGS84_CRS, UTM49N_CRS, always_xy=True)
    to_lonlat = Transformer.from_crs(UTM49N_CRS, WGS84_CRS, always_xy=True)
    return to_utm, to_lonlat


def origin_utm49n() -> tuple[float, float]:
    """Return the fixed origin in UTM49N metres."""
    to_utm, _ = make_transformers()
    ox, oy = to_utm.transform(ORIGIN_LON, ORIGIN_LAT)
    return float(ox), float(oy)


def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    """Convert WGS84 lon/lat degrees to local XY metres."""
    to_utm, _ = make_transformers()
    ox, oy = origin_utm49n()
    easting, northing = to_utm.transform(lon, lat)
    return float(easting - ox), float(northing - oy)


def xy_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """Convert local XY metres to WGS84 lon/lat degrees."""
    _, to_lonlat = make_transformers()
    ox, oy = origin_utm49n()
    lon, lat = to_lonlat.transform(ox + x, oy + y)
    return float(lon), float(lat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert between WGS84 lon/lat and local XY metres relative to "
            f"({ORIGIN_LON} E, {ORIGIN_LAT} N)."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--lonlat",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        help="Input WGS84 longitude and latitude in degrees.",
    )
    mode.add_argument(
        "--xy",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Input local XY coordinates in metres relative to the fixed origin.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places to print (default: 6).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    p = args.precision

    ox, oy = origin_utm49n()
    print(f"origin_lon={ORIGIN_LON:.7f}")
    print(f"origin_lat={ORIGIN_LAT:.7f}")
    print(f"origin_utm49n_easting={ox:.{p}f}")
    print(f"origin_utm49n_northing={oy:.{p}f}")

    if args.lonlat is not None:
        lon, lat = args.lonlat
        x, y = lonlat_to_xy(lon, lat)
        print(f"x={x:.{p}f}")
        print(f"y={y:.{p}f}")
    else:
        x, y = args.xy
        lon, lat = xy_to_lonlat(x, y)
        print(f"lon={lon:.{p}f}")
        print(f"lat={lat:.{p}f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
