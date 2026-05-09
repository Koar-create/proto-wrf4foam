#!/usr/bin/env python3
"""
Generate a single binary STL: unit wind arrow along +X, total length 1 m (shaft + cone tip).
No trimesh/scipy dependency (avoids NumPy ABI issues on some systems).
"""
from __future__ import annotations

import argparse
import math
import os
import struct
from typing import List, Tuple

Vec3 = Tuple[float, float, float]
Tri = Tuple[Vec3, Vec3, Vec3]


def _write_binary_stl(path: str, triangles: List[Tri]) -> None:
    header = b"WRF-OpenFOAM-Coupling wind arrow unit" + b"\0" * (80 - 37)
    assert len(header) == 80
    n = len(triangles)
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n))
        for v0, v1, v2 in triangles:
            ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            nx = ay * bz - az * by
            ny = az * bx - ax * bz
            nz = ax * by - ay * bx
            ln = (nx * nx + ny * ny + nz * nz) ** 0.5
            if ln > 1e-12:
                nx, ny, nz = nx / ln, ny / ln, nz / ln
            f.write(struct.pack("<12fH", nx, ny, nz, *v0, *v1, *v2, 0))


def _cylinder_open_at_x1(x0: float, x1: float, radius: float, n_seg: int) -> List[Tri]:
    """Cylinder along +X, closed disk at x0 (tail), open at x1 (joins cone)."""
    tris: List[Tri] = []
    for i in range(n_seg):
        t0 = 2 * math.pi * i / n_seg
        t1 = 2 * math.pi * (i + 1) / n_seg
        y0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        y1, z1 = radius * math.cos(t1), radius * math.sin(t1)
        a, b, c = (x0, y0, z0), (x1, y0, z0), (x1, y1, z1)
        d = (x0, y1, z1)
        tris.append((a, b, c))
        tris.append((a, c, d))
    # Tail cap (normal -X)
    apex = (x0, 0.0, 0.0)
    for i in range(n_seg):
        t0 = 2 * math.pi * i / n_seg
        t1 = 2 * math.pi * (i + 1) / n_seg
        y0, z0 = radius * math.cos(t0), radius * math.sin(t0)
        y1, z1 = radius * math.cos(t1), radius * math.sin(t1)
        p0, p1 = (x0, y0, z0), (x0, y1, z1)
        tris.append((apex, p0, p1))
    return tris


def _cone_along_x(x_base: float, x_apex: float, r_base: float, n_seg: int) -> List[Tri]:
    tris: List[Tri] = []
    apex = (x_apex, 0.0, 0.0)
    for i in range(n_seg):
        t0 = 2 * math.pi * i / n_seg
        t1 = 2 * math.pi * (i + 1) / n_seg
        y0, z0 = r_base * math.cos(t0), r_base * math.sin(t0)
        y1, z1 = r_base * math.cos(t1), r_base * math.sin(t1)
        p0, p1 = (x_base, y0, z0), (x_base, y1, z1)
        tris.append((apex, p0, p1))
    return tris


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="gazebo_wind_plugin/models/wind_arrow_glyph/meshes/arrow_unit.stl",
        help="Output path for binary STL",
    )
    args = ap.parse_args()

    shaft_end = 0.72
    r_shaft = 0.028
    r_head = 0.11
    n_seg = 32

    tris: List[Tri] = []
    tris.extend(_cylinder_open_at_x1(0.0, shaft_end, r_shaft, n_seg))
    tris.extend(_cone_along_x(shaft_end, 1.0, r_head, n_seg))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _write_binary_stl(args.out, tris)
    print(f"Wrote {len(tris)} triangles to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
