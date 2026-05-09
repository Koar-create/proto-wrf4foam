#!/usr/bin/env python3
"""
Extract 1–2 largest connected building meshes near a demo hotspot from a full-city STL.

Typical inputs:
  - OpenFOAM case: constant/triSurface/buildings.stl
  - Or any large STL; pass --input explicitly.

Requires: pip install trimesh numpy

Outputs (default under data/demo_assets/):
  - collision_building_A.stl (largest component by triangle count)
  - collision_building_B.stl (optional second)

Prints axis-aligned bounding boxes for Gazebo <pose> / <box><size> proxies.

If filtering yields a single huge component (common for merged city meshes), use
--emit-bbox-only to print a recommended box from the filtered face set without STL export,
then place a box collision in SDF manually (see demo_building_collision model comments).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="Path to full buildings STL")
    parser.add_argument("--xo", type=float, default=1470.0, help="Hotspot X (m)")
    parser.add_argument("--yo", type=float, default=1350.0, help="Hotspot Y (m)")
    parser.add_argument("--radius-m", type=float, default=50.0, help="XY radius around hotspot for face filter (m)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/demo_assets"),
        help="Output directory for extracted STLs",
    )
    parser.add_argument("--max-components", type=int, default=2, help="How many largest components to export")
    parser.add_argument(
        "--emit-bbox-only",
        action="store_true",
        help="Only print bbox of filtered faces (no STL write); useful when split() returns one giant mesh",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import trimesh
    except ImportError as e:
        print("Missing dependency: pip install trimesh numpy", file=sys.stderr)
        raise SystemExit(1) from e

    if not args.input.is_file():
        print(f"Input STL not found: {args.input}", file=sys.stderr)
        return 1

    loaded = trimesh.load(str(args.input))
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        print(f"Unsupported mesh type: {type(mesh)}", file=sys.stderr)
        return 1

    centroids = mesh.triangles_center
    dx = centroids[:, 0] - args.xo
    dy = centroids[:, 1] - args.yo
    dist = np.sqrt(dx * dx + dy * dy)
    face_mask = dist < args.radius_m
    kept = int(np.count_nonzero(face_mask))
    if kept == 0:
        print("No triangles within radius; widen --radius-m or check hotspot / units.", file=sys.stderr)
        return 1

    sub = mesh.submesh([np.where(face_mask)[0]], append=True)
    print(f"[info] kept_faces={kept} / {len(mesh.faces)}  vertices={len(sub.vertices)}")

    def print_bbox(name: str, m: trimesh.Trimesh) -> None:
        b = m.bounds
        mn, mx = b[0], b[1]
        size = mx - mn
        center = 0.5 * (mn + mx)
        print(f"[bbox] {name} min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})")
        print(f"[bbox] {name} size_xyz=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f}) center_xyz=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})")

    print_bbox("filtered_union", sub)

    if args.emit_bbox_only:
        return 0

    parts = sub.split(only_watertight=False)
    if not parts:
        print("[warn] split() returned empty; try --emit-bbox-only", file=sys.stderr)
        return 1

    parts_sorted = sorted(parts, key=lambda m: len(m.faces), reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["collision_building_A.stl", "collision_building_B.stl"]
    for i, part in enumerate(parts_sorted[: max(1, args.max_components)]):
        out_path = args.out_dir / labels[i]
        part.export(out_path)
        print(f"[write] {out_path} faces={len(part.faces)}")
        print_bbox(labels[i], part)

    if len(parts_sorted) == 1 and len(parts_sorted[0].faces) > 500_000:
        print(
            "[warn] Single very large component after filter; consider --emit-bbox-only and a box collision proxy in Gazebo.",
            file=sys.stderr,
        )

    print(
        "\nNext: copy collision_building_A.stl to "
        "gazebo_wind_plugin/models/demo_building_collision/meshes/building_A.stl "
        "and optionally switch model.sdf geometry from box to mesh."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
