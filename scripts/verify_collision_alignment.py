#!/usr/bin/env python3
"""
Cross-check the convex-hull collision proxy against the LUT and (optionally)
the visual mesh.

Three-way overlay (top-down slice at `--z`):
  - blue translucent : LUT inside_building cells at the requested Z slice
  - gray             : visual STL triangles intersecting the slice
  - red              : convex-hull boundary outlines at the slice

Tolerance check: every hull's XY footprint at the slice must lie within
`--tolerance-cells` LUT cells of the inside_building mask. Outliers are listed
in stdout and rendered yellow.

Inputs:
  --manifest  data/demo_assets/collision_manifest.json
  --lut       wind_lut.vti (point data must include `inside_building`)
  [--visual-mesh  buildings.stl]  optional, used for the gray overlay
  --z         slice height in metres (default: 80, the hires demo target Z)
  --out       PNG output (default: data/demo_assets/qc_alignment.png)

Exit code 2 if hulls overflow the LUT mask beyond --tolerance-cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lut", type=Path, required=True)
    parser.add_argument("--visual-mesh", type=Path)
    parser.add_argument("--z", type=float, default=80.0, help="Slice height (m).")
    parser.add_argument("--tolerance-cells", type=int, default=2, help="Allowed XY overflow in LUT cells.")
    parser.add_argument("--out", type=Path, default=Path("data/demo_assets/qc_alignment.png"))
    return parser


def load_lut_slice(lut_path: Path, z: float):
    import numpy as np
    import vtk
    from vtk.util import numpy_support

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(lut_path))
    reader.Update()
    img = reader.GetOutput()
    if img is None:
        raise SystemExit(f"VTK reader failed: {lut_path}")
    nx, ny, nz = img.GetDimensions()
    ox, oy, oz = img.GetOrigin()
    sx, sy, sz = img.GetSpacing()
    arr = img.GetPointData().GetArray("inside_building")
    if arr is None:
        raise SystemExit("VTI lacks 'inside_building'")
    flat = numpy_support.vtk_to_numpy(arr).astype(np.uint8)
    mask3 = flat.reshape((nx, ny, nz), order="F")
    iz = int(round((z - oz) / sz))
    iz = max(0, min(nz - 1, iz))
    slab = mask3[:, :, iz]
    xs = ox + sx * np.arange(nx)
    ys = oy + sy * np.arange(ny)
    return slab, xs, ys, (sx, sy)


def load_hulls(manifest: dict, manifest_dir: Path):
    import trimesh

    hulls = []
    for building in manifest.get("buildings", []):
        for h in building["hulls"]:
            path = manifest_dir / h["rel_path"]
            mesh = trimesh.load(str(path), force="mesh")
            hulls.append((building["id"], mesh))
    return hulls


def hull_slice_polygon(mesh, z: float):
    import numpy as np

    try:
        section = mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
        if section is None:
            return None
        planar, _ = section.to_planar()
        if not planar.polygons_full:
            return None
        return [np.asarray(p.exterior.coords) for p in planar.polygons_full]
    except Exception:
        return None


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    manifest = json.loads(args.manifest.read_text())

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        return 1

    slab, xs, ys, (sx, sy) = load_lut_slice(args.lut, args.z)
    hulls = load_hulls(manifest, args.manifest.parent)

    fig, ax = plt.subplots(figsize=(7, 7))
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    ax.pcolormesh(XX, YY, slab, cmap="Blues", alpha=0.45, shading="auto")

    if args.visual_mesh and args.visual_mesh.is_file():
        try:
            import trimesh

            visual = trimesh.load(str(args.visual_mesh), force="mesh")
            section = visual.section(plane_origin=(0, 0, args.z), plane_normal=(0, 0, 1))
            if section is not None:
                planar, _ = section.to_planar()
                for p in planar.polygons_full:
                    poly = np.asarray(p.exterior.coords)
                    ax.plot(poly[:, 0], poly[:, 1], color="0.4", linewidth=0.8, alpha=0.8)
        except Exception as exc:
            print(f"[warn] visual-mesh slice failed: {exc}", file=sys.stderr)

    overflows: List[str] = []
    tol = args.tolerance_cells * max(sx, sy)
    for bid, hull in hulls:
        polys = hull_slice_polygon(hull, args.z)
        if not polys:
            continue
        for poly in polys:
            ax.plot(poly[:, 0], poly[:, 1], color="#d62728", linewidth=1.4, alpha=0.95)
            overflow_pts = []
            for x, y in poly:
                ix = int(round((x - xs[0]) / sx))
                iy = int(round((y - ys[0]) / sy))
                if not (0 <= ix < slab.shape[0] and 0 <= iy < slab.shape[1]):
                    overflow_pts.append((x, y))
                    continue
                if slab[ix, iy] != 0:
                    continue
                ix0 = max(0, ix - args.tolerance_cells)
                ix1 = min(slab.shape[0], ix + args.tolerance_cells + 1)
                iy0 = max(0, iy - args.tolerance_cells)
                iy1 = min(slab.shape[1], iy + args.tolerance_cells + 1)
                if not slab[ix0:ix1, iy0:iy1].any():
                    overflow_pts.append((x, y))
            if overflow_pts:
                overflows.append(
                    f"building {bid:02d}: {len(overflow_pts)} verts overflow >{tol:.1f} m at z={args.z:.0f} m"
                )
                op = np.asarray(overflow_pts)
                ax.scatter(op[:, 0], op[:, 1], s=14, color="#ffd400", edgecolor="black", linewidths=0.4)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"collision/LUT alignment @ z={args.z:.0f} m")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"[write] {args.out}")

    if overflows:
        print("[FAIL] alignment overflow:", file=sys.stderr)
        for line in overflows:
            print(f"  - {line}", file=sys.stderr)
        return 2
    print("[ok] all hull boundaries lie within tolerance of inside_building mask")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
