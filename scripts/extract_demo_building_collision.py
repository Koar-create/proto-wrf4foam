#!/usr/bin/env python3
"""
Extract Gazebo-friendly collision proxies for the buildings nearest the demo hotspot.

Pipeline (Track A in the RBM-4 high-order collision plan):
  1. Load full-city building geometry (`--input <buildings.stl>`) **or** fall back
     to the LUT `inside_building` mask (`--from-lut <wind_lut.vti>`) when the STL
     is unavailable on this host.
  2. Filter triangles within `--radius-m` of one or more `--hotspot x,y` centers.
  3. Split into connected components, sort by oriented bounding box volume, keep
     the top `--max-buildings`.
  4. Decompose each kept building into ODE-friendly convex hulls. Decomposer
     priority: CoACD (pip) > trimesh.decomposition.convex_decomposition (testVHACD)
     > single trimesh.convex_hull fallback.
  5. Write each hull to `<out-dir>/building_<id>/hull_<k>.stl` and emit a single
     `<out-dir>/collision_manifest.json` describing the geometry.
  6. Emit `<out-dir>/qc_overlay.png` (top-down) showing visual triangles, hull
     outlines and hotspots when matplotlib is available.

Outputs are consumed by `scripts/build_demo_collision_model.py` to (re)generate
`gazebo_wind_plugin/models/demo_building_collision/model.sdf` and by
`scripts/verify_collision_alignment.py` to cross-check against the LUT.

Examples:
  python scripts/extract_demo_building_collision.py \
    --input data/microhazard/.../constant/triSurface/buildings.stl \
    --hotspot 1470,1350 --radius-m 50 --max-buildings 2

  # Fallback when buildings.stl is unavailable on this host:
  python scripts/extract_demo_building_collision.py \
    --from-lut ~/wrf_openfoam_coupling_cache/wind_lut/20250903_1400_hires/wind_lut.vti \
    --hotspot 1470,1350 --z-min 0 --z-max 200 --radius-m 50

Backwards compatibility: the legacy `--xo`, `--yo`, `--max-components` and
`--emit-bbox-only` flags are still accepted (translated to the new flags).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def parse_hotspot(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--hotspot expects 'x,y', got {text!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--hotspot floats invalid in {text!r}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", type=Path, help="Path to a city-scale buildings STL (visual mesh).")
    src.add_argument(
        "--from-lut",
        type=Path,
        help="Fallback: path to wind_lut.vti; reconstruct collision boxes from inside_building mask.",
    )

    # Hotspots: support multiple. Default mirrors the hires demo target.
    parser.add_argument(
        "--hotspot",
        action="append",
        type=parse_hotspot,
        help="Hotspot center 'x,y' in metres; may be repeated. Default: 1470,1350.",
    )
    parser.add_argument("--xo", type=float, help="(Legacy) hotspot X.")
    parser.add_argument("--yo", type=float, help="(Legacy) hotspot Y.")

    parser.add_argument("--radius-m", type=float, default=50.0, help="XY filter radius around each hotspot (m).")
    parser.add_argument("--z-min", type=float, default=0.0, help="(LUT path) lower Z bound for mask scanning (m).")
    parser.add_argument("--z-max", type=float, default=200.0, help="(LUT path) upper Z bound for mask scanning (m).")

    parser.add_argument(
        "--max-buildings",
        type=int,
        default=2,
        help="Number of largest connected components to keep (sorted by OBB volume).",
    )
    parser.add_argument("--max-components", type=int, help="(Legacy alias of --max-buildings).")

    parser.add_argument(
        "--decomposer",
        choices=["auto", "coacd", "vhacd", "hull"],
        default="auto",
        help="Convex decomposition backend (auto = coacd > vhacd > hull).",
    )
    parser.add_argument("--coacd-threshold", type=float, default=0.05, help="CoACD concavity threshold.")
    parser.add_argument("--max-hulls", type=int, default=24, help="Cap convex hulls per building.")

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/demo_assets"),
        help="Output directory for hull STLs and manifest JSON.",
    )

    parser.add_argument(
        "--emit-bbox-only",
        action="store_true",
        help="Skip decomposition; only print filtered union bbox (legacy quick-look).",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Skip qc_overlay.png even if matplotlib is available.",
    )
    return parser


def resolve_hotspots(args: argparse.Namespace) -> List[Tuple[float, float]]:
    hotspots: List[Tuple[float, float]] = []
    if args.hotspot:
        hotspots.extend(args.hotspot)
    if args.xo is not None or args.yo is not None:
        legacy = (args.xo if args.xo is not None else 1470.0, args.yo if args.yo is not None else 1350.0)
        if legacy not in hotspots:
            hotspots.append(legacy)
    if not hotspots:
        hotspots.append((1470.0, 1350.0))
    return hotspots


def hash_floats(values: Sequence[float]) -> str:
    h = hashlib.sha1()
    for v in values:
        h.update(f"{v:.6f}".encode("utf-8"))
        h.update(b",")
    return h.hexdigest()[:12]


def import_trimesh():
    try:
        import numpy as np  # noqa: F401
        import trimesh

        return trimesh
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install trimesh numpy") from exc


def candidate_inputs(path: Path) -> str:
    cand = [
        "data/microhazard/<case>/constant/triSurface/buildings.stl",
        "/mnt/e/WRF-OpenFOAM-Coupling/data/microhazard/<case>/constant/triSurface/buildings.stl",
        "~/wrf_openfoam_coupling_cache/buildings.stl",
        "gazebo_wind_plugin/models/guangzhou_buildings/meshes/buildings.stl",
    ]
    return f"--input not found ({path}). Try one of:\n  - " + "\n  - ".join(cand)


def filter_triangles_xy(mesh, hotspots: Sequence[Tuple[float, float]], radius_m: float):
    import numpy as np

    centroids = mesh.triangles_center
    keep_mask = np.zeros(len(centroids), dtype=bool)
    for hx, hy in hotspots:
        d2 = (centroids[:, 0] - hx) ** 2 + (centroids[:, 1] - hy) ** 2
        keep_mask |= d2 < radius_m * radius_m
    return keep_mask


def split_components(submesh):
    parts = submesh.split(only_watertight=False)
    if not parts:
        # Some trimesh versions give back the original on degenerate input.
        return [submesh]
    return list(parts)


def obb_volume(mesh) -> float:
    try:
        ext = mesh.bounding_box_oriented.extents
        return float(ext[0] * ext[1] * ext[2])
    except Exception:
        ext = mesh.bounds[1] - mesh.bounds[0]
        return float(ext[0] * ext[1] * ext[2])


def try_coacd(mesh, threshold: float, max_hulls: int):
    try:
        import coacd  # type: ignore
    except ImportError:
        return None
    import numpy as np
    import trimesh

    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
    )
    hulls = []
    for verts, tris in parts:
        verts = np.asarray(verts, dtype=np.float64)
        tris = np.asarray(tris, dtype=np.int64)
        hull = trimesh.Trimesh(vertices=verts, faces=tris, process=True)
        if not hull.is_volume:
            try:
                hull = hull.convex_hull
            except Exception:
                continue
        hulls.append(hull)
    return hulls or None


def try_trimesh_vhacd(mesh, max_hulls: int):
    try:
        from trimesh import decomposition  # type: ignore
    except Exception:
        return None
    try:
        parts = decomposition.convex_decomposition(mesh, maxhulls=max_hulls)
    except Exception:
        return None
    if not parts:
        return None
    return [p for p in parts if hasattr(p, "vertices") and len(p.vertices) >= 4]


def fallback_single_hull(mesh):
    try:
        return [mesh.convex_hull]
    except Exception:
        return []


def decompose_building(mesh, decomposer: str, threshold: float, max_hulls: int):
    chosen = decomposer
    hulls: Optional[list] = None
    backend_used = "hull"

    if chosen in ("auto", "coacd"):
        hulls = try_coacd(mesh, threshold=threshold, max_hulls=max_hulls)
        if hulls:
            backend_used = "coacd"
    if not hulls and chosen in ("auto", "vhacd"):
        hulls = try_trimesh_vhacd(mesh, max_hulls=max_hulls)
        if hulls:
            backend_used = "vhacd"
    if not hulls:
        hulls = fallback_single_hull(mesh)
        backend_used = "hull"
    return hulls, backend_used


def write_hulls(
    building_id: int,
    hulls,
    out_dir: Path,
):
    bdir = out_dir / f"building_{building_id:02d}"
    bdir.mkdir(parents=True, exist_ok=True)
    items = []
    for k, hull in enumerate(hulls):
        rel = f"building_{building_id:02d}/hull_{k:03d}.stl"
        path = out_dir / rel
        hull.export(path)
        bb_min = [float(x) for x in hull.bounds[0]]
        bb_max = [float(x) for x in hull.bounds[1]]
        items.append(
            {
                "rel_path": rel,
                "vertex_count": int(len(hull.vertices)),
                "face_count": int(len(hull.faces)),
                "bbox_min": bb_min,
                "bbox_max": bb_max,
            }
        )
    return items


def emit_qc_overlay(
    sub,
    hotspots,
    radius_m: float,
    hulls_per_building,
    out_path: Path,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib unavailable; skip qc_overlay.png", file=sys.stderr)
        return False

    fig, ax = plt.subplots(figsize=(7, 7))
    if sub is not None:
        tri = sub.triangles[:, :, :2]
        for t in tri:
            ax.plot(
                [t[0, 0], t[1, 0], t[2, 0], t[0, 0]],
                [t[0, 1], t[1, 1], t[2, 1], t[0, 1]],
                color="0.6",
                linewidth=0.3,
                alpha=0.6,
            )
    cmap = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for bid, hulls in enumerate(hulls_per_building):
        color = cmap[bid % len(cmap)]
        for hull in hulls:
            verts = hull.vertices[:, :2]
            try:
                from scipy.spatial import ConvexHull  # type: ignore

                cv = ConvexHull(verts)
                poly = verts[cv.vertices]
            except Exception:
                poly = verts
            poly = np.vstack([poly, poly[:1]])
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1.5, alpha=0.9)
    for hx, hy in hotspots:
        ax.plot(hx, hy, marker="*", markersize=14, color="#ffcc00", markeredgecolor="black")
        circ = plt.Circle((hx, hy), radius_m, fill=False, linestyle="--", color="#ffcc00", linewidth=1.0)
        ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("demo_building_collision QC overlay")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def reconstruct_from_lut(
    lut_path: Path,
    hotspots,
    radius_m: float,
    z_min: float,
    z_max: float,
    max_buildings: int,
):
    try:
        import numpy as np
        import trimesh
        import vtk
        from vtk.util import numpy_support
    except ImportError as exc:
        raise SystemExit("Missing dependency for --from-lut: pip install vtk numpy trimesh") from exc

    if not lut_path.is_file():
        raise SystemExit(f"--from-lut not found: {lut_path}")

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(lut_path))
    reader.Update()
    img = reader.GetOutput()
    if img is None:
        raise SystemExit(f"VTK reader failed for {lut_path}")
    nx, ny, nz = img.GetDimensions()
    ox, oy, oz = img.GetOrigin()
    sx, sy, sz = img.GetSpacing()

    pd = img.GetPointData()
    arr = pd.GetArray("inside_building")
    if arr is None:
        raise SystemExit("VTI lacks 'inside_building' point array; cannot fall back from LUT")
    flat = numpy_support.vtk_to_numpy(arr).astype(np.uint8)
    # VTI point order: z fastest by default
    mask3 = flat.reshape((nx, ny, nz), order="F")

    iz_lo = max(0, int(round((z_min - oz) / sz)))
    iz_hi = min(nz - 1, int(round((z_max - oz) / sz)))
    if iz_hi < iz_lo:
        iz_lo, iz_hi = iz_hi, iz_lo
    sub_mask = mask3[:, :, iz_lo : iz_hi + 1]

    # 2D ROI mask: any z slice has inside_building within hotspot radius
    column_any = sub_mask.any(axis=2)

    xs = ox + sx * np.arange(nx)
    ys = oy + sy * np.arange(ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    roi = np.zeros_like(column_any, dtype=bool)
    for hx, hy in hotspots:
        roi |= (XX - hx) ** 2 + (YY - hy) ** 2 < radius_m * radius_m
    column_any &= roi

    try:
        from scipy.ndimage import label  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install scipy") from exc

    labeled, n_lab = label(column_any)
    if n_lab == 0:
        raise SystemExit("LUT inside_building has no cells inside hotspot ROI; widen --radius-m or check Z bounds")

    # For each label, build an axis-aligned box from XY extent and slice-wise Z extent
    sub_meshes = []
    for lab in range(1, n_lab + 1):
        ix_y = np.argwhere(labeled == lab)
        if len(ix_y) == 0:
            continue
        ix_min = ix_y[:, 0].min()
        ix_max = ix_y[:, 0].max()
        iy_min = ix_y[:, 1].min()
        iy_max = ix_y[:, 1].max()
        col_mask = sub_mask[ix_min : ix_max + 1, iy_min : iy_max + 1, :]
        z_any = col_mask.any(axis=(0, 1))
        if not z_any.any():
            continue
        iz_idx = np.argwhere(z_any).flatten()
        z_lo = oz + sz * (iz_lo + iz_idx.min())
        z_hi = oz + sz * (iz_lo + iz_idx.max() + 1)
        x_lo = ox + sx * ix_min
        x_hi = ox + sx * (ix_max + 1)
        y_lo = oy + sy * iy_min
        y_hi = oy + sy * (iy_max + 1)
        ext = (x_hi - x_lo, y_hi - y_lo, z_hi - z_lo)
        center = (0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), 0.5 * (z_lo + z_hi))
        if ext[0] < sx or ext[1] < sy or ext[2] < sz:
            continue
        box = trimesh.creation.box(extents=ext)
        box.apply_translation(center)
        sub_meshes.append(box)

    sub_meshes.sort(key=obb_volume, reverse=True)
    return sub_meshes[:max_buildings]


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.max_components is not None:
        args.max_buildings = args.max_components

    hotspots = resolve_hotspots(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trimesh = import_trimesh()
    import numpy as np

    if args.from_lut:
        print(f"[info] LUT fallback path: {args.from_lut}")
        buildings = reconstruct_from_lut(
            args.from_lut,
            hotspots=hotspots,
            radius_m=args.radius_m,
            z_min=args.z_min,
            z_max=args.z_max,
            max_buildings=args.max_buildings,
        )
        if not buildings:
            print("[err] LUT fallback produced 0 buildings", file=sys.stderr)
            return 1
        print(f"[info] LUT fallback: {len(buildings)} building boxes reconstructed")
        sub = None
        face_count = 0
    else:
        if not args.input or not args.input.is_file():
            print(candidate_inputs(args.input or Path("<unset>")), file=sys.stderr)
            return 1

        loaded = trimesh.load(str(args.input))
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        else:
            mesh = loaded
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"Unsupported mesh type: {type(mesh)}", file=sys.stderr)
            return 1

        face_mask = filter_triangles_xy(mesh, hotspots, args.radius_m)
        kept = int(np.count_nonzero(face_mask))
        if kept == 0:
            print("No triangles inside any hotspot radius; widen --radius-m", file=sys.stderr)
            return 1
        sub = mesh.submesh([np.where(face_mask)[0]], append=True)
        print(f"[info] kept_faces={kept} / {len(mesh.faces)}  vertices={len(sub.vertices)}")
        face_count = int(len(sub.faces))

        if args.emit_bbox_only:
            mn, mx = sub.bounds
            size = mx - mn
            center = 0.5 * (mn + mx)
            print(
                f"[bbox] filtered_union min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) "
                f"max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})"
            )
            print(
                f"[bbox] filtered_union size_xyz=({size[0]:.3f},{size[1]:.3f},{size[2]:.3f}) "
                f"center_xyz=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})"
            )
            return 0

        components = split_components(sub)
        components.sort(key=obb_volume, reverse=True)
        buildings = components[: max(1, args.max_buildings)]
        print(f"[info] components_total={len(components)} kept={len(buildings)}")

    manifest_buildings = []
    hulls_per_building = []
    for bid, mesh in enumerate(buildings):
        hulls, backend = decompose_building(
            mesh,
            decomposer=args.decomposer,
            threshold=args.coacd_threshold,
            max_hulls=args.max_hulls,
        )
        hulls = hulls or []
        if not hulls:
            print(f"[warn] building {bid} produced 0 hulls; skipping", file=sys.stderr)
            continue
        items = write_hulls(bid, hulls, out_dir)
        bb_min = [
            float(min(it["bbox_min"][k] for it in items)) for k in range(3)
        ]
        bb_max = [
            float(max(it["bbox_max"][k] for it in items)) for k in range(3)
        ]
        print(
            f"[write] building_{bid:02d} backend={backend} hulls={len(items)} "
            f"bbox_xy=({bb_min[0]:.1f}..{bb_max[0]:.1f}, {bb_min[1]:.1f}..{bb_max[1]:.1f}) "
            f"z=({bb_min[2]:.1f}..{bb_max[2]:.1f})"
        )
        manifest_buildings.append(
            {
                "id": bid,
                "decomposer": backend,
                "hulls": items,
                "bbox_min": bb_min,
                "bbox_max": bb_max,
            }
        )
        hulls_per_building.append(hulls)

    if not manifest_buildings:
        print("[err] no buildings produced any hulls", file=sys.stderr)
        return 1

    union_min = [min(b["bbox_min"][k] for b in manifest_buildings) for k in range(3)]
    union_max = [max(b["bbox_max"][k] for b in manifest_buildings) for k in range(3)]

    manifest = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "source": "lut_fallback" if args.from_lut else str(args.input),
        "hotspots": [list(h) for h in hotspots],
        "radius_m": args.radius_m,
        "decomposer_request": args.decomposer,
        "max_hulls_per_building": args.max_hulls,
        "filtered_face_count": face_count,
        "buildings": manifest_buildings,
        "union_bbox_min": union_min,
        "union_bbox_max": union_max,
        "hash": hash_floats(union_min + union_max + [args.radius_m, len(manifest_buildings)]),
    }

    manifest_path = out_dir / "collision_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {manifest_path}")

    if not args.no_overlay:
        overlay_path = out_dir / "qc_overlay.png"
        if emit_qc_overlay(sub, hotspots, args.radius_m, hulls_per_building, overlay_path):
            print(f"[write] {overlay_path}")

    print(
        "\nNext steps:\n"
        "  1. python scripts/build_demo_collision_model.py "
        f"--manifest {manifest_path} \\\n"
        "       --model-dir gazebo_wind_plugin/models/demo_building_collision\n"
        "  2. (optional) python scripts/verify_collision_alignment.py "
        f"--manifest {manifest_path} \\\n"
        "       --lut ~/wrf_openfoam_coupling_cache/wind_lut/<dataset>/wind_lut.vti \\\n"
        "       --out data/demo_assets/qc_alignment.png"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
