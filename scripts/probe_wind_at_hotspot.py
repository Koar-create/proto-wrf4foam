#!/usr/bin/env python3
"""
Probe the LUT wind field at a demo hotspot to (a) verify the dominant flow
direction the drone will be pushed in, and (b) recommend a pt1 spawn pose on
the upwind side of the collision proxy.

Outputs (default under data/demo_assets/):
  - wind_probe.txt          : ASCII summary (mean wind, dominant azimuth, recommended spawn)
  - wind_probe_quiver.png   : top-down quiver at slice z

If `--manifest` is given, the recommended spawn is offset upwind by
`--upwind-clearance-m` from the union XY centroid of the collision hulls so
that the drone is guaranteed to drift into the proxy under steady wind.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lut", type=Path, required=True, help="Path to wind_lut.vti")
    parser.add_argument("--hotspot", type=str, default="1470,1350,80", help="Hotspot 'x,y,z' in metres.")
    parser.add_argument("--half-width-m", type=float, default=30.0, help="XY half-width around hotspot for sampling.")
    parser.add_argument("--half-height-m", type=float, default=20.0, help="Z half-height around hotspot for sampling.")
    parser.add_argument("--n-xy", type=int, default=11, help="Sample count per XY axis.")
    parser.add_argument("--n-z", type=int, default=5, help="Sample count along Z.")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="collision_manifest.json; if provided, recommend pt1 spawn upwind of the union XY centroid.",
    )
    parser.add_argument(
        "--upwind-clearance-m",
        type=float,
        default=18.0,
        help="Spawn offset from collision XY centroid into the upwind direction (m).",
    )
    parser.add_argument(
        "--spawn-altitude-m",
        type=float,
        default=None,
        help="Override spawn Z; default = hotspot Z.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/demo_assets"))
    return parser


def parse_xyz(text: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 3:
        raise SystemExit(f"--hotspot expects 'x,y,z'; got {text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def lut_load(lut_path: Path):
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
    arr = img.GetPointData().GetArray("U")
    if arr is None:
        raise SystemExit("VTI lacks 'U' point array")
    flat = numpy_support.vtk_to_numpy(arr).astype(np.float32)
    U = flat.reshape((nx, ny, nz, 3), order="F")
    return U, (ox, oy, oz), (sx, sy, sz), (nx, ny, nz)


def trilinear(U, origin, spacing, dims, x, y, z):
    import numpy as np

    ox, oy, oz = origin
    sx, sy, sz = spacing
    nx, ny, nz = dims
    fx = (x - ox) / sx
    fy = (y - oy) / sy
    fz = (z - oz) / sz
    if not (0 <= fx <= nx - 1 and 0 <= fy <= ny - 1 and 0 <= fz <= nz - 1):
        return np.zeros(3, dtype=np.float32)
    ix0 = int(np.floor(fx))
    iy0 = int(np.floor(fy))
    iz0 = int(np.floor(fz))
    ix1 = min(nx - 1, ix0 + 1)
    iy1 = min(ny - 1, iy0 + 1)
    iz1 = min(nz - 1, iz0 + 1)
    tx = fx - ix0
    ty = fy - iy0
    tz = fz - iz0
    c000 = U[ix0, iy0, iz0]
    c100 = U[ix1, iy0, iz0]
    c010 = U[ix0, iy1, iz0]
    c110 = U[ix1, iy1, iz0]
    c001 = U[ix0, iy0, iz1]
    c101 = U[ix1, iy0, iz1]
    c011 = U[ix0, iy1, iz1]
    c111 = U[ix1, iy1, iz1]
    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz


def main() -> int:
    args = build_arg_parser().parse_args()
    hx, hy, hz = parse_xyz(args.hotspot)

    try:
        import numpy as np
    except ImportError:
        print("numpy required", file=sys.stderr)
        return 1

    U, origin, spacing, dims = lut_load(args.lut)

    xs = np.linspace(hx - args.half_width_m, hx + args.half_width_m, args.n_xy)
    ys = np.linspace(hy - args.half_width_m, hy + args.half_width_m, args.n_xy)
    zs = np.linspace(hz - args.half_height_m, hz + args.half_height_m, args.n_z)

    samples: List[Tuple[float, float, float, float, float, float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                v = trilinear(U, origin, spacing, dims, x, y, z)
                samples.append((float(x), float(y), float(z), float(v[0]), float(v[1]), float(v[2])))

    arr = np.asarray(samples)
    u_mean = float(arr[:, 3].mean())
    v_mean = float(arr[:, 4].mean())
    w_mean = float(arr[:, 5].mean())
    speed = float(math.hypot(u_mean, v_mean))
    azimuth_deg = float(math.degrees(math.atan2(v_mean, u_mean)))
    if azimuth_deg < 0:
        azimuth_deg += 360.0

    # Recommend spawn upwind of either hotspot or the collision union centroid.
    spawn_anchor = (hx, hy)
    anchor_label = "hotspot"
    bbox_min_xy: List[float] = []
    bbox_max_xy: List[float] = []
    if args.manifest and args.manifest.is_file():
        manifest = json.loads(args.manifest.read_text())
        bb_min = manifest.get("union_bbox_min")
        bb_max = manifest.get("union_bbox_max")
        if bb_min and bb_max:
            spawn_anchor = (
                0.5 * (bb_min[0] + bb_max[0]),
                0.5 * (bb_min[1] + bb_max[1]),
            )
            anchor_label = "collision_union_centroid"
            bbox_min_xy = [float(bb_min[0]), float(bb_min[1])]
            bbox_max_xy = [float(bb_max[0]), float(bb_max[1])]

    if speed < 1e-3:
        ux, uy = 1.0, 0.0
        upwind_note = "wind_speed near zero; spawn defaulted east of anchor"
    else:
        ux = u_mean / speed
        uy = v_mean / speed
        upwind_note = (
            f"upwind unit vector=({-ux:+.2f},{-uy:+.2f}); spawn = anchor + (clearance + bbox_exit) * upwind"
        )

    # Step out along +upwind (= -wind direction) until the candidate spawn is
    # outside the collision union XY bbox, then add the requested clearance.
    sx_lut = max(1.0, args.upwind_clearance_m * 0.05)
    spawn_xy = (spawn_anchor[0] - ux * args.upwind_clearance_m, spawn_anchor[1] - uy * args.upwind_clearance_m)
    if bbox_min_xy and bbox_max_xy:
        n_steps = 0
        max_iter = 10000
        while n_steps < max_iter and (
            bbox_min_xy[0] <= spawn_xy[0] <= bbox_max_xy[0]
            and bbox_min_xy[1] <= spawn_xy[1] <= bbox_max_xy[1]
        ):
            spawn_xy = (spawn_xy[0] - ux * sx_lut, spawn_xy[1] - uy * sx_lut)
            n_steps += 1
        # add an extra clearance once safely outside
        spawn_xy = (
            spawn_xy[0] - ux * args.upwind_clearance_m,
            spawn_xy[1] - uy * args.upwind_clearance_m,
        )
        upwind_note += f"; raycast_steps={n_steps} (bbox_avoidance, cell={sx_lut:.2f} m)"

    spawn_z = args.spawn_altitude_m if args.spawn_altitude_m is not None else hz

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "wind_probe.txt"
    with txt_path.open("w") as f:
        f.write(f"# wind probe at hotspot ({hx},{hy},{hz})\n")
        f.write(f"lut={args.lut}\n")
        f.write(f"sampled n={len(samples)} points\n")
        f.write(
            f"mean wind (m/s): u={u_mean:+.3f} v={v_mean:+.3f} w={w_mean:+.3f}  |U_h|={speed:.3f}  "
            f"azimuth(deg from +x ccw)={azimuth_deg:.1f}\n"
        )
        f.write(f"spawn anchor: {anchor_label} = ({spawn_anchor[0]:.2f},{spawn_anchor[1]:.2f})\n")
        f.write(f"upwind clearance: {args.upwind_clearance_m:.2f} m\n")
        f.write(f"recommended pt1 spawn pose: ({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f}, {spawn_z:.2f})\n")
        f.write(f"note: {upwind_note}\n")
    print(f"[write] {txt_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z_idx = arr[:, 2] == hz
        if not z_idx.any():
            zc = arr[:, 2]
            sel = np.abs(zc - hz) == np.abs(zc - hz).min()
            z_idx = sel
        sub = arr[z_idx]
        fig, ax = plt.subplots(figsize=(7, 7))
        if sub.size > 0:
            ax.quiver(sub[:, 0], sub[:, 1], sub[:, 3], sub[:, 4], scale=40, width=0.004)
        ax.plot(hx, hy, marker="*", markersize=14, color="#ffcc00", markeredgecolor="black", label="hotspot")
        ax.plot(
            spawn_anchor[0],
            spawn_anchor[1],
            marker="o",
            markersize=8,
            color="#1f77b4",
            label=anchor_label,
        )
        ax.plot(
            spawn_xy[0],
            spawn_xy[1],
            marker="s",
            markersize=10,
            color="#2ca02c",
            label="recommended pt1 spawn",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"wind quiver @ z={hz:.0f} m")
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        out_png = out_dir / "wind_probe_quiver.png"
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"[write] {out_png}")
    except ImportError:
        print("[warn] matplotlib unavailable; skip quiver plot", file=sys.stderr)

    print()
    print("Recommended pt1 spawn for guangzhou_demo_pt1_crash.world iris include pose:")
    print(f"  <pose>{spawn_xy[0]:.2f} {spawn_xy[1]:.2f} {spawn_z:.2f} 0 0 0</pose>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
