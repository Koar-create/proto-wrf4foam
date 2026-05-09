#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrowSample:
    x: float
    y: float
    z: float
    u: float
    v: float
    w: float

    @property
    def mag(self) -> float:
        return float(math.sqrt(self.u * self.u + self.v * self.v + self.w * self.w))


def _nearest_index(coords: np.ndarray, value: float) -> int:
    # coords is monotonic increasing
    i = int(np.searchsorted(coords, value))
    if i <= 0:
        return 0
    if i >= len(coords):
        return len(coords) - 1
    # choose closer of i-1, i
    return i if abs(coords[i] - value) < abs(coords[i - 1] - value) else i - 1


def _color_for_speed(speed: float, vmin: float, vmax: float) -> tuple[float, float, float, float]:
    # Simple blue->red ramp (--color-mode legacy)
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (speed - vmin) / (vmax - vmin)))
    r = t
    g = 0.15 + 0.25 * (1.0 - abs(t - 0.5) * 2.0)
    b = 1.0 - t
    a = 0.85
    return (float(r), float(g), float(b), float(a))


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """h in [0,1), standard HSV to RGB."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (float(r), float(g), float(b))


def _color_hsv_speed(speed: float, vmin: float, vmax: float) -> tuple[float, float, float, float]:
    """
    Map speed to blue (low) -> red (high) using hue sweep on HSV.
    Hue: blue ~240° -> red 0° => h from 2/3 down to 0.
    """
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (speed - vmin) / (vmax - vmin)))
    # t=0 -> blue (h=2/3), t=1 -> red (h=0)
    h = (2.0 / 3.0) * (1.0 - t)
    r, g, b = _hsv_to_rgb(h, 0.92, 0.98)
    a = 0.88
    return (r, g, b, a)


def _pose_rpy_for_dir(u: float, v: float, w: float) -> tuple[float, float, float]:
    # Arrow aligned to +X in its local frame (length along X).
    # yaw rotates around Z, pitch around Y.
    h = math.sqrt(u * u + v * v)
    yaw = math.atan2(v, u)
    pitch = math.atan2(w, h)
    roll = 0.0
    return roll, pitch, yaw


def _generate_samples_centered(
    U: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    z_levels: list[float],
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    z_xy_stagger: bool,
) -> list[ArrowSample]:
    xs = [center_x + (i - (nx - 1) / 2.0) * dx for i in range(nx)]
    ys = [center_y + (j - (ny - 1) / 2.0) * dy for j in range(ny)]

    out: list[ArrowSample] = []
    for zi, z in enumerate(z_levels):
        ox, oy = (
            _xy_stagger_for_layer(zi, dx, dy)
            if z_xy_stagger and len(z_levels) > 1
            else (0.0, 0.0)
        )
        iz = _nearest_index(z_coords, z)
        for y in ys:
            iy = _nearest_index(y_coords, y + oy)
            for x in xs:
                ix = _nearest_index(x_coords, x + ox)
                u, v, w = (float(U[ix, iy, iz, 0]), float(U[ix, iy, iz, 1]), float(U[ix, iy, iz, 2]))
                out.append(ArrowSample(x=x + ox, y=y + oy, z=float(z_coords[iz]), u=u, v=v, w=w))
    return out


def _frange_inclusive(a: float, b: float, step: float) -> list[float]:
    """Inclusive-ish grid from a to b with step (handles float endpoints)."""
    out: list[float] = []
    x = a
    while x <= b + step * 1e-6:
        out.append(round(x, 6))
        x += step
    return out


def _xy_stagger_for_layer(layer_index: int, step_x: float, step_y: float) -> tuple[float, float]:
    """
    Per-height horizontal shift so multi-z-level arrows do not share identical (x, y).

    Layer 0 is the reference grid; higher layers use sub-cell offsets (brick-style) so
    columns do not stack visually on the same XY pillars.
    """
    if layer_index <= 0:
        return (0.0, 0.0)
    k = (layer_index - 1) % 4
    if k == 0:
        return (0.5 * step_x, 0.0)
    if k == 1:
        return (0.0, 0.5 * step_y)
    if k == 2:
        return (0.5 * step_x, 0.5 * step_y)
    return (0.25 * step_x, 0.25 * step_y)


def _generate_samples_bbox(
    U: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    step: float,
    z_levels: list[float],
    z_xy_stagger: bool,
) -> list[ArrowSample]:
    xs = _frange_inclusive(x_min, x_max, step)
    ys = _frange_inclusive(y_min, y_max, step)
    out: list[ArrowSample] = []
    for zi, z in enumerate(z_levels):
        ox, oy = (
            _xy_stagger_for_layer(zi, step, step)
            if z_xy_stagger and len(z_levels) > 1
            else (0.0, 0.0)
        )
        iz = _nearest_index(z_coords, z)
        for y in ys:
            iy = _nearest_index(y_coords, y + oy)
            for x in xs:
                ix = _nearest_index(x_coords, x + ox)
                u, v, w = (float(U[ix, iy, iz, 0]), float(U[ix, iy, iz, 1]), float(U[ix, iy, iz, 2]))
                out.append(ArrowSample(x=x + ox, y=y + oy, z=float(z_coords[iz]), u=u, v=v, w=w))
    return out


def _write_model_config(model_dir: str, model_name: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    model_config = f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>auto</name>
    <email>n/a</email>
  </author>
  <description>
    Visual-only wind arrows sampled from Wind LUT (static).
  </description>
</model>
"""
    with open(os.path.join(model_dir, "model.config"), "w", encoding="utf-8") as f:
        f.write(model_config)


def _write_model(
    model_dir: str,
    model_name: str,
    samples: list[ArrowSample],
    *,
    color_mode: str,
    mesh_uri: str,
    mesh_len_min: float,
    mesh_len_max: float,
    mesh_len_k: float,
    mesh_thick: float,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    _write_model_config(model_dir, model_name)

    speeds = [s.mag for s in samples]
    vmin = float(np.percentile(speeds, 10)) if speeds else 0.0
    vmax = float(np.percentile(speeds, 95)) if speeds else 1.0

    pick_color = _color_hsv_speed if color_mode == "hsv" else _color_for_speed

    # Many separate links each loading the same STL often renders only one instance; use one link + N visuals.
    lines: list[str] = []
    lines.append('<?xml version="1.0" ?>')
    lines.append('<sdf version="1.6">')
    lines.append(f'  <model name="{model_name}">')
    lines.append("    <static>true</static>")
    lines.append('    <link name="arrows_root">')
    lines.append("      <pose>0 0 0 0 0 0</pose>")
    lines.append("      <gravity>false</gravity>")
    for i, s in enumerate(samples):
        mag = s.mag
        roll, pitch, yaw = _pose_rpy_for_dir(s.u, s.v, s.w)
        r, g, b, a = pick_color(mag, vmin, vmax)
        L = max(mesh_len_min, min(mesh_len_max, mag * mesh_len_k))
        tw = mesh_thick
        lines.append(f'      <visual name="arrow_{i}_mesh">')
        lines.append(f"        <pose>{s.x:.3f} {s.y:.3f} {s.z:.3f} {roll:.6f} {pitch:.6f} {yaw:.6f}</pose>")
        lines.append("        <geometry>")
        lines.append("          <mesh>")
        lines.append(f"            <scale>{L:.6f} {tw:.6f} {tw:.6f}</scale>")
        lines.append(f"            <uri>{mesh_uri}</uri>")
        lines.append("          </mesh>")
        lines.append("        </geometry>")
        lines.append("        <material>")
        lines.append(f"          <ambient>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</ambient>")
        lines.append(f"          <diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.3f}</diffuse>")
        lines.append("        </material>")
        lines.append("        <cast_shadows>false</cast_shadows>")
        lines.append("      </visual>")
    lines.append("    </link>")
    lines.append("  </model>")
    lines.append("</sdf>")
    with open(os.path.join(model_dir, "model.sdf"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a static Gazebo Classic model containing wind arrows.")
    ap.add_argument(
        "--npz",
        default="data/wind_lut/20250903_1400/wind_lut.npz",
        help="Path to wind_lut.npz",
    )
    ap.add_argument("--model-name", default="wind_arrows_hotspot", help="SDF model name / model.config name")
    ap.add_argument(
        "--arrow-mesh-uri",
        default="model://wind_arrow_glyph/meshes/arrow_unit.stl",
        help="STL mesh URI for arrow glyph (must resolve on GAZEBO_MODEL_PATH)",
    )
    ap.add_argument(
        "--color-mode",
        choices=("legacy", "hsv"),
        default="legacy",
        help="legacy: RGB ramp; hsv: blue->red via HSV",
    )
    ap.add_argument("--center-x", type=float, default=1416.0)
    ap.add_argument("--center-y", type=float, default=-881.0)
    ap.add_argument(
        "--z-levels",
        default="50,145",
        help="Comma-separated z levels (meters). With multiple levels, XY grid is staggered per layer by default (see --no-z-xy-stagger).",
    )
    ap.add_argument(
        "--no-z-xy-stagger",
        action="store_true",
        help="Disable per-z-layer XY offset (restores stacked columns on the same horizontal grid).",
    )
    ap.add_argument("--nx", type=int, default=7)
    ap.add_argument("--ny", type=int, default=7)
    ap.add_argument("--dx", type=float, default=40.0, help="Sampling spacing in X (meters), centered mode")
    ap.add_argument("--dy", type=float, default=40.0, help="Sampling spacing in Y (meters), centered mode")
    ap.add_argument(
        "--bbox-mode",
        action="store_true",
        help="Use axis-aligned bbox grid (--x-min/--x-max/...) instead of centered grid",
    )
    ap.add_argument("--x-min", type=float, default=1050.0)
    ap.add_argument("--x-max", type=float, default=1850.0)
    ap.add_argument("--y-min", type=float, default=950.0)
    ap.add_argument("--y-max", type=float, default=1750.0)
    ap.add_argument(
        "--step",
        type=float,
        default=40.0,
        help="Grid step for bbox mode (meters); smaller => denser arrows",
    )
    ap.add_argument(
        "--mesh-len-min",
        type=float,
        default=10.0,
        help="Mesh arrow length L = clamp(k*|U|, min, max) along +X (meters)",
    )
    ap.add_argument("--mesh-len-max", type=float, default=36.0)
    ap.add_argument("--mesh-len-k", type=float, default=5.5, help="Length scale factor vs |U| (s*m^-1)")
    ap.add_argument(
        "--mesh-thick",
        type=float,
        default=3.2,
        help="Y/Z scale of unit STL (thickens shaft and head for visibility)",
    )
    ap.add_argument(
        "--out-model-dir",
        default="gazebo_wind_plugin/models/wind_arrows_hotspot",
        help="Output model directory",
    )
    args = ap.parse_args()

    z_levels = [float(x.strip()) for x in args.z_levels.split(",") if x.strip()]
    data = np.load(args.npz)
    U = data["U"]
    x_coords = data["x_coords"]
    y_coords = data["y_coords"]
    z_coords = data["z_coords"]

    if U.ndim != 4 or U.shape[-1] != 3:
        raise ValueError(f"Unexpected U shape: {U.shape}")

    z_xy_stagger = not args.no_z_xy_stagger

    if args.bbox_mode:
        samples = _generate_samples_bbox(
            U,
            x_coords,
            y_coords,
            z_coords,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            step=args.step,
            z_levels=z_levels,
            z_xy_stagger=z_xy_stagger,
        )
    else:
        samples = _generate_samples_centered(
            U,
            x_coords,
            y_coords,
            z_coords,
            center_x=args.center_x,
            center_y=args.center_y,
            z_levels=z_levels,
            nx=args.nx,
            ny=args.ny,
            dx=args.dx,
            dy=args.dy,
            z_xy_stagger=z_xy_stagger,
        )

    _write_model(
        args.out_model_dir,
        args.model_name,
        samples,
        color_mode=args.color_mode,
        mesh_uri=args.arrow_mesh_uri,
        mesh_len_min=args.mesh_len_min,
        mesh_len_max=args.mesh_len_max,
        mesh_len_k=args.mesh_len_k,
        mesh_thick=args.mesh_thick,
    )
    print(f"Wrote {len(samples)} arrows to {args.out_model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
