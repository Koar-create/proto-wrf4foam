#!/usr/bin/env python3
"""
OpenFOAM Domain & Refinement Region Visualizer
===============================================
Reads blockMeshDict / snappyHexMeshDict parameters and building geometry
from STL (3-D) with optional SHP footprint fallback, then generates a
reference-paper-style figure with:
  - Left  : 3-D perspective view (BC labels, x/y/z triad, y=0)
  - Top-right  : Plan view (X-Y)
  - Bottom-right: Elevation view (X-Z)

Usage:
    python3 scripts/visualize_openfoam_domain.py
    python3 scripts/visualize_openfoam_domain.py --no-titles

Default STL   : constant/triSurface/buildings.stl
Default SHP   : data/Guangzhou_shp_file/project_UTM49/Export_Output.shp
Default output: docs/project/openfoam_domain_visualization.png

Dependencies:
    pip install matplotlib numpy pyshp
"""

import argparse
import os
import struct

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapefile

matplotlib.rcParams["font.family"] = "DejaVu Sans"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STL = os.path.join(REPO_ROOT, "constant", "triSurface", "buildings.stl")
DEFAULT_SHP = os.path.join(
    REPO_ROOT, "data", "Guangzhou_shp_file", "project_UTM49", "Export_Output.shp"
)
DEFAULT_OUT = os.path.join(REPO_ROOT, "docs", "project", "openfoam_domain_visualization.png")

# Presentation-friendly font sizes (meeting-room / projector)
FS_SUITITLE = 22
FS_TITLE = 18
FS_LABEL = 15
FS_TICK = 13
FS_LEGEND = 14
FS_CORNER = 13
FS_DIM = 12
FS_BC = 12
FS_XYZ = 16

# ─────────────────────────────────────────────────────────────────────────────
# Domain parameters  (parsed from blockMeshDict / snappyHexMeshDict)
# ─────────────────────────────────────────────────────────────────────────────

# blockMeshDict: outer computational domain  [m]
DOMAIN = dict(xmin=-5000, xmax=5000, ymin=-5000, ymax=5000, zmin=0, zmax=2000)

# snappyHexMeshDict  ->  refinementRegions / refineBox  [m]
REFINE_BOX = dict(xmin=-2500, xmax=2500, ymin=-2500, ymax=2500, zmin=0, zmax=800)

# SHP geographic origin  (UTM centre of bounding box  ->  OpenFOAM origin)
SHP_CX, SHP_CY = 737789.45, 2557954.55

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C_DOMAIN = "#111111"
C_REFINE = "#1A56C4"
C_BUILD = "#3DAA55"
C_Y0 = "#FF4D00"
C_TOP = "#666666"
BG = "#FFFFFF"
GRID = "#DDDDDD"

# ─────────────────────────────────────────────────────────────────────────────
# Geometry loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_stl_triangles(stl_path, max_triangles=None):
    """Load STL triangles as (N, 3, 3) array in metres (OpenFOAM local coords)."""
    with open(stl_path, "rb") as f:
        header = f.read(80)
        if header.lstrip().lower().startswith(b"solid"):
            raise ValueError(f"ASCII STL not supported: {stl_path}")

        n_tri = struct.unpack("<I", f.read(4))[0]
        step = 1
        if max_triangles and n_tri > max_triangles:
            step = int(np.ceil(n_tri / max_triangles))

        tris = []
        for i in range(n_tri):
            data = f.read(50)
            if i % step:
                continue
            tri = np.array([
                struct.unpack("<3f", data[12:24]),
                struct.unpack("<3f", data[24:36]),
                struct.unpack("<3f", data[36:48]),
            ], dtype=np.float64)
            tris.append(tri)

    arr = np.asarray(tris, dtype=np.float64)
    print(f"[info] Loaded {len(arr)} / {n_tri} STL triangles from {stl_path}")
    return arr


def load_footprints(shp_path):
    polys = []
    try:
        with open(shp_path, "rb") as f:
            sf = shapefile.Reader(shp=f)
            for shape in sf.shapes():
                if len(shape.points) < 3:
                    continue
                pts = np.array([[p[0] - SHP_CX, p[1] - SHP_CY]
                                for p in shape.points])
                polys.append(pts)
        print(f"[info] Loaded {len(polys)} building polygons from {shp_path}")
    except Exception as e:
        print(f"[warn] Could not read SHP: {e}")
    return polys


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def rect2d(ax, x0, x1, y0, y1, ec, lw=1.8, ls="-", fc="none", label=None, zorder=3):
    r = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                          linewidth=lw, edgecolor=ec, facecolor=fc,
                          linestyle=ls, label=label, zorder=zorder)
    ax.add_patch(r)


def box3d(ax, x0, x1, y0, y1, z0, z1, color, lw=1.6, ls="-", alpha=1.0):
    corners = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        ax.plot([corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color=color, lw=lw, ls=ls, alpha=alpha, zorder=4)


def draw_arrow3d(ax, p0, p1, color, shaft_lw=4.5, head_len=0.42, head_width=0.22):
    """Thick 3-D shaft + triangular head (more reliable than FancyArrowPatch)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return
    d = vec / length
    head_len = min(head_len, 0.45 * length)
    shaft_end = p1 - d * head_len
    ax.plot(
        [p0[0], shaft_end[0]], [p0[1], shaft_end[1]], [p0[2], shaft_end[2]],
        color=color, lw=shaft_lw, solid_capstyle="round", zorder=9,
    )
    n_xy = np.array([-d[1], d[0], 0.0])
    n_norm = np.linalg.norm(n_xy)
    if n_norm < 1e-8:
        n_xy = np.array([1.0, 0.0, 0.0])
    else:
        n_xy = n_xy / n_norm
    n_xy = n_xy * head_width
    n_z = np.array([0.0, 0.0, head_width * 0.55])
    verts = [
        [p1.tolist(), (shaft_end + n_xy).tolist(), (shaft_end - n_xy).tolist()],
        [p1.tolist(), (shaft_end + n_z).tolist(), (shaft_end - n_z).tolist()],
    ]
    ax.add_collection3d(
        Poly3DCollection(verts, facecolor=color, edgecolor=color, linewidths=0.4, zorder=9)
    )


def add_bc_annotations(ax, D):
    """Face labels and Top (slip) callout on the 3-D panel."""
    kw = dict(fontsize=FS_BC, fontweight="bold", color="#111111",
              ha="center", va="center", zorder=10, clip_on=False)
    # Near faces (camera azim=-50 looks from +x, -y): inflow at low z.
    ax.text(0.0, D["ymin"] - 0.12, 0.18, "south (inflow)", **kw)
    ax.text(D["xmax"] + 0.12, 0.0, 0.18, "east (inflow)", **kw)
    # Far faces: outflow at high z.
    ax.text(D["xmin"] - 0.08, 0.0, D["zmax"] - 0.12, "west (outflow)", **kw)
    ax.text(0.0, D["ymax"] + 0.08, D["zmax"] - 0.12, "north (outflow)", **kw)

    top_pt = (0.0, 0.0, D["zmax"])
    text_pt = (4.1, 4.4, D["zmax"] + 0.28)
    ax.scatter(
        [top_pt[0]], [top_pt[1]], [top_pt[2]],
        color=C_TOP, s=36, depthshade=False, zorder=11,
    )
    ax.plot(
        [top_pt[0], text_pt[0]], [top_pt[1], text_pt[1]], [top_pt[2], text_pt[2]],
        color=C_TOP, lw=1.4, zorder=10,
    )
    ax.text(
        text_pt[0], text_pt[1], text_pt[2], "Top (slip)",
        fontsize=FS_BC, fontweight="bold", color="#111111",
        ha="left", va="bottom", zorder=11, clip_on=False,
    )


def add_xyz_triad(ax, D):
    """Small x/y/z triad at the visible south-west ground corner."""
    ox = D["xmin"] + 0.55
    oy = D["ymin"] + 0.55
    oz = 0.0
    lx, ly, lz = 1.55, 1.55, 0.70
    draw_arrow3d(ax, [ox, oy, oz], [ox + lx, oy, oz], "#111111",
                 shaft_lw=2.2, head_len=0.32, head_width=0.16)
    draw_arrow3d(ax, [ox, oy, oz], [ox, oy + ly, oz], "#111111",
                 shaft_lw=2.2, head_len=0.32, head_width=0.16)
    draw_arrow3d(ax, [ox, oy, oz], [ox, oy, oz + lz], "#111111",
                 shaft_lw=2.2, head_len=0.18, head_width=0.12)
    tkw = dict(fontsize=FS_XYZ, fontweight="bold", color="#111111",
               zorder=12, clip_on=False)
    ax.text(ox + lx + 0.12, oy, oz, "x", ha="left", va="center", **tkw)
    ax.text(ox, oy + ly + 0.12, oz, "y", ha="center", va="bottom", **tkw)
    ax.text(ox, oy, oz + lz + 0.10, "z", ha="center", va="bottom", **tkw)


def add_y0_marker(ax3d, ax_xy, D):
    """Conspicuous y = 0 (Pearl River corridor) on the 3-D and plan panels."""
    x0, x1 = D["xmin"], D["xmax"]
    z0, z1 = D["zmin"], D["zmax"]
    ax3d.plot([x0, x1], [0.0, 0.0], [z0, z0], color=C_Y0, lw=2.6, zorder=8)
    ax3d.plot([x0, x1], [0.0, 0.0], [z1, z1], color=C_Y0, lw=1.8, ls="--", zorder=8)
    ax3d.plot([x0, x0], [0.0, 0.0], [z0, z1], color=C_Y0, lw=1.8, zorder=8)
    ax3d.plot([x1, x1], [0.0, 0.0], [z0, z1], color=C_Y0, lw=1.8, zorder=8)
    ax3d.text(
        x1 + 0.18, 0.0, 0.55, "y = 0",
        color=C_Y0, fontsize=FS_BC + 1, fontweight="bold",
        ha="left", va="center", zorder=12, clip_on=False,
    )

    ax_xy.plot(
        [x0, x1], [0.0, 0.0], color=C_Y0, lw=2.4, zorder=7, solid_capstyle="butt",
    )
    ax_xy.text(
        x1 + 0.08, 0.0, "y = 0",
        color=C_Y0, fontsize=FS_CORNER, fontweight="bold",
        ha="left", va="center", zorder=8,
    )


def add_stl_to_axes(stl_tris, ax3d, ax_xy, ax_xz):
    """Render STL buildings on 3-D, plan, and elevation panels."""
    tris_km = stl_tris / 1000.0

    coll3d = Poly3DCollection(
        tris_km, alpha=0.55, facecolor=C_BUILD, edgecolor="none", linewidths=0, zorder=2
    )
    ax3d.add_collection3d(coll3d)

    plan_polys = tris_km[:, :, :2]
    ax_xy.add_collection(
        PolyCollection(plan_polys, facecolor=C_BUILD, edgecolor="none", alpha=0.55, zorder=2)
    )

    elev_polys = tris_km[:, :, [0, 2]]
    ax_xz.add_collection(
        PolyCollection(elev_polys, facecolor=C_BUILD, edgecolor="none", alpha=0.55, zorder=2)
    )


def add_footprints_to_axes(footprints, ax3d, ax_xy):
    """Fallback: flat SHP footprints when STL is unavailable."""
    if not footprints:
        return

    step = max(1, len(footprints) // 1200)
    verts = []
    for poly in footprints[::step]:
        xs = poly[:, 0] / 1000
        ys = poly[:, 1] / 1000
        verts.append(list(zip(xs, ys, np.zeros(len(xs)))))
    ax3d.add_collection3d(
        Poly3DCollection(verts, alpha=0.45, facecolor=C_BUILD, edgecolor="none", zorder=2)
    )

    plist = [MplPolygon(p / 1000, closed=True) for p in footprints]
    ax_xy.add_collection(
        PatchCollection(plist, facecolor=C_BUILD, edgecolor="none", alpha=0.55, zorder=2)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build figure
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(stl_tris=None, footprints=None, show_titles=True):
    top = 0.90 if show_titles else 0.96
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.subplots_adjust(left=0.04, right=0.98, top=top, bottom=0.07,
                        wspace=0.32, hspace=0.42)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d", facecolor=BG)
    ax_xy = fig.add_subplot(2, 2, 2, facecolor=BG)
    ax_xz = fig.add_subplot(2, 2, 4, facecolor=BG)

    def km(d):
        return {k: v / 1000 for k, v in d.items()}

    D = km(DOMAIN)
    RB = km(REFINE_BOX)

    # ══════════════════════════════════════════════════════════════════════
    # (A) 3-D perspective
    # ══════════════════════════════════════════════════════════════════════
    ax3d.set_box_aspect([10, 10, 4])

    box3d(ax3d, D["xmin"], D["xmax"], D["ymin"], D["ymax"],
          D["zmin"], D["zmax"], C_DOMAIN, lw=1.8)
    box3d(ax3d, RB["xmin"], RB["xmax"], RB["ymin"], RB["ymax"],
          RB["zmin"], RB["zmax"], C_REFINE, lw=1.5, ls=(0, (5, 3)))

    if stl_tris is not None:
        add_stl_to_axes(stl_tris, ax3d, ax_xy, ax_xz)
    else:
        add_footprints_to_axes(footprints or [], ax3d, ax_xy)

    add_bc_annotations(ax3d, D)
    add_xyz_triad(ax3d, D)
    add_y0_marker(ax3d, ax_xy, D)

    ax3d.set_xlabel("x   Easting [km]", fontsize=FS_LABEL, labelpad=8)
    ax3d.set_ylabel("y   Northing [km]", fontsize=FS_LABEL, labelpad=8)
    ax3d.set_zlabel("z   Height [km]", fontsize=FS_LABEL, labelpad=8)
    if show_titles:
        ax3d.set_title("3D View", fontsize=FS_TITLE, fontweight="bold", pad=10)
    ax3d.view_init(elev=25, azim=-50)
    ax3d.tick_params(labelsize=FS_TICK)
    ax3d.set_xlim(D["xmin"] - 0.6, D["xmax"] + 0.8)
    ax3d.set_ylim(D["ymin"] - 0.6, D["ymax"] + 0.6)
    ax3d.set_zlim(D["zmin"], D["zmax"] + 0.40)

    # ══════════════════════════════════════════════════════════════════════
    # (B) Plan view  X-Y
    # ══════════════════════════════════════════════════════════════════════
    rect2d(ax_xy, D["xmin"], D["xmax"], D["ymin"], D["ymax"],
           C_DOMAIN, lw=2.0, label="Computational Domain", zorder=4)
    rect2d(ax_xy, RB["xmin"], RB["xmax"], RB["ymin"], RB["ymax"],
           C_REFINE, lw=1.8, ls="--", label="Refinement Box", zorder=5)

    off = 0.08
    if show_titles:
        ax_xy.text(D["xmin"] + off, D["ymax"] - off, "domain",
                   color=C_DOMAIN, fontsize=FS_CORNER, fontweight="bold", va="top", zorder=6)
        ax_xy.text(RB["xmin"] + off, RB["ymax"] - off, "refineBox",
                   color=C_REFINE, fontsize=FS_CORNER, fontweight="bold", va="top", zorder=6)

    ax_xy.set_xlim(D["xmin"] - 0.35, D["xmax"] + 0.85)
    ax_xy.set_ylim(D["ymin"] - 0.35, D["ymax"] + 0.35)
    ax_xy.set_xlabel("x   Easting [km]", fontsize=FS_LABEL)
    ax_xy.set_ylabel("y   Northing [km]", fontsize=FS_LABEL)
    if show_titles:
        ax_xy.set_title("(a)  Plan View (X\u2013Y)", fontsize=FS_TITLE, fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, lw=0.4, color=GRID, zorder=0)
    ax_xy.tick_params(labelsize=FS_TICK)

    # ══════════════════════════════════════════════════════════════════════
    # (C) Elevation  X-Z
    # ══════════════════════════════════════════════════════════════════════
    rect2d(ax_xz, D["xmin"], D["xmax"], D["zmin"], D["zmax"],
           C_DOMAIN, lw=2.0, label="Computational Domain", zorder=4)
    rect2d(ax_xz, RB["xmin"], RB["xmax"], RB["zmin"], RB["zmax"],
           C_REFINE, lw=1.8, ls="--", label="Refinement Box", zorder=5)

    if show_titles:
        ax_xz.text(D["xmin"] + off, D["zmax"] - 0.015, "domain",
                   color=C_DOMAIN, fontsize=FS_CORNER, fontweight="bold", va="top", zorder=6)
        ax_xz.text(RB["xmin"] + off, RB["zmax"] - 0.015, "refineBox",
                   color=C_REFINE, fontsize=FS_CORNER, fontweight="bold", va="top", zorder=6)

    ax_xz.set_xlim(D["xmin"] - 0.35, D["xmax"] + 0.35)
    ax_xz.set_ylim(-0.05, D["zmax"] + 0.12)
    ax_xz.set_xlabel("x   Easting [km]", fontsize=FS_LABEL)
    ax_xz.set_ylabel("z   Height [km]", fontsize=FS_LABEL)
    if show_titles:
        ax_xz.set_title("(b)  Elevation View (X\u2013Z)", fontsize=FS_TITLE, fontweight="bold", pad=20)
    ax_xz.grid(True, lw=0.4, color=GRID, zorder=0)
    ax_xz.tick_params(labelsize=FS_TICK)

    # ══════════════════════════════════════════════════════════════════════
    # Legend + title
    # ══════════════════════════════════════════════════════════════════════
    build_label = "Buildings (STL)" if stl_tris is not None else "Buildings (SHP footprints)"
    leg_handles = [
        Line2D([0], [0], color=C_DOMAIN, lw=2.0, label="Computational Domain"),
        Line2D([0], [0], color=C_REFINE, lw=1.8, ls="--", label="Refinement Box"),
        patches.Patch(facecolor=C_BUILD, alpha=0.6, label=build_label),
    ]
    legend_y = 0.985 if show_titles else 0.995
    fig.legend(handles=leg_handles, loc="upper center", ncol=3,
               fontsize=FS_LEGEND, frameon=True, framealpha=0.92,
               bbox_to_anchor=(0.5, legend_y),
               edgecolor="#aaaaaa", fancybox=False)

    if show_titles:
        fig.suptitle("OpenFOAM  Computational Domain  &  Refinement Regions",
                     fontsize=FS_SUITITLE, fontweight="bold", y=1.04)

    # Dimension annotations on elevation
    def annotate_dim(ax, x0, x1, y, label, color, ytext=None):
        ytext = ytext if ytext else y
        ax.annotate("", xy=(x1, ytext), xytext=(x0, ytext),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.2),
                    zorder=7)
        ax.text((x0 + x1) / 2, ytext + 0.025, label,
                color=color, fontsize=FS_DIM, ha="center", va="bottom", zorder=7)

    annotate_dim(ax_xz, D["xmin"], D["xmax"], D["zmax"] + 0.06,
                 f'{DOMAIN["xmax"] - DOMAIN["xmin"]} m', C_DOMAIN)
    annotate_dim(ax_xz, RB["xmin"], RB["xmax"], RB["zmax"] + 0.03,
                 f'{REFINE_BOX["xmax"] - REFINE_BOX["xmin"]} m', C_REFINE)

    ax_xz.annotate("", xy=(D["xmax"] + 0.15, D["zmax"]),
                   xytext=(D["xmax"] + 0.15, D["zmin"]),
                   arrowprops=dict(arrowstyle="<->", color=C_DOMAIN, lw=1.2),
                   zorder=7)
    ax_xz.text(D["xmax"] + 0.22, D["zmax"] / 2,
               f'{DOMAIN["zmax"]} m', color=C_DOMAIN, fontsize=FS_DIM,
               ha="left", va="center", rotation=90, zorder=7)

    ax_xz.annotate("", xy=(RB["xmax"] + 0.07, RB["zmax"]),
                   xytext=(RB["xmax"] + 0.07, RB["zmin"]),
                   arrowprops=dict(arrowstyle="<->", color=C_REFINE, lw=1.2),
                   zorder=7)
    ax_xz.text(RB["xmax"] + 0.14, RB["zmax"] / 2,
               f'{REFINE_BOX["zmax"]} m', color=C_REFINE, fontsize=FS_DIM,
               ha="left", va="center", rotation=90, zorder=7)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize OpenFOAM domain and refinement regions.")
    p.add_argument("--stl", default=DEFAULT_STL, help=f"Building STL (default: {DEFAULT_STL})")
    p.add_argument("--shp", default=DEFAULT_SHP, help=f"Footprint SHP fallback (default: {DEFAULT_SHP})")
    p.add_argument("-o", "--output", default=DEFAULT_OUT, help=f"Output PNG (default: {DEFAULT_OUT})")
    p.add_argument("--no-titles", action="store_true",
                   help="Hide figure title, subplot titles, and corner labels")
    p.add_argument("--max-triangles", type=int, default=None,
                   help="Optional cap on STL triangles for faster preview")
    return p.parse_args()


def main():
    args = parse_args()
    show_titles = not args.no_titles

    stl_tris = None
    if os.path.isfile(args.stl):
        try:
            stl_tris = load_stl_triangles(args.stl, max_triangles=args.max_triangles)
        except Exception as e:
            print(f"[warn] Could not read STL: {e}")
    else:
        print(f"[warn] STL not found: {args.stl}")

    footprints = []
    if stl_tris is None and os.path.isfile(args.shp):
        footprints = load_footprints(args.shp)
    elif stl_tris is None:
        print(f"[warn] SHP not found: {args.shp}")

    fig = make_figure(stl_tris=stl_tris, footprints=footprints, show_titles=show_titles)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[done] Saved -> {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
