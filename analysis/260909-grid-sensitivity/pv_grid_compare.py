"""Three-mesh ParaView comparison for the 2025-09-01 18:00 (UTC+8) snapshot.

Run with paraview-env (offscreen):

  /hpc2hdd/home/zyang248/miniconda3/envs/paraview-env/bin/pvbatch \\
      analysis/260909-grid-sensitivity/pv_grid_compare.py

Outputs under results/grid_sensitivity/paraview/  (never /tmp).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from paraview.simple import *  # noqa: F403

paraview.simple._DisableFirstRenderCameraReset()

REPO = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling")
OUT = REPO / "results" / "grid_sensitivity" / "paraview"
ABL = REPO / "steady_experiments_finer_ABL"
SENS = REPO / "experiments" / "sensitivity_experiments"
STL = (
    ABL
    / "20250901_0000_two_boundaries_as_outlet"
    / "constant"
    / "triSurface"
    / "buildings.stl"
)
CASES = [
    {
        "label": "M1 3.78M",
        "key": "M1",
        "foam": SENS / "20250901_1000_two_boundaries_as_outlet-grid_3.0M" / "myExpxx.foam",
        "time": 4850.0,
    },
    {
        "label": "M2 5.47M",
        "key": "M2",
        "foam": ABL / "20250901_1000_two_boundaries_as_outlet" / "myExpxx.foam",
        "time": 5000.0,
    },
    {
        "label": "M3 8.71M",
        "key": "M3",
        "foam": SENS / "20250901_1000_two_boundaries_as_outlet-grid_8.0M" / "myExpxx.foam",
        "time": 5000.0,
    },
]
SITES = {
    "GAW103": (975.0, -320.0),
    "GAW104": (450.0, 350.0),
    "GAW111": (75.0, 30.0),
}
# Dense north-bank cluster around GAW104 (mesh close-up).
MESH_BOX = (250.0, 650.0, 150.0, 550.0, 0.0, 220.0)
UMAX = 4.0
Z_SLICE = 60.0
Y_SLICE = -320.0  # through GAW103
# Vertical-slice framing for matplotlib (x–z through GAW103).
# Tightened around the south-bank building strip + LiDAR column.
UY_XMIN, UY_XMAX = 200.0, 1500.0
UY_ZMIN, UY_ZMAX = 0.0, 300.0
UY_NX, UY_NZ = 520, 240
UY_STL_DY = 120.0  # project buildings with |y+320|<this onto the elevation
UY_BUILDING_NOTE = r"buildings: $|y+320|\!\leq\!120\,\mathrm{m}$ projected"


def _try_set(obj, attr, value):
    try:
        setattr(obj, attr, value)
    except Exception:
        pass


def _region_names(reader):
    for getter in (
        lambda: list(reader.MeshRegions.Available),
        lambda: list(reader.GetProperty("MeshRegions").Available),
    ):
        try:
            names = getter()
            if names:
                return [str(n) for n in names]
        except Exception:
            pass
    return []


def _pick_region(names, *needles):
    lower = [(n, n.lower()) for n in names]
    for needle in needles:
        for n, nl in lower:
            if needle in nl:
                return n
    return None


def _resolve_time(reader, requested):
    avail = list(reader.TimestepValues) if reader.TimestepValues else []
    if not avail:
        return requested
    if requested in avail:
        return requested
    closest = min(avail, key=lambda t: abs(t - requested))
    print(f"[warn] t={requested} not in {avail}; using {closest}")
    return closest


def _load(foam, time, regions, arrays):
    src = OpenFOAMReader(FileName=str(foam))
    UpdatePipeline(proxy=src)
    names = _region_names(src)
    print(f"[info] regions: {names}")
    chosen = []
    for spec in regions:
        if spec == "internal":
            r = _pick_region(names, "internalmesh", "internal")
        elif spec == "buildings":
            r = _pick_region(names, "buildings")
        else:
            r = spec if spec in names else None
        if r:
            chosen.append(r)
        else:
            print(f"[warn] no region matching {spec!r}")
    if chosen:
        src.MeshRegions = chosen
    if arrays:
        src.CellArrays = arrays
    else:
        try:
            src.CellArrays = []
        except Exception:
            pass
    UpdatePipeline(proxy=src)
    t = _resolve_time(src, time)
    UpdatePipeline(t, proxy=src)
    return src, t


def _make_view(width, height):
    view = CreateView("RenderView")
    SetActiveView(view)
    view.ViewSize = [width, height]
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 1
    _try_set(view, "OrientationAxesLabelColor", [0.15, 0.15, 0.15])
    for bg_mode in ("Single Color", "Background Color"):
        try:
            view.BackgroundColorMode = bg_mode
            break
        except Exception:
            pass
    _try_set(view, "UseColorPaletteForBackground", 0)
    return view


def _apply_jet_mag(display, vmin, vmax):
    try:
        ColorBy(display, ("CELLS", "U", "Magnitude"))
    except Exception:
        ColorBy(display, ("POINTS", "U", "Magnitude"))
    lut = GetColorTransferFunction("U")
    lut.ApplyPreset("Jet", True)
    lut.RescaleTransferFunction(vmin, vmax)
    pwf = GetOpacityTransferFunction("U")
    pwf.RescaleTransferFunction(vmin, vmax)
    return lut


def _add_colorbar(lut, view, title):
    sb = GetScalarBar(lut, view)
    sb.Title = title
    sb.Visibility = 1
    sb.Orientation = "Vertical"
    for loc in ("Lower Right Corner", "LowerRightCorner"):
        try:
            sb.WindowLocation = loc
            break
        except Exception:
            pass
    _try_set(sb, "ComponentTitle", "")
    _try_set(sb, "TitleFontSize", 16)
    _try_set(sb, "LabelFontSize", 12)
    _try_set(sb, "ScalarBarLength", 0.35)
    return sb


def _clip_box(inp, t, bounds, invert=1):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    cl = Clip(Input=inp)
    cl.ClipType = "Box"
    cl.Invert = invert
    try:
        cl.ClipType.Bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    except Exception:
        cl.ClipType.Position = [
            0.5 * (xmin + xmax),
            0.5 * (ymin + ymax),
            0.5 * (zmin + zmax),
        ]
        cl.ClipType.Scale = [
            0.5 * (xmax - xmin),
            0.5 * (ymax - ymin),
            0.5 * (zmax - zmin),
        ]
    UpdatePipeline(t, proxy=cl)
    n = cl.GetDataInformation().GetNumberOfCells()
    if n == 0:
        Delete(cl)
        cl = Clip(Input=inp)
        cl.ClipType = "Box"
        cl.Invert = 1 - invert
        try:
            cl.ClipType.Bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
        except Exception:
            pass
        UpdatePipeline(t, proxy=cl)
    return cl


def _mark_sites(view, z, radius=35.0):
    for name, (x, y) in SITES.items():
        sph = Sphere()
        sph.Center = [x, y, z]
        sph.Radius = radius
        sph.ThetaResolution = 16
        sph.PhiResolution = 16
        disp = Show(sph, view)
        disp.Representation = "Surface"
        disp.AmbientColor = [0.05, 0.05, 0.05]
        disp.DiffuseColor = [0.05, 0.05, 0.05]


def _save(view, path, width, height):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    Render(view)
    SaveScreenshot(
        str(path),
        view,
        ImageResolution=[width, height],
        OverrideColorPalette="",
        TransparentBackground=0,
    )
    print(f"[save] {path}")


def _reset():
    Disconnect()
    Connect()
    paraview.simple._DisableFirstRenderCameraReset()


def render_mesh(case):
    """Building surfaces + z=60 volume-mesh edges around GAW104."""
    w, h = 1400, 1200
    view = _make_view(w, h)
    src, t = _load(case["foam"], case["time"], ["internal"], [])

    sl = Slice(Input=src)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [0.0, 0.0, 1.0]
    sl.SliceType.Origin = [450.0, 350.0, 40.0]
    sl.Triangulatetheslice = 0
    UpdatePipeline(t, proxy=sl)
    slc = _clip_box(sl, t, MESH_BOX)
    sld = Show(slc, view)
    sld.Representation = "Surface With Edges"
    sld.EdgeColor = [0.15, 0.15, 0.15]
    sld.AmbientColor = [0.82, 0.90, 0.95]
    sld.DiffuseColor = [0.82, 0.90, 0.95]
    try:
        sld.SetScalarBarVisibility(view, False)
        ColorBy(sld, None)
    except Exception:
        pass
    _try_set(sld, "MapScalars", 0)

    bsrc, _ = _load(case["foam"], case["time"], ["buildings"], [])
    bclip = _clip_box(bsrc, t, MESH_BOX)
    bd = Show(bclip, view)
    bd.Representation = "Surface With Edges"
    bd.EdgeColor = [0.05, 0.05, 0.05]
    bd.AmbientColor = [0.55, 0.62, 0.48]
    bd.DiffuseColor = [0.55, 0.62, 0.48]
    try:
        ColorBy(bd, None)
    except Exception:
        pass

    _mark_sites(view, 80.0, radius=18.0)
    view.CameraPosition = [780.0, 20.0, 420.0]
    view.CameraFocalPoint = [450.0, 350.0, 40.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = 0
    view.OrientationAxesVisibility = 1
    _save(view, OUT / f"panel_mesh_{case['key']}.png", w, h)


def render_uz(case):
    """Top-down |U| at z=60 m, urban core ±2 km."""
    w, h = 1400, 1400
    view = _make_view(w, h)
    src, t = _load(case["foam"], case["time"], ["internal"], ["U"])
    sl = Slice(Input=src)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [0.0, 0.0, 1.0]
    sl.SliceType.Origin = [0.0, 0.0, Z_SLICE]
    UpdatePipeline(t, proxy=sl)
    slc = _clip_box(sl, t, (-2000.0, 2000.0, -2000.0, 2000.0, Z_SLICE - 5.0, Z_SLICE + 5.0))
    disp = Show(slc, view)
    disp.Representation = "Surface"
    lut = _apply_jet_mag(disp, 0.0, UMAX)
    _add_colorbar(lut, view, "|U| (m s$^{-1}$)")
    _mark_sites(view, Z_SLICE + 8.0, radius=28.0)

    if STL.is_file():
        bld = STLReader(FileNames=[str(STL)])
        UpdatePipeline(proxy=bld)
        bclip = _clip_box(bld, t, (-2000.0, 2000.0, -2000.0, 2000.0, -20.0, 400.0))
        bd = Show(bclip, view)
        bd.Representation = "Surface"
        bd.AmbientColor = [0.35, 0.35, 0.35]
        bd.DiffuseColor = [0.35, 0.35, 0.35]
        bd.Opacity = 1.0

    view.CameraPosition = [0.0, 0.0, 4200.0]
    view.CameraFocalPoint = [0.0, 0.0, Z_SLICE]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = 2100.0
    view.OrientationAxesVisibility = 1
    _save(view, OUT / f"panel_Uz60_{case['key']}.png", w, h)

    rti = ResampleToImage(Input=slc)
    rti.UseInputBounds = 0
    rti.SamplingDimensions = [400, 400, 1]
    rti.SamplingBounds = [-2000.0, 2000.0, -2000.0, 2000.0, Z_SLICE, Z_SLICE]
    UpdatePipeline(t, proxy=rti)
    csv_path = OUT / f"Uz60_{case['key']}.csv"
    SaveData(str(csv_path), proxy=rti)
    print(f"[save] {csv_path}")


def render_uy(case):
    """Export |U| on y = -320 m to a regular x–z CSV for matplotlib panels.

    ParaView screenshots are not the primary product here: axes, buildings, and
    a shared colorbar are drawn later in plot_uy_matplotlib().
    """
    src, t = _load(case["foam"], case["time"], ["internal"], ["U"])
    c2p = CellDatatoPointData(Input=src)
    c2p.ProcessAllArrays = 1
    UpdatePipeline(t, proxy=c2p)

    sl = Slice(Input=c2p)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [0.0, 1.0, 0.0]
    sl.SliceType.Origin = [SITES["GAW103"][0], Y_SLICE, 150.0]
    UpdatePipeline(t, proxy=sl)

    rti = ResampleToImage(Input=sl)
    rti.UseInputBounds = 0
    rti.SamplingDimensions = [UY_NX, 1, UY_NZ]
    rti.SamplingBounds = [
        UY_XMIN, UY_XMAX,
        Y_SLICE, Y_SLICE,
        UY_ZMIN, UY_ZMAX,
    ]
    UpdatePipeline(t, proxy=rti)
    csv_path = OUT / f"Uy-320_{case['key']}.csv"
    SaveData(str(csv_path), proxy=rti)
    print(f"[save] {csv_path}")


def _load_uy_mag(key):
    """Load Uy-320 resample CSV → (xx, zz, |U|) with shape (NZ, NX)."""
    import csv
    import numpy as np

    p = OUT / f"Uy-320_{key}.csv"
    if not p.exists():
        return None
    with p.open() as f:
        reader = csv.DictReader(f)
        u0s, u1s, u2s, mask = [], [], [], []
        for row in reader:
            u0s.append(float(row["U:0"]))
            u1s.append(float(row["U:1"]))
            u2s.append(float(row.get("U:2", "0") or 0.0))
            mask.append(float(row.get("vtkValidPointMask", "1")))
    v = np.sqrt(np.asarray(u0s) ** 2 + np.asarray(u1s) ** 2 + np.asarray(u2s) ** 2)
    m = np.asarray(mask) > 0.5
    v = np.where(m, v, np.nan)
    # ResampleToImage order for dims [nx,1,nz] is typically z-fastest or x-fastest.
    # Prefer reshape that matches SamplingDimensions; ParaView writes x varying fastest.
    if len(v) != UY_NX * UY_NZ:
        print(f"[warn] Uy-320_{key} n={len(v)} != {UY_NX*UY_NZ}")
        return None
    # VTK image: i (x) fastest, then j (y=1), then k (z) → reshape (nz, nx)
    arr = v.reshape(UY_NZ, UY_NX)
    x1d = np.linspace(UY_XMIN, UY_XMAX, UY_NX)
    z1d = np.linspace(UY_ZMIN, UY_ZMAX, UY_NZ)
    xx, zz = np.meshgrid(x1d, z1d)
    return xx, zz, arr


def _stl_xz_segments(stl_path, y0):
    """Intersect buildings.stl with plane y=y0 → list of ((x1,z1),(x2,z2))."""
    import numpy as np
    import struct

    stl_path = Path(stl_path)
    if not stl_path.is_file():
        return []

    def _edge_hit(p, q):
        dy = q[1] - p[1]
        if abs(dy) < 1.0e-12:
            return None
        if (p[1] - y0) * (q[1] - y0) > 0.0:
            return None
        t = (y0 - p[1]) / dy
        if t < -1.0e-9 or t > 1.0 + 1.0e-9:
            return None
        t = min(max(t, 0.0), 1.0)
        pt = p + t * (q - p)
        return (float(pt[0]), float(pt[2]))

    segs = []
    with stl_path.open("rb") as f:
        header = f.read(80)
        if header.lstrip().lower().startswith(b"solid"):
            print(f"[warn] ASCII STL not supported: {stl_path}")
            return []
        n_tri = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_tri):
            data = f.read(50)
            tri = np.array(
                [
                    struct.unpack("<3f", data[12:24]),
                    struct.unpack("<3f", data[24:36]),
                    struct.unpack("<3f", data[36:48]),
                ],
                dtype=np.float64,
            )
            ymin = float(tri[:, 1].min())
            ymax = float(tri[:, 1].max())
            if ymin > y0 or ymax < y0:
                continue
            hits = []
            for a, b in ((0, 1), (1, 2), (2, 0)):
                h = _edge_hit(tri[a], tri[b])
                if h is not None:
                    hits.append(h)
            # Deduplicate near-identical hits (vertex on plane).
            uniq = []
            for h in hits:
                if all(abs(h[0] - u[0]) > 0.05 or abs(h[1] - u[1]) > 0.05 for u in uniq):
                    uniq.append(h)
            if len(uniq) >= 2:
                segs.append((uniq[0], uniq[1]))
    print(f"[info] STL∩y={y0}: {len(segs)} segments")
    return segs


def _stl_xz_building_polys(stl_path, y0, dy=120.0, z_ground=0.0):
    """Side-elevation schematic: project buildings with |y-y0|<dy onto x–z.

    Exact plane intersection at GAW103's y misses the nearby high-rises (their
    footprints do not cross y=-320). A ±dy slab projection is the useful
    building context for this transect.
    """
    import numpy as np
    import struct

    stl_path = Path(stl_path)
    if not stl_path.is_file():
        return [], []

    # Exact intersection segments (thin lines on the true plane).
    segs = _stl_xz_segments(stl_path, y0)

    # Skyline envelope from nearby buildings.
    nx = 500
    xedges = np.linspace(UY_XMIN, UY_XMAX, nx + 1)
    hmax = np.zeros(nx, dtype=float)
    with stl_path.open("rb") as f:
        header = f.read(80)
        if header.lstrip().lower().startswith(b"solid"):
            print(f"[warn] ASCII STL not supported: {stl_path}")
            return [], segs
        n_tri = struct.unpack("<I", f.read(4))[0]
        for _ in range(n_tri):
            data = f.read(50)
            tri = np.array(
                [
                    struct.unpack("<3f", data[12:24]),
                    struct.unpack("<3f", data[24:36]),
                    struct.unpack("<3f", data[36:48]),
                ],
                dtype=np.float64,
            )
            if abs(float(tri[:, 1].mean()) - y0) > dy:
                continue
            ztop = float(tri[:, 2].max())
            if ztop <= z_ground + 1.0:
                continue
            xa = float(tri[:, 0].min())
            xb = float(tri[:, 0].max())
            if xb < UY_XMIN or xa > UY_XMAX:
                continue
            i0 = max(int(np.searchsorted(xedges, xa, side="right") - 1), 0)
            i1 = min(int(np.searchsorted(xedges, xb, side="left")), nx - 1)
            if i1 < i0:
                continue
            hmax[i0 : i1 + 1] = np.maximum(hmax[i0 : i1 + 1], ztop)

    polys = []
    # Merge contiguous runs of similar height into rectangles.
    i = 0
    while i < nx:
        if hmax[i] <= z_ground + 1.0:
            i += 1
            continue
        j = i
        href = hmax[i]
        while j + 1 < nx and abs(hmax[j + 1] - href) < 1.0 and hmax[j + 1] > z_ground + 1.0:
            j += 1
            href = max(href, hmax[j])
        polys.append(
            np.array(
                [
                    [xedges[i], z_ground],
                    [xedges[j + 1], z_ground],
                    [xedges[j + 1], href],
                    [xedges[i], href],
                ],
                dtype=float,
            )
        )
        i = j + 1
    print(
        f"[info] building skyline polys |y-{y0}|<{dy}: {len(polys)} "
        f"(hmax={hmax.max():.1f} m); plane segments={len(segs)}"
    )
    return polys, segs


def _draw_uy_panel(ax, xx, zz, u, stl_polys, stl_segs, title, show_ylabel=True):
    import numpy as np
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.colors import ListedColormap

    pcm = ax.pcolormesh(
        xx, zz, np.ma.masked_invalid(u),
        shading="auto", cmap="jet", vmin=0.0, vmax=UMAX, zorder=1,
    )
    building = ~np.isfinite(u)
    if np.any(building):
        ax.pcolormesh(
            xx, zz,
            np.ma.masked_where(~building, np.ones_like(u)),
            shading="auto",
            cmap=ListedColormap(["#2f2f2f"]),
            vmin=0, vmax=1,
            zorder=2,
        )
    if stl_polys:
        ax.add_collection(
            PolyCollection(
                stl_polys,
                facecolor="#2b2b2b",
                edgecolor="#111111",
                linewidths=0.6,
                alpha=0.95,
                zorder=5,
            )
        )
    if stl_segs:
        ax.add_collection(
            LineCollection(stl_segs, colors="#000000", linewidths=0.7, zorder=6)
        )
    gaw_x = SITES["GAW103"][0]
    ax.axvline(gaw_x, color="0.1", lw=0.9, ls="--", zorder=7)
    ax.plot(gaw_x, 60.0, marker="o", color="k", ms=5.5, zorder=8)
    ax.annotate(
        "GAW103",
        xy=(gaw_x, 60.0),
        xytext=(gaw_x - 200, 110),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="0.2", lw=0.7),
        zorder=8,
    )
    ax.set_xlim(UY_XMIN, UY_XMAX)
    ax.set_ylim(UY_ZMIN, UY_ZMAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    if show_ylabel:
        ax.set_ylabel("z (m AGL)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([200, 400, 600, 800, 1000, 1200, 1400])
    ax.set_yticks([0, 50, 100, 150, 200, 250, 300])
    ax.tick_params(direction="out", length=4)
    ax.grid(True, which="major", ls=":", lw=0.4, alpha=0.45, zorder=0)
    return pcm


def plot_uy_matplotlib():
    """1×3 |U|(x,z) at y=-320 with buildings, height ticks, shared colorbar."""
    import matplotlib.pyplot as plt

    fields = {k: _load_uy_mag(k) for k in ("M1", "M2", "M3")}
    if any(v is None for v in fields.values()):
        missing = [k for k, v in fields.items() if v is None]
        print(f"[warn] skip Uy matplotlib; missing CSV for {missing}")
        return

    stl_polys, stl_segs = _stl_xz_building_polys(STL, Y_SLICE, dy=UY_STL_DY)
    labels = {"M1": "M1  3.78M", "M2": "M2  5.47M", "M3": "M3  8.71M"}

    for key in ("M1", "M2", "M3"):
        xx, zz, u = fields[key]
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        pcm = _draw_uy_panel(
            ax, xx, zz, u, stl_polys, stl_segs,
            title=f"{labels[key]}  ·  y = {Y_SLICE:.0f} m  ·  2025-09-01 18:00 (UTC+8)",
        )
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.2%", pad=0.08)
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.set_label(r"$|U|$ (m s$^{-1}$)")
        ax.text(
            0.01, 0.98, UY_BUILDING_NOTE,
            transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            color="0.25",
        )
        fig.tight_layout()
        out = OUT / f"panel_Uy-320_{key}.png"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[save] {out}")

    # Manual 3×1 placement sized to data aspect → no tall empty axes bands.
    ax_w_in = 7.4
    ax_h_in = ax_w_in * (UY_ZMAX - UY_ZMIN) / (UY_XMAX - UY_XMIN)
    gap_in = 0.05
    top_in, bot_in, left_in, cbar_gap_in, cbar_w_in = 0.32, 0.40, 0.85, 0.10, 0.16
    fig_w = left_in + ax_w_in + cbar_gap_in + cbar_w_in + 0.35
    fig_h = bot_in + 3 * ax_h_in + 2 * gap_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _ax_rect(row):
        # row 0 = top
        x0 = left_in / fig_w
        y0 = (bot_in + (2 - row) * (ax_h_in + gap_in)) / fig_h
        return [x0, y0, ax_w_in / fig_w, ax_h_in / fig_h]

    axes = [fig.add_axes(_ax_rect(i)) for i in range(3)]
    pcm = None
    for i, (ax, key) in enumerate(zip(axes, ("M1", "M2", "M3"))):
        xx, zz, u = fields[key]
        pcm = _draw_uy_panel(
            ax, xx, zz, u, stl_polys, stl_segs,
            title="",
            show_ylabel=True,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.5, 0.97, labels[key],
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=10,
        )
        ax.tick_params(labelleft=True)
        if i < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        if i == 0:
            ax.text(
                0.01, 0.97, UY_BUILDING_NOTE,
                transform=ax.transAxes, fontsize=7, va="top", ha="left",
                color="0.25", zorder=10,
            )
    # Short colorbar next to the middle panel only.
    mid = _ax_rect(1)
    cax = fig.add_axes([
        mid[0] + mid[2] + cbar_gap_in / fig_w,
        mid[1] + 0.22 * mid[3],
        cbar_w_in / fig_w,
        0.56 * mid[3],
    ])
    cbar = fig.colorbar(pcm, cax=cax)
    cbar.set_label(r"$|U|$ (m s$^{-1}$)")
    fig.suptitle(
        f"Vertical slice at y = {Y_SLICE:.0f} m (through GAW103), 2025-09-01 18:00 (UTC+8)",
        fontsize=11,
        y=0.99,
    )
    out = OUT / "grid_compare_U_y-320_GAW103.png"
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print(f"[save] {out}")


def stitch_and_diff():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.image import imread

    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)

    def row(prefix, outfile, titles=True):
        keys = ["M1", "M2", "M3"]
        labels = ["M1  3.78M", "M2  5.47M", "M3  8.71M"]
        imgs = [imread(str(OUT / f"{prefix}_{k}.png")) for k in keys]
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6 if "Uy" in prefix else 4.9))
        for ax, im, lab in zip(axes, imgs, labels):
            ax.imshow(im)
            ax.set_axis_off()
            if titles:
                ax.set_title(lab, fontsize=11, pad=4)
        fig.tight_layout(w_pad=0.15)
        fig.savefig(outfile, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"[save] {outfile}")

    row("panel_mesh", OUT / "grid_compare_mesh_GAW104.png")
    row("panel_Uz60", OUT / "grid_compare_U_z60m.png")
    plot_uy_matplotlib()

    # Difference maps from resampled CSVs (400×400 image; no XYZ columns).
    def load_mag(key):
        p = OUT / f"Uz60_{key}.csv"
        if not p.exists():
            return None
        import csv

        with p.open() as f:
            reader = csv.DictReader(f)
            u0s, u1s, u2s, mask = [], [], [], []
            for row in reader:
                u0s.append(float(row["U:0"]))
                u1s.append(float(row["U:1"]))
                u2s.append(float(row.get("U:2", "0") or 0.0))
                mask.append(float(row.get("vtkValidPointMask", "1")))
        v = np.sqrt(np.asarray(u0s) ** 2 + np.asarray(u1s) ** 2 + np.asarray(u2s) ** 2)
        m = np.asarray(mask) > 0.5
        v = np.where(m, v, np.nan)
        n = int(np.sqrt(len(v)))
        if n * n != len(v):
            print(f"[warn] {key} resample n={len(v)} not square")
            return None
        # ResampleToImage SamplingBounds = [-2000,2000]×[-2000,2000]
        x1d = np.linspace(-2000.0, 2000.0, n)
        y1d = np.linspace(-2000.0, 2000.0, n)
        xx, yy = np.meshgrid(x1d, y1d)
        return xx, yy, v.reshape(n, n)

    fields = {k: load_mag(k) for k in ("M1", "M2", "M3")}
    if all(fields[k] is not None for k in fields):
        x, y, u2 = fields["M2"]
        u1 = fields["M1"][2]
        u3 = fields["M3"][2]
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), sharey=True)
        for ax, d, title in (
            (axes[0], u1 - u2, r"$|U|_{\mathrm{M1}}-|U|_{\mathrm{M2}}$"),
            (axes[1], u3 - u2, r"$|U|_{\mathrm{M3}}-|U|_{\mathrm{M2}}$"),
        ):
            pcm = ax.pcolormesh(
                x, y, np.ma.masked_invalid(d), shading="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6
            )
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_title(title)
            ax.set_xlim(-2000, 2000)
            ax.set_ylim(-2000, 2000)
            for name, (sx, sy) in SITES.items():
                ax.plot(sx, sy, "k.", ms=4)
                ax.text(sx + 40, sy + 40, name, fontsize=7)
        axes[0].set_ylabel("y (m)")
        fig.colorbar(pcm, ax=axes, shrink=0.85, label=r"$\Delta |U|$ (m s$^{-1}$)")
        fig.suptitle("2025-09-01 18:00 (UTC+8) | z = 60 m", y=1.02, fontsize=11)
        fig.savefig(OUT / "grid_compare_dU_z60m.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {OUT / 'grid_compare_dU_z60m.png'}")
        print("RMSE M1-M2 (valid)", float(np.sqrt(np.nanmean((u1 - u2) ** 2))))
        print("RMSE M3-M2 (valid)", float(np.sqrt(np.nanmean((u3 - u2) ** 2))))
        print("max |M1-M2|", float(np.nanmax(np.abs(u1 - u2))))
        print("max |M3-M2|", float(np.nanmax(np.abs(u3 - u2))))


def main():
    OUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    modes = sys.argv[1:] or ["mesh", "uz", "uy", "stitch"]
    if "mesh" in modes or "uz" in modes or "uy" in modes:
        for case in CASES:
            if not case["foam"].is_file():
                case["foam"].touch()
            print(f"\n==== {case['label']}  {case['foam']} ====")
            if "mesh" in modes:
                render_mesh(case)
                _reset()
            if "uz" in modes:
                render_uz(case)
                _reset()
            if "uy" in modes:
                render_uy(case)
                _reset()
    if "stitch" in modes or "uyplot" in modes:
        if "uyplot" in modes and "stitch" not in modes:
            plot_uy_matplotlib()
        else:
            stitch_and_diff()


if __name__ == "__main__":
    os.chdir(str(REPO))
    main()
