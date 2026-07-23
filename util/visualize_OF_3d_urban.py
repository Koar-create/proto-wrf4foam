"""
3D Urban OpenFOAM Flow Visualization  (pvbatch)
===============================================
Produces publication-quality 3-D perspective renderings of an OpenFOAM
urban-ABL steady-state case:

  slice  — horizontal velocity-magnitude slice at height z + gray buildings
           (matches Fig. 5 style: coloured ground-level plane + 3-D building blocks)

  qcrit  — Q-criterion isosurfaces coloured by |U| + dark-gray buildings
           (matches Fig. 6 style: vortex structures over city)

Both modes load the reconstructed case (root-level time directory), apply
CellDatatoPointData, and render an offscreen 3-D perspective view.

Note on qcrit for steady RANS
------------------------------
Q-criterion is most informative for instantaneous velocity fields from unsteady
(transient / LES / DNS) simulations.  For steady RANS the time-averaged
velocity gradient is smooth and Q_max is tiny (~ 1e-6 s⁻² when U ~ 1 m/s),
so no visible vortex structures appear.  Scale the threshold approximately as
q_val ≈ (U_case / U_ref)² × 0.35  where U_ref = 8 m/s for the OKC reference.

Usage
-----
  conda activate paraview-env
  pvbatch util/visualize_OF_3d_urban.py <foam> [options]

Examples
--------
  # Fig. 5 style – slice at z=8 m, auto vmax
  pvbatch util/visualize_OF_3d_urban.py \\
      steady_experiments_finer_ABL/20250904_0000_two_boundaries_as_outlet/myExpxx.foam \\
      --mode slice --z 8

  # Fig. 5 style – explicit vmax
  pvbatch util/visualize_OF_3d_urban.py \\
      steady_experiments_finer_ABL/20250904_0000_two_boundaries_as_outlet/myExpxx.foam \\
      --mode slice --z 8 --vmax 1.5

  # Fig. 6 style – Q-criterion (best for LES/transient; threshold scales with U²)
  pvbatch util/visualize_OF_3d_urban.py \\
      steady_experiments_finer_ABL/20250904_0000_two_boundaries_as_outlet/myExpxx.foam \\
      --mode qcrit --q-val 5e-7 --clip-xy 1500 --clip-z 300

Case geometry (20250904_0000_two_boundaries_as_outlet)
------------------------------------------------------
  Domain  : 10 km × 10 km × 2 km  (±5000 m in X/Y, 0–2000 m in Z)
  Inlets  : east + south  (WRF-mapped)
  Outlets : west + north  (p = 0)
  U₀      : (−1.68, 1.38, 0) m/s  → wind from ESE, flowing WNW
  Solver  : simpleFoam  endTime=5000
"""

from paraview.simple import *
import argparse
import os
import sys

# Suppress automatic camera-reset on first render (avoids view jumping)
paraview.simple._DisableFirstRenderCameraReset()


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="3D urban OpenFOAM flow visualisation (pvbatch).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("foam", help="Path to the .foam case file")
    p.add_argument(
        "--mode",
        choices=["slice", "qcrit"],
        default="slice",
        help="Visualisation mode",
    )
    p.add_argument("--z", type=float, default=8.0, help="Slice height in m (slice mode)")
    p.add_argument("--time", type=float, default=5000.0, help="OpenFOAM time step to load")
    p.add_argument(
        "--q-val",
        default="auto",
        help=(
            "Q-criterion isosurface value in s^-2 (qcrit mode). "
            "Use 'auto' to derive a visible threshold from the current case."
        ),
    )
    p.add_argument(
        "--no-auto-q-fallback",
        action="store_true",
        help="Do not lower the Q threshold automatically when the requested contour is empty.",
    )
    p.add_argument(
        "--no-vorticity-fallback",
        action="store_true",
        help=(
            "Do not fall back to |vorticity| isosurfaces when the Q contour is empty "
            "(useful for steady RANS cases)."
        ),
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colour-scale upper bound in m/s (default: 11 for slice, 8 for qcrit)",
    )
    p.add_argument("--out", default=None, help="Output PNG path")
    p.add_argument("--width",     type=int,   default=2400, help="Image width in px")
    p.add_argument("--height-px", type=int,   default=1600, help="Image height in px")
    p.add_argument(
        "--clip-xy",
        type=float,
        default=2500.0,
        help="Half-side of XY clip box in m (qcrit mode, reduces memory)",
    )
    p.add_argument(
        "--clip-z",
        type=float,
        default=600.0,
        help="Upper Z limit of clip box in m (qcrit mode)",
    )
    p.add_argument(
        "--min-iso-cells",
        type=int,
        default=500,
        help=(
            "Minimum contour cells before switching to |vorticity| fallback "
            "on steady RANS fields (qcrit mode)."
        ),
    )
    return p.parse_args()


# ── OpenFOAM reader helpers ───────────────────────────────────────────────────

def _resolve_time(reader, requested):
    """Return the closest available timestep to *requested*."""
    avail = list(reader.TimestepValues) if reader.TimestepValues else []
    if not avail:
        print(f"[warn] No timesteps found in reader; using {requested}")
        return requested
    if requested in avail:
        return requested
    closest = min(avail, key=lambda t: abs(t - requested))
    print(f"[warn] t={requested} not available; using closest t={closest}. "
          f"Available: {avail}")
    return closest


def _load_foam(foam_path, time):
    """Load the OpenFOAM case and return (reader, actual_time)."""
    src = OpenFOAMReader(FileName=foam_path)
    src.MeshRegions = ["internalMesh"]
    src.CellArrays  = ["U"]
    # First UpdatePipeline to populate TimestepValues
    UpdatePipeline(proxy=src)
    t = _resolve_time(src, time)
    UpdatePipeline(t, proxy=src)
    print(f"[info] Loaded {foam_path}  t={t}")
    return src, t


# ── Auto-range helper ────────────────────────────────────────────────────────

def _auto_vmax(source, time, array_name="U", fraction=0.98):
    """
    Compute the approximate *fraction* percentile of the magnitude of
    *array_name* by running a Calculator over *source*.
    Returns the magnitude value, or None on failure.
    """
    try:
        calc = Calculator(Input=source)
        calc.AttributeType    = "Point Data"
        calc.ResultArrayName  = "_UMag_tmp"
        calc.Function         = f"mag({array_name})"
        UpdatePipeline(time, proxy=calc)
        info      = calc.GetDataInformation()
        arr_info  = info.GetPointDataInformation().GetArrayInformation("_UMag_tmp")
        if arr_info is None:
            return None
        mag_range = arr_info.GetComponentRange(0)
        Delete(calc)
        return float(mag_range[1]) * fraction
    except Exception as exc:
        print(f"[warn] _auto_vmax failed: {exc}")
        return None


# ── Colour-map helpers ────────────────────────────────────────────────────────

def _apply_jet(display, array_name, vmin, vmax):
    """
    Colour the display by the magnitude of *array_name* using the Jet preset.
    Returns the colour-transfer LUT.
    """
    ColorBy(display, ("POINTS", array_name, "Magnitude"))
    lut = GetColorTransferFunction(array_name)
    lut.ApplyPreset("Jet", True)
    lut.RescaleTransferFunction(vmin, vmax)
    pwf = GetOpacityTransferFunction(array_name)
    pwf.RescaleTransferFunction(vmin, vmax)
    return lut


def _try_set(obj, attr, value):
    """Silently set an attribute; swallow errors for properties absent in some PV versions."""
    try:
        setattr(obj, attr, value)
    except Exception:
        pass


def _data_counts(source):
    """Return (n_points, n_cells, bounds) for a ParaView pipeline object."""
    info = source.GetDataInformation()
    return info.GetNumberOfPoints(), info.GetNumberOfCells(), list(info.GetBounds())


def _describe_dataset(source, label):
    """Print compact diagnostics for data-size and spatial bounds debugging."""
    try:
        n_points, n_cells, bounds = _data_counts(source)
        btxt = ", ".join(f"{v:.3g}" for v in bounds)
        print(f"[diag] {label}: points={n_points:,} cells={n_cells:,} bounds=[{btxt}]")
    except Exception as exc:
        print(f"[warn] Could not describe {label}: {exc}")


def _array_assoc_and_range(source, array_name):
    """Return (association, range) for an array on POINTS or CELLS, or (None, None)."""
    try:
        info = source.GetDataInformation()
        for assoc, getter in (
            ("POINTS", info.GetPointDataInformation),
            ("CELLS", info.GetCellDataInformation),
        ):
            arr = getter().GetArrayInformation(array_name)
            if arr is not None:
                return assoc, arr.GetComponentRange(0)
    except Exception:
        pass
    return None, None


def _choose_visible_q(q_range):
    """Choose a conservative positive Q isovalue from the actual data range."""
    if not q_range:
        return None
    q_min, q_max = float(q_range[0]), float(q_range[1])
    if q_max <= 0.0:
        return None
    # Steady RANS Q peaks are tiny; contour much closer to Q_max to get any surface.
    if q_max < 1.0e-4:
        return max(q_max * 0.02, 1.0e-12)
    if q_min < 0.0:
        return max(q_max * 0.20, 1.0e-12)
    return max(q_min + (q_max - q_min) * 0.35, q_max * 0.20, 1.0e-12)


def _choose_visible_scalar(s_range, tiny_max=1.0e-2, fraction=0.25):
    """Pick an isovalue from a positive scalar range (e.g. |vorticity|)."""
    if not s_range:
        return None
    s_min, s_max = float(s_range[0]), float(s_range[1])
    if s_max <= 0.0:
        return None
    if s_max < tiny_max:
        return max(s_max * 0.08, 1.0e-8)
    return max(s_min + (s_max - s_min) * fraction, s_max * 0.12, 1.0e-8)


def _make_vorticity_isosurface(clip, t):
    """
    Build a |vorticity| isosurface as a RANS-friendly fallback when Q is empty.
    Returns (iso_proxy, iso_value, n_cells) or (None, None, 0) on failure.
    """
    try:
        vort = Vorticity(Input=clip)
    except NameError:
        print("[warn] Vorticity filter unavailable in this ParaView build.")
        return None, None, 0

    for attr in ("InputArray", "Vectors", "Input"):
        if hasattr(vort, attr):
            try:
                setattr(vort, attr, ["POINTS", "U"])
                break
            except Exception:
                pass

    UpdatePipeline(t, proxy=vort)
    mag = Calculator(Input=vort)
    mag.AttributeType   = "Point Data"
    mag.ResultArrayName = "_VortMag"
    mag.Function        = "mag(Vorticity)"
    UpdatePipeline(t, proxy=mag)

    v_assoc, v_range = _array_assoc_and_range(mag, "_VortMag")
    if v_range:
        print(f"[diag] |vorticity| range ({v_assoc}) = [{v_range[0]:.6g}, {v_range[1]:.6g}]")
    else:
        print("[warn] Could not compute |vorticity| for fallback.")
        Delete(mag)
        Delete(vort)
        return None, None, 0

    iso_val = _choose_visible_scalar(v_range)
    if iso_val is None:
        Delete(mag)
        Delete(vort)
        return None, None, 0

    iso = Contour(Input=mag)
    iso.ContourBy   = [v_assoc or "POINTS", "_VortMag"]
    iso.Isosurfaces = [iso_val]
    UpdatePipeline(t, proxy=iso)
    _, n_cells, _ = _data_counts(iso)
    print(f"[diag] |vorticity| contour cells at {iso_val:.6g} = {n_cells:,}")
    if n_cells == 0:
        Delete(iso)
        Delete(mag)
        Delete(vort)
        return None, None, 0
    return iso, iso_val, n_cells


def _bounds_span(bounds):
    """Return (x_half, y_half, z_max) from a 6-element bounds list."""
    if not bounds or len(bounds) < 6:
        return None
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return max(abs(xmin), abs(xmax)), max(abs(ymin), abs(ymax)), zmax


def _select_core_clip(c2p, t, clip_xy, clip_z):
    """
    Extract the urban-core subvolume with a Box clip.
    ParaView versions disagree on Invert semantics, so evaluate both and pick
    the candidate whose XY extent best matches the requested core box.
    """
    bounds = [-clip_xy, clip_xy, -clip_xy, clip_xy, 0.0, clip_z]
    candidates = []
    parent_cells = _data_counts(c2p)[1]
    for invert in (0, 1):
        candidate = Clip(Input=c2p)
        candidate.ClipType = "Box"
        candidate.Invert   = invert
        try:
            candidate.ClipType.Bounds = bounds
        except Exception:
            candidate.ClipType.Position = [0.0, 0.0, clip_z / 2.0]
            candidate.ClipType.Scale    = [clip_xy, clip_xy, clip_z / 2.0]
        UpdatePipeline(t, proxy=candidate)
        _, n_cells, cbounds = _data_counts(candidate)
        span = _bounds_span(cbounds)
        if span is None:
            score = float("inf")
        else:
            x_half, y_half, z_max = span
            score = abs(x_half - clip_xy) + abs(y_half - clip_xy) + abs(z_max - clip_z)
        candidates.append((score, n_cells, candidate, invert, cbounds))

    non_empty = [item for item in candidates if item[1] > 0]
    if not non_empty:
        print("[warn] Box clip produced no cells; using unclipped cell-to-point data.")
        return c2p, None

    # Prefer bounds that match the requested core; tie-break on smaller subset.
    score, n_cells, clip, invert, cbounds = min(
        non_empty,
        key=lambda item: (item[0], item[1] if item[1] < parent_cells * 0.5 else item[1] * 1.0e6),
    )
    print(
        f"[qcrit] Box clip applied with Invert={invert} "
        f"(±{clip_xy} m × {clip_z} m, cells={n_cells:,}, score={score:.3g})"
    )
    for _, _, other, _, _ in candidates:
        if other is not clip:
            Delete(other)
    return clip, cbounds


def _add_colorbar(lut, view, title, location="UpperLeftCorner"):
    """Add and style a vertical scalar bar for *lut* in *view*."""
    sb = GetScalarBar(lut, view)
    sb.Title          = title
    sb.Visibility     = 1
    sb.Orientation    = "Vertical"
    # PV 6.x uses space-separated location strings (e.g. "Upper Left Corner")
    # PV 5.x used camelCase (e.g. "UpperLeftCorner") – try new style first
    _location_map = {
        "UpperLeftCorner":  "Upper Left Corner",
        "UpperRightCorner": "Upper Right Corner",
        "LowerLeftCorner":  "Lower Left Corner",
        "LowerRightCorner": "Lower Right Corner",
        "UpperCenter":      "Upper Center",
        "LowerCenter":      "Lower Center",
    }
    loc_pv6 = _location_map.get(location, location)
    try:
        sb.WindowLocation = loc_pv6
    except Exception:
        sb.WindowLocation = location   # fall back to original string
    _try_set(sb, "ComponentTitle",     "")
    _try_set(sb, "TitleFontSize",      20)
    _try_set(sb, "LabelFontSize",      16)
    _try_set(sb, "TitleBold",          1)
    _try_set(sb, "LabelBold",          0)
    _try_set(sb, "DrawTickMarks",      1)
    _try_set(sb, "DrawTickLabels",     1)
    _try_set(sb, "NumberOfLabels",     5)
    _try_set(sb, "ScalarBarLength",    0.33)
    _try_set(sb, "ScalarBarThickness", 18)
    return sb


# ── Camera presets ────────────────────────────────────────────────────────────

def _set_camera_slice(view):
    """
    Oblique top-down view from the ESE (upstream side of the domain).
    Wind blows from ESE toward WNW, so the camera is placed behind the
    inlet (ESE) looking across the urban area — matching Fig. 5's perspective
    with flow appearing to come from the right of the image.
    """
    view.CameraPosition   = [6500, -6500, 5000]
    view.CameraFocalPoint = [0, 0, 150]
    view.CameraViewUp     = [0, 0, 1]
    view.CameraParallelProjection = 0


def _set_camera_qcrit(view):
    """Slightly steeper overhead oblique for the denser Q-criterion scene."""
    view.CameraPosition   = [2600, -2600, 1800]
    view.CameraFocalPoint = [0, 0, 160]
    view.CameraViewUp     = [0, 0, 1]
    view.CameraParallelProjection = 0


# ── Flow-direction annotation ─────────────────────────────────────────────────

def _add_flow_annotation(view):
    """
    Overlay a 2-D text label indicating wind direction.
    Wind is from ESE (positive-x, negative-y direction), which appears
    from the right of the image when the camera is at [+x, -y, +z].
    """
    try:
        txt = Text()
        txt.Text = "Flow direction  \u25c4"   # ◄ BLACK LEFT-POINTING POINTER
        disp = Show(txt, view)
        # PV 6.x uses "Any Location" (with space); older versions used "AnyLocation"
        try:
            disp.WindowLocation = "Any Location"
        except Exception:
            disp.WindowLocation = "AnyLocation"
        disp.Position  = [0.60, 0.88]
        disp.FontSize  = 15
        disp.Bold      = 1
        disp.Italic    = 0
        disp.Color     = [0.0, 0.0, 0.0]
        disp.Opacity   = 1.0
    except Exception as exc:
        print(f"[warn] Flow-direction annotation skipped: {exc}")


# ── Building surface ──────────────────────────────────────────────────────────

def _show_buildings(stl_path, view, dark=False, clip_xy=None):
    """
    Load *buildings.stl* (binary) and display as a solid-coloured surface.
    *dark=True* uses near-black (Q-criterion style), otherwise mid-grey.
    """
    if not os.path.isfile(stl_path):
        print(f"[warn] buildings.stl not found at {stl_path!r}; skipping.")
        return None
    bldgs = STLReader(FileNames=[stl_path])
    UpdatePipeline(proxy=bldgs)
    shown = bldgs
    if clip_xy:
        try:
            bclip = Clip(Input=bldgs)
            bclip.ClipType = "Box"
            bclip.ClipType.Bounds = [-clip_xy, clip_xy, -clip_xy, clip_xy, -50.0, 900.0]
            bclip.Invert = 1
            UpdatePipeline(proxy=bclip)
            shown = bclip
        except Exception as exc:
            print(f"[warn] Building clip skipped: {exc}")
    disp = Show(shown, view)
    disp.Representation = "Surface"
    # STL files have no field arrays, so PV defaults to solid-colour display.
    # Just set the colour properties directly – no ColorBy call needed.
    c = [0.22, 0.22, 0.22] if dark else [0.55, 0.55, 0.55]
    disp.AmbientColor  = c
    disp.DiffuseColor  = c
    disp.SpecularColor = [1.0, 1.0, 1.0]
    disp.Specular      = 0.12
    disp.Opacity       = 1.0
    return bldgs


# ── View setup ────────────────────────────────────────────────────────────────

def _make_view(width, height_px):
    """Create (or reuse) the RenderView and set basic display properties."""
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize                  = [width, height_px]
    view.Background                = [1.0, 1.0, 1.0]   # white background
    view.OrientationAxesVisibility = 0
    # PV 6.x: use BackgroundColorMode instead of UseGradientBackground
    for bg_mode in ("Single Color", "Background Color"):
        try:
            view.BackgroundColorMode = bg_mode
            break
        except Exception:
            pass
    _try_set(view, "Background2", [1.0, 1.0, 1.0])
    _try_set(view, "UseColorPaletteForBackground", 0)
    return view


# ── SLICE MODE ────────────────────────────────────────────────────────────────

def visualize_slice(foam_path, stl_path, z, time, vmax, out_path, width, height_px):
    """
    Render a horizontal velocity-magnitude slice at height *z* together with
    the gray building surfaces.  Saves a PNG to *out_path*.
    """
    print(f"\n{'='*60}")
    print(f"[slice] z={z} m  |  t={time}  |  vmax={vmax} m/s")
    print(f"[slice] output → {out_path}")
    print(f"{'='*60}\n")

    view = _make_view(width, height_px)

    # ── Load case ────────────────────────────────────────────────────────────
    src, t = _load_foam(foam_path, time)

    # ── Cell → Point ─────────────────────────────────────────────────────────
    c2p = CellDatatoPointData(Input=src)
    c2p.ProcessAllArrays = 1
    UpdatePipeline(t, proxy=c2p)

    # ── Horizontal slice at z ─────────────────────────────────────────────────
    sl = Slice(Input=c2p)
    sl.SliceType        = "Plane"
    sl.SliceType.Normal = [0.0, 0.0, 1.0]
    sl.SliceType.Origin = [0.0, 0.0, z]
    UpdatePipeline(t, proxy=sl)

    # Auto-compute vmax from slice data if not specified
    if vmax is None:
        vmax = _auto_vmax(sl, t) or 3.0
        print(f"[slice] Auto vmax = {vmax:.3f} m/s")

    # ── Display slice ─────────────────────────────────────────────────────────
    sl_disp = Show(sl, view)
    sl_disp.Representation = "Surface"
    lut = _apply_jet(sl_disp, "U", 0.0, vmax)

    # ── Buildings ─────────────────────────────────────────────────────────────
    _show_buildings(stl_path, view, dark=False)

    # ── Camera, annotation, colorbar ─────────────────────────────────────────
    _set_camera_slice(view)
    _add_flow_annotation(view)
    _add_colorbar(lut, view, "u(t)", location="UpperLeftCorner")

    # ── Render & save ─────────────────────────────────────────────────────────
    RenderAllViews()
    SaveScreenshot(
        out_path,
        view,
        ImageResolution=[width, height_px],
        OverrideColorPalette="",
    )
    print(f"\n[slice] Saved → {out_path}")


# ── QCRIT MODE ────────────────────────────────────────────────────────────────

def visualize_qcrit(foam_path, stl_path, q_val, time, vmax,
                    out_path, width, height_px, clip_xy, clip_z,
                    auto_q_fallback=True, vorticity_fallback=True,
                    min_iso_cells=500):
    """
    Compute Q-criterion isosurfaces from the velocity gradient, colour them by
    |U|, and render with dark-gray buildings.  A box clip to the urban core
    (±clip_xy m, 0–clip_z m) is applied before gradient computation to reduce
    memory usage.  Saves a PNG to *out_path*.
    """
    print(f"\n{'='*60}")
    print(f"[qcrit] Q={q_val} s⁻²  |  t={time}  |  vmax={vmax} m/s")
    print(f"[qcrit] clip_xy=±{clip_xy} m  clip_z=0–{clip_z} m")
    print(f"[qcrit] output → {out_path}")
    print(f"{'='*60}\n")

    view = _make_view(width, height_px)

    # ── Load case ────────────────────────────────────────────────────────────
    src, t = _load_foam(foam_path, time)

    # ── Cell → Point ─────────────────────────────────────────────────────────
    c2p = CellDatatoPointData(Input=src)
    c2p.ProcessAllArrays = 1
    UpdatePipeline(t, proxy=c2p)
    _describe_dataset(c2p, "cell-to-point")

    # ── Box clip: keep only the urban core ───────────────────────────────────
    clip, _ = _select_core_clip(c2p, t, clip_xy, clip_z)
    _describe_dataset(clip, "qcrit core")

    # ── Velocity gradient + Q-criterion ──────────────────────────────────────
    # GradientOfUnstructuredDataset computes ∇U and, optionally, Q-criterion.
    # Q = −½ (S:S − Ω:Ω)  where S = sym(∇U), Ω = antisym(∇U)
    try:
        grad = GradientOfUnstructuredDataset(Input=clip)
    except NameError:
        # PV 6.x alias
        grad = Gradient(Input=clip)

    # Specify input vector field (try both known property names)
    for attr in ("ScalarArray", "Scalars", "Vectors"):
        if hasattr(grad, attr):
            try:
                setattr(grad, attr, ["POINTS", "U"])
                break
            except Exception:
                pass

    grad.ComputeQCriterion = 1
    # QCriterionArrayName may not exist in all versions – wrap defensively
    try:
        grad.QCriterionArrayName = "Q-criterion"
    except Exception:
        pass
    UpdatePipeline(t, proxy=grad)
    print(f"[qcrit] Velocity gradient + Q-criterion computed")
    q_assoc, q_range = _array_assoc_and_range(grad, "Q-criterion")
    if q_range:
        print(f"[diag] Q-criterion range ({q_assoc}) = [{q_range[0]:.6g}, {q_range[1]:.6g}]")
    else:
        print("[warn] Q-criterion array not found after gradient computation")

    # ── Isosurface at Q = q_val ───────────────────────────────────────────────
    if q_val is None:
        q_val = _choose_visible_q(q_range)
        if q_val is None:
            print("[warn] Could not derive a positive Q threshold; falling back to 1e-12.")
            q_val = 1.0e-12
        print(f"[qcrit] Auto Q isovalue = {q_val:.6g} s^-2")

    q_assoc = q_assoc or "POINTS"
    iso = Contour(Input=grad)
    iso.ContourBy   = [q_assoc, "Q-criterion"]
    iso.Isosurfaces = [q_val]
    UpdatePipeline(t, proxy=iso)
    _, iso_cells, _ = _data_counts(iso)
    print(f"[diag] Q contour cells at {q_val:.6g} = {iso_cells:,}")
    if iso_cells == 0 and auto_q_fallback:
        fallback_q = _choose_visible_q(q_range)
        if fallback_q is not None and abs(fallback_q - q_val) > max(abs(q_val), 1.0) * 1.0e-12:
            print(
                f"[warn] Requested Q={q_val:.6g} produced no surface; "
                f"retrying with Q={fallback_q:.6g}."
            )
            Delete(iso)
            q_val = fallback_q
            iso = Contour(Input=grad)
            iso.ContourBy   = [q_assoc, "Q-criterion"]
            iso.Isosurfaces = [q_val]
            UpdatePipeline(t, proxy=iso)
            _, iso_cells, _ = _data_counts(iso)
            print(f"[diag] Q contour cells at fallback {q_val:.6g} = {iso_cells:,}")

    used_vorticity = False
    sparse_q = 0 < iso_cells < min_iso_cells
    if (iso_cells == 0 or sparse_q) and vorticity_fallback:
        if sparse_q:
            print(
                f"[warn] Q isosurface is too sparse ({iso_cells:,} cells < "
                f"{min_iso_cells:,}); falling back to |vorticity| isosurfaces."
            )
        else:
            print(
                "[warn] Q isosurface is empty on this steady RANS field; "
                "falling back to |vorticity| isosurfaces."
            )
        Delete(iso)
        iso, iso_val, iso_cells = _make_vorticity_isosurface(clip, t)
        used_vorticity = iso is not None and iso_cells > 0
        if used_vorticity:
            q_val = iso_val
    elif iso_cells == 0:
        print(
            "[warn] Q isosurface is empty. Steady RANS mean fields rarely "
            "produce Fig. 6-style Q structures; try LES/transient data or "
            "rerun without --no-vorticity-fallback."
        )

    # Auto-compute vmax from isosurface data if not specified
    if vmax is None:
        vmax = _auto_vmax(iso, t) or 4.0
        print(f"[qcrit] Auto vmax = {vmax:.3f} m/s")

    # ── Display isosurface coloured by |U| ───────────────────────────────────
    iso_disp = Show(iso, view)
    iso_disp.Representation = "Surface"
    iso_disp.Opacity = 0.92
    _try_set(iso_disp, "Specular", 0.15)
    lut = _apply_jet(iso_disp, "U", 0.0, vmax)

    # ── Buildings ─────────────────────────────────────────────────────────────
    _show_buildings(stl_path, view, dark=True, clip_xy=clip_xy * 1.15)

    # ── Camera, colorbar ─────────────────────────────────────────────────────
    _set_camera_qcrit(view)
    _add_colorbar(lut, view, "U Mag (m/s)", location="LowerRightCorner")

    # ── Render & save ─────────────────────────────────────────────────────────
    RenderAllViews()
    SaveScreenshot(
        out_path,
        view,
        ImageResolution=[width, height_px],
        OverrideColorPalette="",
    )
    metric = "|vorticity|" if used_vorticity else "Q-criterion"
    print(f"\n[qcrit] Saved → {out_path}  (metric={metric}, isovalue={q_val:.6g})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    foam_path = os.path.abspath(args.foam)
    if not os.path.isfile(foam_path):
        print(f"Error: .foam file not found: {foam_path}")
        sys.exit(1)

    foam_dir = os.path.dirname(foam_path)
    stl_path = os.path.join(foam_dir, "constant", "triSurface", "buildings.stl")
    out_dir  = os.path.join(foam_dir, "postProcessing")
    os.makedirs(out_dir, exist_ok=True)

    # Colour-scale upper bound: use user-specified value, or auto-compute from data
    vmax = args.vmax  # None triggers auto-calculation inside each mode function
    if str(args.q_val).lower() == "auto":
        q_val = None
        q_tag = "auto"
    else:
        try:
            q_val = float(args.q_val)
        except ValueError:
            print(f"Error: --q-val must be a float or 'auto', got {args.q_val!r}")
            sys.exit(2)
        q_tag = f"{q_val:g}".replace(".", "p")

    # Default output filename
    if args.out:
        out_path = (
            os.path.abspath(args.out)
            if os.path.isabs(args.out)
            else os.path.join(foam_dir, args.out)
        )
    elif args.mode == "slice":
        z_tag = str(int(args.z)) if args.z == int(args.z) else str(args.z)
        out_path = os.path.join(out_dir, f"3d_slice_z{z_tag}m.png")
    else:
        out_path = os.path.join(out_dir, f"3d_qcrit_Q{q_tag}.png")

    if args.mode == "slice":
        visualize_slice(
            foam_path, stl_path,
            z=args.z, time=args.time, vmax=vmax,
            out_path=out_path,
            width=args.width, height_px=args.height_px,
        )
    else:
        visualize_qcrit(
            foam_path, stl_path,
            q_val=q_val, time=args.time, vmax=vmax,
            out_path=out_path,
            width=args.width, height_px=args.height_px,
            clip_xy=args.clip_xy, clip_z=args.clip_z,
            auto_q_fallback=not args.no_auto_q_fallback,
            vorticity_fallback=not args.no_vorticity_fallback,
            min_iso_cells=args.min_iso_cells,
        )


if __name__ == "__main__":
    main()
