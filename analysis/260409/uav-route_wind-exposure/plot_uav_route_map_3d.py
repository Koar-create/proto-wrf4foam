#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV route 3D overview map (buildings + 30 m routes only).

Independent of uav_route_wind_shear_analysis.py — does not modify that script.

Camera / framing deliberately matches the reference VTK-style view:
  - domain box X∈[-2000,2000], Y∈[-1500,1500], Z∈[0,500]
  - viewpoint from (−X, −Y), looking toward (+X, +Y)
  - opaque light-gray buildings, dark background, white cube axes

Requires the paraview-env VTK build, e.g.:
  conda run -n paraview-env python analysis/260409/plot_uav_route_map_3d.py

Only one altitude is drawn per run (default z = 30 m). Use ``--z 120`` for 120 m.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_STL_DIR = REPO_ROOT / "constant" / "triSurface"
STL_WITH_CANTON = _STL_DIR / "buildings.stl"
STL_BEFORE_CANTON = _STL_DIR / "buildings_before_canton_lod2.stl"
DEFAULT_UAV_STL = REPO_ROOT / "gazebo_wind_plugin" / "models" / "iris_wind_quad" / "meshes" / "iris.stl"
DEFAULT_OUT = (
    REPO_ROOT
    / "results"
    / "uav_route_wind_shear"
    / "20250903_1200"
    / "route_overview_map_3d_z30.png"
)

# One-way routes (same local XY as the 2D analysis; z fixed at 30 m)
ROUTE_SPECS = {
    "Route1_open_river": {
        "label": "Route 1 (open river)",
        "start": (-1500.0, 0.0),
        "end": (1500.0, 0.0),
        "color": (0.05, 0.28, 0.63),  # deep blue
    },
    "Route2_urban_canyon": {
        "label": "Route 2 (urban canyon)",
        "start": (400.0, -1000.0),
        "end": (400.0, 1000.0),
        "color": (0.72, 0.11, 0.11),  # deep red
    },
}

# View box matching the reference figure
XLIM = (-2000.0, 2000.0)
YLIM = (-1500.0, 1500.0)
ZLIM = (0.0, 500.0)

# Camera tuned to the reference: elevated look from (−X, −Y) toward (+X, +Y)
# Focal point near domain centre / low altitude; ViewUp = +Z.
CAMERA = {
    "position": (-4800.0, -4100.0, 2700.0),
    "focal_point": (0.0, 0.0, 80.0),
    "view_up": (0.0, 0.0, 1.0),
    "view_angle": 23.0,
}

ROUTE_Z = 30.0
ROUTE_SAMPLE_STEP = 20.0
# Iris STL is ~0.5 m across; exaggerate so it reads clearly on a km-scale map.
DEFAULT_UAV_SPAN_M = 350.0


def _require_vtk():
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError as exc:
        raise SystemExit(
            "This script needs VTK (paraview-env).\n"
            "  conda run -n paraview-env python "
            "analysis/260409/plot_uav_route_map_3d.py\n"
            f"Import error: {exc}"
        ) from exc
    return vtk, numpy_to_vtk


def build_route_xyz(
    p_start: tuple[float, float],
    p_end: tuple[float, float],
    z: float,
    step: float,
) -> np.ndarray:
    """Return (N, 3) one-way polyline at constant height z."""
    x0, y0 = p_start
    x1, y1 = p_end
    length = float(np.hypot(x1 - x0, y1 - y0))
    n_seg = max(1, int(round(length / step)))
    t = np.linspace(0.0, 1.0, n_seg + 1)
    xyz = np.column_stack(
        [
            x0 + t * (x1 - x0),
            y0 + t * (y1 - y0),
            np.full(t.shape, z, dtype=float),
        ]
    )
    return xyz


def make_polyline_actor(vtk, xyz: np.ndarray, color: tuple[float, float, float], width: float = 4.0):
    """Build an opaque thick polyline (tube) actor — LineWidth is unreliable in GL."""
    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    for x, y, z in xyz:
        pts.InsertNextPoint(float(x), float(y), float(z))

    lines = vtk.vtkCellArray()
    poly_line = vtk.vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(len(xyz))
    for i in range(len(xyz)):
        poly_line.GetPointIds().SetId(i, i)
    lines.InsertNextCell(poly_line)

    pdata = vtk.vtkPolyData()
    pdata.SetPoints(pts)
    pdata.SetLines(lines)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(pdata)
    tube.SetRadius(max(8.0, width * 2.5))
    tube.SetNumberOfSides(12)
    tube.CappingOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(1.0)
    actor.GetProperty().LightingOff()
    return actor


def make_uav_actor(
    vtk,
    uav_stl: Path,
    position: np.ndarray,
    heading_xy: tuple[float, float],
    *,
    target_span_m: float = DEFAULT_UAV_SPAN_M,
    color: tuple[float, float, float] = (0.12, 0.14, 0.18),
    accent: tuple[float, float, float] | None = None,
):
    """Place a scaled Iris quadcopter STL at ``position``, nose along ``heading_xy``.

    The mesh is centred, uniformly scaled so its longest XY span equals
    ``target_span_m`` (metres in the local CFD/WRF frame), then yawed so
    model +X aligns with the route heading.
    """
    if not uav_stl.is_file():
        raise FileNotFoundError(uav_stl)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(uav_stl))
    reader.Update()
    poly = reader.GetOutput()
    b = poly.GetBounds()  # xmin,xmax,ymin,ymax,zmin,zmax
    cx = 0.5 * (b[0] + b[1])
    cy = 0.5 * (b[2] + b[3])
    cz = 0.5 * (b[4] + b[5])
    span = max(b[1] - b[0], b[3] - b[2], 1e-6)
    scale = float(target_span_m) / span

    dx, dy = heading_xy
    yaw_deg = float(np.degrees(np.arctan2(dy, dx)))  # 0 = +X

    tform = vtk.vtkTransform()
    tform.PostMultiply()
    tform.Translate(-cx, -cy, -cz)
    tform.Scale(scale, scale, scale)
    # Lift slightly so gear clears the route tube
    tform.Translate(0.0, 0.0, 0.15 * target_span_m)
    tform.RotateZ(yaw_deg)
    tform.Translate(float(position[0]), float(position[1]), float(position[2]))

    tf = vtk.vtkTransformPolyDataFilter()
    tf.SetInputConnection(reader.GetOutputPort())
    tf.SetTransform(tform)
    tf.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(tf.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*(accent or color))
    prop.SetOpacity(1.0)
    prop.SetAmbient(0.40)
    prop.SetDiffuse(0.70)
    prop.SetSpecular(0.35)
    prop.SetSpecularPower(40.0)
    print(
        f"[3D] UAV at ({position[0]:.0f},{position[1]:.0f},{position[2]:.0f})  "
        f"yaw={yaw_deg:.1f}°  span≈{target_span_m:.0f} m  (scale={scale:.1f}×)",
        flush=True,
    )
    return actor


def load_buildings_actor(vtk, stl_path: Path):
    """Opaque light-gray buildings from binary STL, clipped to view box."""
    if not stl_path.is_file():
        raise FileNotFoundError(stl_path)

    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()

    # Clip to the reference framing box (keeps memory / draw cost down)
    box = vtk.vtkBox()
    box.SetBounds(XLIM[0], XLIM[1], YLIM[0], YLIM[1], ZLIM[0], ZLIM[1] + 50.0)

    clip = vtk.vtkClipPolyData()
    clip.SetInputConnection(reader.GetOutputPort())
    clip.SetClipFunction(box)
    clip.InsideOutOn()
    clip.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(clip.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Light gray, fully opaque — matches the reference
    prop = actor.GetProperty()
    prop.SetColor(0.72, 0.72, 0.72)
    prop.SetOpacity(1.0)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.65)
    prop.SetSpecular(0.08)
    prop.SetSpecularPower(12.0)
    return actor


def make_cube_axes(vtk, renderer):
    """White wireframe domain box with axis labels (VTK CubeAxesActor)."""
    axes = vtk.vtkCubeAxesActor()
    axes.SetBounds(XLIM[0], XLIM[1], YLIM[0], YLIM[1], ZLIM[0], ZLIM[1])
    axes.SetCamera(renderer.GetActiveCamera())
    axes.SetXTitle("X Axis")
    axes.SetYTitle("Y Axis")
    axes.SetZTitle("Z Axis")
    axes.SetXLabelFormat("%-#6.0f")
    axes.SetYLabelFormat("%-#6.0f")
    axes.SetZLabelFormat("%-#6.0f")
    # Prevent VTK from rewriting labels as "(x10³)" etc.
    try:
        axes.SetLabelScaling(False, 0, 0, 0)
    except Exception:
        pass
    axes.SetFlyModeToOuterEdges()
    axes.SetGridLineLocation(axes.VTK_GRID_LINES_FURTHEST)

    for i in range(3):
        axes.GetTitleTextProperty(i).SetColor(1.0, 1.0, 1.0)
        axes.GetLabelTextProperty(i).SetColor(1.0, 1.0, 1.0)
        axes.GetTitleTextProperty(i).SetFontFamilyToArial()
        axes.GetLabelTextProperty(i).SetFontFamilyToArial()
        axes.GetTitleTextProperty(i).SetFontSize(16)
        axes.GetLabelTextProperty(i).SetFontSize(14)
        axes.GetTitleTextProperty(i).BoldOn()

    axes.DrawXGridlinesOn()
    axes.DrawYGridlinesOn()
    axes.DrawZGridlinesOn()
    axes.GetXAxesGridlinesProperty().SetColor(0.65, 0.65, 0.65)
    axes.GetYAxesGridlinesProperty().SetColor(0.65, 0.65, 0.65)
    axes.GetZAxesGridlinesProperty().SetColor(0.65, 0.65, 0.65)

    for getter in (
        axes.GetXAxesLinesProperty,
        axes.GetYAxesLinesProperty,
        axes.GetZAxesLinesProperty,
    ):
        getter().SetColor(1.0, 1.0, 1.0)
        getter().SetLineWidth(1.4)

    return axes


def make_orientation_marker(vtk, renderer, interactor=None):
    """Small triad in the lower-left corner (R / Y / G as in the reference)."""
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(1.0, 1.0, 1.0)
    axes.SetShaftTypeToCylinder()
    axes.SetCylinderRadius(0.04)
    axes.SetConeRadius(0.25)

    # Match reference colours: X=red, Y=yellow, Z=green
    axes.GetYAxisShaftProperty().SetColor(1.0, 0.85, 0.0)
    axes.GetYAxisTipProperty().SetColor(1.0, 0.85, 0.0)
    axes.GetZAxisShaftProperty().SetColor(0.1, 0.85, 0.15)
    axes.GetZAxisTipProperty().SetColor(0.1, 0.85, 0.15)

    axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
    for caption in (
        axes.GetXAxisCaptionActor2D(),
        axes.GetYAxisCaptionActor2D(),
        axes.GetZAxisCaptionActor2D(),
    ):
        caption.GetCaptionTextProperty().SetFontSize(12)
        caption.GetCaptionTextProperty().SetColor(1, 1, 1)

    if interactor is None:
        return None, axes

    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.16, 0.16)
    widget.SetEnabled(1)
    widget.InteractiveOff()
    return widget, axes


def render_scene(
    stl_path: Path,
    out_path: Path,
    *,
    route_z: float = ROUTE_Z,
    uav_stl: Path = DEFAULT_UAV_STL,
    uav_span_m: float = DEFAULT_UAV_SPAN_M,
    width: int = 2400,
    height: int = 1600,
    camera: dict | None = None,
) -> None:
    vtk, _ = _require_vtk()
    cam_cfg = camera or CAMERA

    # Off-screen rendering (HPC / headless safe)
    if hasattr(vtk, "vtkRenderWindow"):
        pass
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.22, 0.20, 0.18)  # dark charcoal / brownish-gray

    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.AddRenderer(ren)
    ren_win.SetSize(width, height)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(ren_win)
    iren.Initialize()

    # Buildings (opaque)
    print(f"[3D] Loading buildings: {stl_path}", flush=True)
    bldg = load_buildings_actor(vtk, stl_path)
    ren.AddActor(bldg)

    # Routes at constant altitude + oversized UAV model at each start
    for name, spec in ROUTE_SPECS.items():
        xyz = build_route_xyz(spec["start"], spec["end"], route_z, ROUTE_SAMPLE_STEP)
        print(f"[3D] {name}: {len(xyz)} pts at z={route_z:.0f} m", flush=True)
        ren.AddActor(make_polyline_actor(vtk, xyz, spec["color"], width=7.0))
        heading = (
            float(spec["end"][0] - spec["start"][0]),
            float(spec["end"][1] - spec["start"][1]),
        )
        ren.AddActor(
            make_uav_actor(
                vtk,
                uav_stl,
                xyz[0],
                heading,
                target_span_m=uav_span_m,
                accent=spec["color"],
            )
        )

    # Cube axes + orientation marker
    axes = make_cube_axes(vtk, ren)
    ren.AddActor(axes)
    marker, _ = make_orientation_marker(vtk, ren, iren)

    # Camera — strict match to the reference framing
    cam = ren.GetActiveCamera()
    cam.SetPosition(*cam_cfg["position"])
    cam.SetFocalPoint(*cam_cfg["focal_point"])
    cam.SetViewUp(*cam_cfg["view_up"])
    cam.SetViewAngle(cam_cfg.get("view_angle", 22.0))
    ren.ResetCameraClippingRange()

    # Soft key light from the viewing side
    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(*cam_cfg["position"])
    light.SetFocalPoint(*cam_cfg["focal_point"])
    light.SetColor(1.0, 1.0, 1.0)
    light.SetIntensity(0.95)
    ren.AddLight(light)
    ren.SetAmbient(0.25, 0.25, 0.25)

    print(
        f"[3D] Camera pos={cam_cfg['position']}  "
        f"focal={cam_cfg['focal_point']}  view_angle={cam_cfg.get('view_angle', 22.0)}",
        flush=True,
    )
    ren_win.Render()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPNGWriter()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(ren_win)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    writer.SetFileName(str(out_path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"[3D] Saved → {out_path}", flush=True)

    # Keep marker alive until after render
    _ = marker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--canton-lod2",
        action="store_true",
        help="Use buildings.stl (default: buildings_before_canton_lod2.stl)",
    )
    p.add_argument(
        "--stl-path",
        type=Path,
        default=None,
        help="Buildings binary STL (overrides --canton-lod2)",
    )
    p.add_argument("--uav-stl", type=Path, default=DEFAULT_UAV_STL, help="UAV / Iris binary STL")
    p.add_argument(
        "--uav-span",
        type=float,
        default=DEFAULT_UAV_SPAN_M,
        help=f"Visual UAV span in metres (default {DEFAULT_UAV_SPAN_M:.0f}; exaggerated for readability)",
    )
    p.add_argument(
        "--z",
        type=float,
        default=ROUTE_Z,
        help=f"Route / UAV altitude in metres (default {ROUTE_Z:.0f})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: .../route_overview_map_3d_z{Z}.png)",
    )
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1600, help="Image height in pixels")
    p.add_argument("--cam-x", type=float, default=CAMERA["position"][0])
    p.add_argument("--cam-y", type=float, default=CAMERA["position"][1])
    p.add_argument("--cam-z", type=float, default=CAMERA["position"][2])
    p.add_argument("--focal-x", type=float, default=CAMERA["focal_point"][0])
    p.add_argument("--focal-y", type=float, default=CAMERA["focal_point"][1])
    p.add_argument("--focal-z", type=float, default=CAMERA["focal_point"][2])
    p.add_argument("--view-angle", type=float, default=CAMERA["view_angle"])
    return p.parse_args()


def default_out_path(route_z: float) -> Path:
    z_tag = int(round(route_z))
    return (
        REPO_ROOT
        / "results"
        / "uav_route_wind_shear"
        / "20250903_1200"
        / f"route_overview_map_3d_z{z_tag}.png"
    )


def resolve_stl_path(args: argparse.Namespace) -> Path:
    if args.stl_path is not None:
        return Path(args.stl_path)
    if args.canton_lod2:
        return STL_WITH_CANTON
    return STL_BEFORE_CANTON


def main() -> int:
    args = parse_args()
    route_z = float(args.z)
    stl_path = resolve_stl_path(args)
    out_path = Path(args.out) if args.out else default_out_path(route_z)
    camera = {
        "position": (args.cam_x, args.cam_y, args.cam_z),
        "focal_point": (args.focal_x, args.focal_y, args.focal_z),
        "view_up": (0.0, 0.0, 1.0),
        "view_angle": args.view_angle,
    }
    render_scene(
        stl_path,
        out_path,
        route_z=route_z,
        uav_stl=Path(args.uav_stl),
        uav_span_m=float(args.uav_span),
        width=args.width,
        height=args.height,
        camera=camera,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
