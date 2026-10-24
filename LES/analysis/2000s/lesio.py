"""I/O: OpenFOAM probes, VTP slices, UAV routes, RANS CSVs."""
from __future__ import annotations

import base64
import re
import struct
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from paths import (
    CACHE,
    INNER_DX,
    INNER_LIM,
    LES_CASE,
    PROBE_Z,
    RANS_CASE,
    ROUTE1_X,
    ROUTE2_GAPS,
    ROUTE2_Y,
    SLICE_Z,
    STATIONS,
    TIMES,
    URANS_CASE,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "util"))
from analyze_long_dfi_probes import merge_probe_files  # noqa: E402
from analyze_probes_les import read_probes_U  # noqa: E402

PROBE_HDR = re.compile(r"# Probe\s+(\d+)\s+\(\s*([^\)]+)\)")


def parse_probe_xyz(path: Path) -> np.ndarray:
    xyz = {}
    for line in path.read_text(errors="replace").splitlines():
        m = PROBE_HDR.match(line)
        if not m:
            continue
        i = int(m.group(1))
        xyz[i] = np.array(m.group(2).split(), dtype=float)
    n = max(xyz) + 1
    out = np.full((n, 3), np.nan)
    for i, v in xyz.items():
        out[i] = v
    return out


def read_probes_scalar(path: Path):
    t_rows, v_rows = [], []
    with path.open() as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t_rows.append(float(parts[0]))
            v_rows.append([float(x) for x in parts[1:]])
    return np.asarray(t_rows), np.asarray(v_rows, float)


def merge_probe_scalars(paths: list[Path]):
    chunks = []
    for p in paths:
        if p.is_file():
            chunks.append(read_probes_scalar(p))
    if not chunks:
        raise FileNotFoundError("no scalar probe files")
    chunks.sort(key=lambda x: x[0][0])
    out_t, out_v, last = [], [], -np.inf
    for t, v in chunks:
        sel = t > last + 1e-6
        out_t.append(t[sel])
        out_v.append(v[sel])
        last = float(t[-1])
    return np.concatenate(out_t), np.concatenate(out_v)


def load_les_probes(force: bool = False) -> dict:
    dest = CACHE / "probes_les.npz"
    if dest.is_file() and not force:
        z = np.load(dest, allow_pickle=False)
        return {k: z[k] for k in z.files}
    base = LES_CASE / "postProcessing" / "probesLidar"
    u_paths = sorted(base.glob("*/U"), key=lambda p: float(p.parent.name))
    p_paths = [p.parent / "p" for p in u_paths]
    t, U = merge_probe_files(u_paths)
    tp, P = merge_probe_scalars(p_paths)
    if len(tp) != len(t) or not np.allclose(tp, t):
        P = np.interp(t, tp, P, axis=0) if len(tp) else np.full((len(t), U.shape[1]), np.nan)
    xyz = parse_probe_xyz(u_paths[0])
    np.savez_compressed(
        dest,
        t=t.astype(np.float32),
        U=U.astype(np.float32),
        p=P.astype(np.float32),
        xyz=xyz.astype(np.float32),
        z=np.asarray(PROBE_Z, np.float32),
    )
    return dict(t=t, U=U, p=P, xyz=xyz, z=np.asarray(PROBE_Z, float))


def load_urans_probes(force: bool = False) -> dict:
    dest = CACHE / "probes_urans.npz"
    if dest.is_file() and not force:
        z = np.load(dest, allow_pickle=False)
        return {k: z[k] for k in z.files}
    base = URANS_CASE / "postProcessing" / "probesLidar"
    u_paths = sorted(base.glob("*/U"), key=lambda p: float(p.parent.name))
    t, U = merge_probe_files(u_paths)
    xyz = parse_probe_xyz(u_paths[0])
    np.savez_compressed(
        dest,
        t=t.astype(np.float32),
        U=U.astype(np.float32),
        xyz=xyz.astype(np.float32),
        z=np.asarray(PROBE_Z, np.float32),
    )
    return dict(t=t, U=U, xyz=xyz, z=np.asarray(PROBE_Z, float))


def compare_probe_xyz(les_xyz: np.ndarray, urans_xyz: np.ndarray) -> dict:
    ok = les_xyz.shape == urans_xyz.shape and np.allclose(les_xyz, urans_xyz, atol=1e-4)
    dmax = float(np.nanmax(np.linalg.norm(les_xyz - urans_xyz, axis=1))) if ok or les_xyz.shape == urans_xyz.shape else np.nan
    return {
        "same_layout": bool(ok),
        "n_les": int(les_xyz.shape[0]),
        "n_urans": int(urans_xyz.shape[0]),
        "max_coord_diff_m": dmax,
        "match_by": "index" if ok else "xy",
    }


# ---------------------------------------------------------------------------
# VTP
# ---------------------------------------------------------------------------
def vtk_xml_binary_array(el: ET.Element, n: int, ncomp: int, dtype: str) -> np.ndarray:
    raw = base64.b64decode("".join((el.text or "").split()))
    header_nbytes = struct.unpack("<Q", raw[:8])[0]
    payload = raw[8 : 8 + header_nbytes]
    dt = np.dtype("<f4") if dtype == "Float32" else np.dtype("<i4")
    arr = np.frombuffer(payload, dtype=dt)
    return arr.reshape(n, ncomp) if ncomp > 1 else arr


def read_vtp_points_U(path: Path, field: str = "U"):
    tree = ET.parse(path)
    piece = tree.find(".//Piece")
    n = int(piece.attrib["NumberOfPoints"])
    pts_el = piece.find("./Points/DataArray")
    u_el = None
    for da in piece.iter("DataArray"):
        if da.attrib.get("Name") == field:
            u_el = da
            break
    if u_el is None:
        raise ValueError(f"No {field} array in {path}")
    pts = vtk_xml_binary_array(pts_el, n, 3, "Float32")
    ncomp = int(u_el.attrib.get("NumberOfComponents", "1"))
    u = vtk_xml_binary_array(u_el, n, ncomp, "Float32")
    return pts, u


def vtp_path(z: int | str, t: int) -> Path:
    if z == 100 or z == "z100":
        return LES_CASE / "postProcessing" / "surfacesU" / str(t) / "z100.vtp"
    name = z if isinstance(z, str) else f"z{z}"
    return LES_CASE / "postProcessing" / "surfacesUAV" / str(t) / f"{name}.vtp"


def inner_axes(dx: float = INNER_DX, lim: float = INNER_LIM):
    xs = np.arange(-lim, lim + 0.5 * dx, dx)
    return xs, xs.copy()


def outer_axes(dx: float = 50.0, lim: float = 5000.0):
    xs = np.arange(-lim, lim + 0.5 * dx, dx)
    return xs, xs.copy()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _route_xy_path(t: int, route: int, z: int) -> Path:
    return (
        LES_CASE
        / "postProcessing"
        / "setsRoutes"
        / str(t)
        / f"route{route}_z{z}_U.xy"
    )


def regular_route_axis(route: int):
    if route == 1:
        x = np.asarray(ROUTE1_X, float)
        return x, np.zeros_like(x)
    y = np.asarray(ROUTE2_Y, float)
    return np.full_like(y, 400.0), y


def load_one_route_xy(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def backfill_route(raw: np.ndarray, route: int, z: int, roof) -> dict:
    xs, ys = regular_route_axis(route)
    n = len(xs)
    U = np.full((n, 3), np.nan)
    written = np.zeros(n, bool)
    axis = raw[:, 0] if route == 1 else raw[:, 1]
    target = xs if route == 1 else ys
    for row in raw:
        ax = row[0] if route == 1 else row[1]
        j = int(np.round((ax - target[0]) / 20.0))
        if 0 <= j < n and abs(ax - target[j]) < 1.0:
            U[j] = row[3:6]
            written[j] = True
    roof_z = roof.query_many(xs, ys) if roof is not None else np.zeros(n)
    solid = roof_z >= float(z)
    # OpenFOAM already dropped solid points; keep NaN there even if roof query disagrees
    missing = ~written
    return dict(
        x=xs, y=ys, U=U, written=written, missing=missing,
        solid=solid, roof_z=roof_z, z=float(z), route=int(route),
    )


def load_all_routes(roof, force: bool = False) -> dict:
    dest = CACHE / "routes.npz"
    if dest.is_file() and not force:
        z = np.load(dest, allow_pickle=False)
        return {k: z[k] for k in z.files}
    times = np.asarray(TIMES, float)
    pack = {"times": times}
    for route in (1, 2):
        for z in (30, 60, 120):
            xs, ys = regular_route_axis(route)
            n = len(xs)
            U = np.full((len(times), n, 3), np.nan, np.float32)
            written = None
            roof_z = None
            for it, t in enumerate(TIMES):
                raw = load_one_route_xy(_route_xy_path(t, route, z))
                bf = backfill_route(raw, route, z, roof)
                U[it] = bf["U"]
                if written is None:
                    written = bf["written"]
                    roof_z = bf["roof_z"]
                    solid = bf["solid"]
            key = f"r{route}_z{z}"
            pack[f"{key}_x"] = xs.astype(np.float32)
            pack[f"{key}_y"] = ys.astype(np.float32)
            pack[f"{key}_U"] = U
            pack[f"{key}_written"] = written
            pack[f"{key}_solid"] = solid
            pack[f"{key}_roof"] = roof_z.astype(np.float32)
    np.savez_compressed(dest, **pack)
    return pack


def expected_route2_gaps(z: int) -> np.ndarray:
    return np.asarray(ROUTE2_GAPS[int(z)], float)


# ---------------------------------------------------------------------------
# RANS regular slices
# ---------------------------------------------------------------------------
def load_rans_horizontal(z: int):
    """Return x(1d), y(1d), U(ny,nx,3), k(ny,nx) from the RANS CSV."""
    csv = RANS_CASE / "postProcessing" / f"{int(z)}m.csv"
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    # U:0,U:1,U:2,eps,k,nut,p,mask,ghost,x,y,z
    u, v, w = data[:, 0], data[:, 1], data[:, 2]
    k = data[:, 4]
    x, y = data[:, 9], data[:, 10]
    xs = np.unique(np.round(x, 6))
    ys = np.unique(np.round(y, 6))
    nx, ny = xs.size, ys.size
    U = np.column_stack([u, v, w]).reshape(ny, nx, 3)
    K = k.reshape(ny, nx)
    return xs, ys, U, K


def station_index(name: str) -> tuple[int, int]:
    s = STATIONS[name]
    return s["i0"], s["i1"]
