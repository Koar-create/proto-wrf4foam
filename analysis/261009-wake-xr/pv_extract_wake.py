"""Probe the two pilot RANS cases on each selected tall building's wake.

Run with paraview-env, from the repo root:

  pvbatch --force-offscreen-rendering analysis/261009-wake-xr/pv_extract_wake.py
  pvbatch --force-offscreen-rendering analysis/261009-wake-xr/pv_extract_wake.py --bids 1301

One OpenFOAM read per case. For each building the script first probes a ring
at z = H/2 to get the approach direction, then probes the wake grids from
wake_geom.sample_grids. Existing npz files are left untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from paraview import servermanager
from paraview.simple import *  # noqa: F403

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wake_geom import (  # noqa: E402
    approach_ring,
    count_points,
    frame_from_hull,
    local_to_global,
    sample_grids,
    theta_from_ring,
)

paraview.simple._DisableFirstRenderCameraReset()

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "steady_experiments_finer_ABL"
DATA = REPO / "data" / "261009"
PROC = DATA / "processed"
RAW = DATA / "raw" / "wake_boxes"
SCRATCH = RAW / "_scratch"
CASES = (
    "20250901_1000_two_boundaries_as_outlet",
    "20250908_0000_two_boundaries_as_outlet",
)


def _vtk_to_numpy(vtk_array):
    try:
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        from paraview.vtk.util.numpy_support import vtk_to_numpy
    return vtk_to_numpy(vtk_array)


def _parse_args(argv: list[str]) -> tuple[tuple[str, ...], set[int] | None]:
    cases = CASES
    bids = None
    i = 0
    while i < len(argv):
        if argv[i] == "--case":
            cases = (argv[i + 1],)
            i += 2
        elif argv[i] == "--bids":
            bids = {int(x) for x in argv[i + 1].split(",")}
            i += 2
        else:
            raise SystemExit(f"unknown argument {argv[i]}")
    return cases, bids


def _pick_region(names: list[str]) -> str | None:
    for n in names:
        if "internal" in n.lower():
            return n
    return None


def _available(reader, prop: str) -> list[str]:
    try:
        return [str(n) for n in reader.GetProperty(prop).Available]
    except Exception:
        return []


def load_case(foam: Path):
    src = OpenFOAMReader(FileName=str(foam))
    src.UpdatePipelineInformation()
    regions = _available(src, "MeshRegions")
    arrays = _available(src, "CellArrays")
    print(f"[info] regions={regions}", flush=True)
    print(f"[info] arrays={arrays}", flush=True)
    region = _pick_region(regions)
    if region:
        src.MeshRegions = [region]
    wanted = [a for a in arrays if a in ("U", "k", "nut", "epsilon")]
    if wanted:
        src.CellArrays = wanted
    times = list(src.TimestepValues) if src.TimestepValues is not None else []
    if not times:
        raise RuntimeError(f"no timesteps in {foam}")
    time = float(times[-1])
    src.UpdatePipeline(time=time)
    c2p = CellDatatoPointData(Input=src)
    c2p.ProcessAllArrays = 1
    c2p.UpdatePipeline(time=time)
    print(f"[info] time={time}", flush=True)
    return src, c2p, time


def _write_csv(path: Path, xyz: np.ndarray) -> None:
    with path.open("w") as f:
        f.write("x,y,z\n")
        np.savetxt(f, xyz, fmt="%.5f", delimiter=",")


def probe(c2p, xyz: np.ndarray, time: float, scratch_csv: Path) -> dict[str, np.ndarray]:
    _write_csv(scratch_csv, xyz)
    table = CSVReader(FileName=str(scratch_csv))
    table.UpdatePipeline()
    pts = TableToPoints(Input=table)
    pts.XColumn = "x"
    pts.YColumn = "y"
    pts.ZColumn = "z"
    pts.UpdatePipeline()
    try:
        sampled = ResampleWithDataset(SourceDataArrays=c2p, DestinationMesh=pts)
    except TypeError:
        sampled = ResampleWithDataset(Input=c2p, Source=pts)
    sampled.UpdatePipeline(time=time)
    fetched = servermanager.Fetch(sampled)
    n = fetched.GetNumberOfPoints()
    if n != len(xyz):
        Delete(sampled)
        Delete(pts)
        Delete(table)
        raise RuntimeError(f"probe returned {n} points, expected {len(xyz)}")
    got = _vtk_to_numpy(fetched.GetPoints().GetData()).reshape(-1, 3)
    if not np.allclose(got, xyz, atol=0.05):
        # TableToPoints kept the rows, but a reader may round; map back by coordinate.
        order = _match_order(xyz, got)
        got = got[order]
    else:
        order = None
    pd = fetched.GetPointData()
    out: dict[str, np.ndarray] = {}
    for name in ("U", "k", "nut", "vtkValidPointMask"):
        arr = pd.GetArray(name)
        if arr is None:
            continue
        values = _vtk_to_numpy(arr)
        if order is not None:
            values = values[order]
        out[name] = np.array(values, copy=True)
    Delete(sampled)
    Delete(pts)
    Delete(table)
    return out


def _match_order(requested: np.ndarray, fetched: np.ndarray) -> np.ndarray:
    key = np.round(fetched, 3)
    buckets: dict[tuple, list[int]] = {}
    for i, row in enumerate(key):
        buckets.setdefault((row[0], row[1], row[2]), []).append(i)
    order = np.empty(len(requested), dtype=np.int64)
    req = np.round(requested, 3)
    for i, row in enumerate(req):
        hits = buckets.get((row[0], row[1], row[2]))
        if not hits:
            raise RuntimeError("probed point missing after rounding to 1 mm")
        order[i] = hits.pop()
    return order


def _reshape(values: np.ndarray, nz: int, ny: int, nx: int) -> np.ndarray:
    if values.ndim == 1:
        return values.reshape(nz, ny, nx)
    return values.reshape(nz, ny, nx, values.shape[-1])


def load_hulls(path: Path) -> dict[int, np.ndarray]:
    blob = np.load(path)
    out = {}
    cursor = 0
    xy = blob["high_xy"]
    for bid, n in zip(blob["bid"], blob["high_len"]):
        n = int(n)
        out[int(bid)] = xy[cursor : cursor + n]
        cursor += n
    return out


def buildings(bids_filter: set[int] | None) -> list[dict]:
    rows = []
    with (PROC / "tall_buildings.csv").open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            rec = dict(zip(header, line.strip().split(",")))
            bid = int(rec["bid"])
            if bids_filter is not None and bid not in bids_filter:
                continue
            rows.append(
                {
                    "bid": bid,
                    "cx": float(rec["cx"]),
                    "cy": float(rec["cy"]),
                    "H": float(rec["H"]),
                }
            )
    return rows


def save_building(path: Path, meta: dict, grids: dict, fields: dict) -> None:
    payload = {}
    for key in ("theta", "U_approach", "n_approach", "W", "L", "x_lee", "x_wind", "H", "cx", "cy", "bid"):
        payload[key] = np.asarray(meta[key])
    payload["ex"] = np.asarray(meta["ex"], dtype=np.float64)
    payload["ey"] = np.asarray(meta["ey"], dtype=np.float64)
    for key, value in grids.items():
        payload[key] = np.asarray(value, dtype=np.float64)
    payload.update(fields)
    np.savez_compressed(path, **payload)


def run_case(case: str, bids_filter: set[int] | None, hulls: dict[int, np.ndarray]) -> None:
    foam = ABL / case / "myExpxx.foam"
    out_dir = RAW / case
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[case] {case}", flush=True)
    src, c2p, time = load_case(foam)
    scratch_csv = SCRATCH / f"{case}_points.csv"
    for b in buildings(bids_filter):
        bid = b["bid"]
        dest = out_dir / f"{bid}.npz"
        if dest.is_file() and dest.stat().st_size > 1000:
            print(f"[skip] {case} bid={bid}", flush=True)
            continue
        origin = np.array([b["cx"], b["cy"]], dtype=np.float64)
        ring = approach_ring(b["cx"], b["cy"], b["H"])
        ring_fields = probe(c2p, ring, time, scratch_csv)
        if "U" not in ring_fields:
            print(f"[fail] {case} bid={bid} ring has no U", flush=True)
            continue
        valid = ring_fields.get("vtkValidPointMask")
        if valid is None:
            valid = np.isfinite(ring_fields["U"]).all(axis=1)
        theta, speed, n_up = theta_from_ring(ring, ring_fields["U"][:, :2], valid > 0, b["cx"], b["cy"])
        if not np.isfinite(theta):
            print(f"[fail] {case} bid={bid} no approach direction (n={n_up})", flush=True)
            continue
        fr = frame_from_hull(hulls[bid], origin, theta)
        grids = sample_grids(b["H"], fr["W"], fr["x_lee"], fr["x_wind"])
        counts = count_points(grids)
        print(
            f"[probe] {case} bid={bid} H={b['H']:.0f} theta={np.degrees(theta):.1f} "
            f"W={fr['W']:.0f} L={fr['L']:.0f} n={counts['total']}",
            flush=True,
        )
        fields = {}
        groups = (
            ("center", "x_center", "y_center", "z_center"),
            ("lat", "x_lat", "y_lat", "z_lat"),
            ("cav", "x_cav", "y_cav", "z_cav"),
            ("plane", "x_plane", "y_plane", "z_plane"),
            ("amb", "x_amb", "y_amb", "z_amb"),
        )
        for name, xk, yk, zk in groups:
            xyz = local_to_global(grids[xk], grids[yk], grids[zk], origin, fr["ex"], fr["ey"])
            sampled = probe(c2p, xyz, time, scratch_csv)
            nz, ny, nx = len(grids[zk]), len(grids[yk]), len(grids[xk])
            if "U" not in sampled:
                raise RuntimeError(f"{case} bid={bid} group {name} has no U")
            fields[f"U_{name}"] = _reshape(sampled["U"], nz, ny, nx).astype(np.float32)
            mask = sampled.get("vtkValidPointMask")
            if mask is None:
                mask = np.isfinite(sampled["U"]).all(axis=-1).astype(np.uint8)
            fields[f"valid_{name}"] = _reshape(np.asarray(mask).reshape(-1), nz, ny, nx).astype(np.uint8)
            for scalar in ("k", "nut"):
                if scalar in sampled:
                    fields[f"{scalar}_{name}"] = _reshape(np.asarray(sampled[scalar]).reshape(-1), nz, ny, nx).astype(np.float32)
        meta = {
            "theta": fr["theta"],
            "U_approach": speed,
            "n_approach": n_up,
            "W": fr["W"],
            "L": fr["L"],
            "x_lee": fr["x_lee"],
            "x_wind": fr["x_wind"],
            "H": b["H"],
            "cx": b["cx"],
            "cy": b["cy"],
            "bid": bid,
            "ex": fr["ex"],
            "ey": fr["ey"],
        }
        save_building(dest, meta, grids, fields)
        print(f"[wrote] {dest} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    Delete(c2p)
    Delete(src)


def main() -> None:
    cases, bids = _parse_args(sys.argv[1:])
    SCRATCH.mkdir(parents=True, exist_ok=True)
    hulls = load_hulls(PROC / "tall_footprints.npz")
    for case in cases:
        run_case(case, bids, hulls)


if __name__ == "__main__":
    main()
