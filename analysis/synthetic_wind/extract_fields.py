#!/usr/bin/env python3
"""Extract epsilon/nut and compute turbulence statistics onto the regular grid."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as script from repo root
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    OUTPUT_ROOT,
    list_case_ids,
    case_dir,
    enhanced_hdf5_path,
)
from analysis.synthetic_wind.foam_io import (  # noqa: E402
    read_cell_centres,
    read_foam_scalar_field,
)
from analysis.synthetic_wind.grid import (  # noqa: E402
    fill_nan_nearest_3d,
    load_legacy_hdf5,
    make_coords,
    voxel_bin_scalar,
    write_enhanced_hdf5,
)
from analysis.synthetic_wind.statistics import (  # noqa: E402
    integral_scales,
    reynolds_stress_tensor,
    summarize_case_stats,
    classify_stability,
)

CACHE_DIR = OUTPUT_ROOT / "cache"
CELL_CENTRES_CACHE = CACHE_DIR / "cell_centres.npy"
MESH_META_CACHE = CACHE_DIR / "mesh_meta.json"


def _load_cell_centres(reference_case: str) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CELL_CENTRES_CACHE.is_file() and MESH_META_CACHE.is_file():
        meta = json.loads(MESH_META_CACHE.read_text(encoding="utf-8"))
        if meta.get("reference_case") == reference_case:
            return np.load(CELL_CENTRES_CACHE)
    coords = read_cell_centres(case_dir(reference_case))
    np.save(CELL_CENTRES_CACHE, coords)
    MESH_META_CACHE.write_text(
        json.dumps({"reference_case": reference_case, "n_cells": int(coords.shape[0])}),
        encoding="utf-8",
    )
    return coords


def verify_mesh_consistency(case_id: str, reference_n: int) -> None:
    cdir = case_dir(case_id)
    path = cdir / "0" / "C"
    with path.open("r", errors="replace") as f:
        head = f.read(5000)
    import re
    m = re.search(r"nonuniform\s+List<vector>\s*\n\s*(\d+)", head)
    if not m or int(m.group(1)) != reference_n:
        raise ValueError(f"Mesh cell count mismatch in {case_id}")


def extract_one(case_id: str, timestep: str = "5000", force: bool = False) -> Path:
    out = enhanced_hdf5_path(case_id)
    if out.is_file() and not force:
        return out

    t0 = time.time()
    legacy = load_legacy_hdf5(case_id)
    U = legacy["U"]
    k = legacy["k"]
    coords_x = legacy["coords_x"]
    coords_y = legacy["coords_y"]
    coords_z = legacy["coords_z"]

    ref_case = list_case_ids()[0]
    cell_c = _load_cell_centres(ref_case)
    verify_mesh_consistency(case_id, cell_c.shape[0])

    cdir = case_dir(case_id)
    tdir = cdir / timestep
    eps_raw = read_foam_scalar_field(tdir / "epsilon")
    nut_raw = read_foam_scalar_field(tdir / "nut")
    if eps_raw.size == 1:
        eps_raw = np.full(cell_c.shape[0], eps_raw.item())
    if nut_raw.size == 1:
        nut_raw = np.full(cell_c.shape[0], nut_raw.item())

    x, y, z = cell_c[:, 0], cell_c[:, 1], cell_c[:, 2]
    epsilon = voxel_bin_scalar(x, y, z, eps_raw, coords_x, coords_y, coords_z)
    nut = voxel_bin_scalar(x, y, z, nut_raw, coords_x, coords_y, coords_z)
    epsilon = fill_nan_nearest_3d(epsilon)
    nut = fill_nan_nearest_3d(nut)

    if k.ndim == 4:
        k2 = k.squeeze(0)
    else:
        k2 = k
    R = reynolds_stress_tensor(U, k2, nut, coords_x, coords_y, coords_z)
    L, T = integral_scales(k2, epsilon)

    summary = summarize_case_stats(k2, epsilon, U, coords_z)
    stability = classify_stability(summary)

    path = write_enhanced_hdf5(
        case_id,
        U=U,
        k=k2 if k.ndim == 3 else k,
        epsilon=epsilon,
        nut=nut,
        R=R,
        L=L,
        T=T,
        coords_x=coords_x,
        coords_y=coords_y,
        coords_z=coords_z,
        extra_meta={
            "stability": stability,
            "llj": summary["llj"],
            "k_aloft": summary["k_aloft"],
            "k_max": summary["k_max"],
            "extract_elapsed_s": time.time() - t0,
        },
    )
    print(f"  [{case_id}] wrote {path} ({time.time()-t0:.1f}s, {stability})", flush=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract enhanced RANS grid statistics.")
    parser.add_argument("--case", default="all", help="Case ID or 'all'")
    parser.add_argument("--timestep", default="5000")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    if args.case == "all":
        cases = list_case_ids()
    else:
        cases = [args.case]
    if args.max_cases:
        cases = cases[: args.max_cases]

    print(f"Extracting {len(cases)} case(s)...")
    for cid in cases:
        try:
            extract_one(cid, timestep=args.timestep, force=args.force)
        except Exception as exc:
            print(f"  [{cid}] FAILED: {exc}")


if __name__ == "__main__":
    main()
