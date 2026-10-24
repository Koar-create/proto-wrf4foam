#!/usr/bin/env python3
"""P0 unit checks: Route-2 gaps, nested-block additivity, probe layout, height map."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import numpy as np

from align import lidar_gates_in_cfd_range, z_agl
from lesio import expected_route2_gaps, load_all_routes, load_les_probes, load_urans_probes, compare_probe_xyz
from paths import CACHE, PROBE_Z, ROUTE2_GAPS, STATIONS
from stats import band_variance, structure_function, default_lag_edges


def _ok(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    return bool(cond)


def test_band_additive():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 3600, 800))
    x = np.sin(2 * np.pi * t / 90) + 0.3 * np.sin(2 * np.pi * t / 400) + rng.normal(0, 0.2, t.size)
    b = band_variance(t, x)
    return _ok("band_variance additive", abs(b["lo"] + b["mid"] + b["hi"] - b["tot"]) < 1e-9, str({k: round(b[k], 4) for k in ("lo", "mid", "hi", "tot")}))


def test_structure_fn_shape():
    t = np.arange(0, 200, 0.2)
    x = np.sin(t / 10)
    edges = default_lag_edges(1.5, 80, 12)
    D, n = structure_function(t, x, edges, min_n=10)
    return _ok("structure_function finite bins", np.isfinite(D).sum() > 3, f"nfinite={np.isfinite(D).sum()}")


def test_height_alignment():
    h, z = lidar_gates_in_cfd_range("GAW111")
    ok1 = _ok("GAW111 lowest gate ~83 m", abs(z[0] - 82.9) < 0.2, f"z0={z[0]:.1f}")
    ok2 = _ok("GAW111 60/80 m probes have no lidar gate", not np.any(np.isin(np.round(z), [60, 80])))
    ok3 = _ok("GAW103 104 m gate → 111.2 AGL", abs(z_agl(104.0, "GAW103") - 111.2) < 1e-6)
    return ok1 and ok2 and ok3


def test_probes():
    if not (CACHE / "probes_les.npz").is_file():
        print("  [SKIP] probes_les.npz not built yet")
        return True
    les = load_les_probes()
    ur = load_urans_probes()
    cmp_ = compare_probe_xyz(les["xyz"], ur["xyz"])
    ok1 = _ok("LES U shape (9500,33,3)", les["U"].shape[1:] == (33, 3) and les["U"].shape[0] >= 9000, str(les["U"].shape))
    ok2 = _ok("LES/URANS probe xyz match by index", cmp_["same_layout"], str(cmp_))
    z = les["xyz"][:11, 2]
    ok3 = _ok("LES probe z matches PROBE_Z", np.allclose(z, PROBE_Z, atol=1e-4), str(z))
    return ok1 and ok2 and ok3


def test_route2_gaps():
    if not (CACHE / "routes.npz").is_file():
        print("  [SKIP] routes.npz not built yet")
        return True
    routes = load_all_routes(roof=None)
    all_ok = True
    for z, exp in ROUTE2_GAPS.items():
        y = routes[f"r2_z{z}_y"]
        written = routes[f"r2_z{z}_written"]
        missing = np.sort(np.round(y[~written]).astype(int))
        exp_i = np.sort(np.round(expected_route2_gaps(z)).astype(int))
        U = routes[f"r2_z{z}_U"]
        nan_ok = np.all(np.isnan(U[:, ~written, :]))
        all_ok = _ok(f"Route2 z={z} gap y", np.array_equal(missing, exp_i), f"missing={missing.tolist()}") and all_ok
        all_ok = _ok(f"Route2 z={z} NaN in gaps", nan_ok) and all_ok
        # STL solid should cover the gaps
        solid = routes[f"r2_z{z}_solid"]
        gap_solid = solid[~written]
        all_ok = _ok(f"Route2 z={z} STL solid at gaps", bool(np.all(gap_solid)), f"frac={gap_solid.mean():.2f}") and all_ok
    r1w = routes["r1_z30_written"]
    all_ok = _ok("Route1 z=30 fully written (151)", int(r1w.sum()) == 151 and r1w.size == 151) and all_ok
    return all_ok


def main():
    print("P0 tests")
    results = [
        test_band_additive(),
        test_structure_fn_shape(),
        test_height_alignment(),
        test_probes(),
        test_route2_gaps(),
    ]
    nfail = sum(not r for r in results)
    print(f"\n{nfail} group(s) failed" if nfail else "\nall groups passed")
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
