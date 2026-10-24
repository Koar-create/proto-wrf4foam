#!/usr/bin/env python3
"""P0: probes, routes, lidar, height alignment, S1, sampling operator."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

from align import save_alignment_meta, write_s1
from buildings import load_roof_index
from lesio import (
    compare_probe_xyz,
    expected_route2_gaps,
    load_all_routes,
    load_les_probes,
    load_urans_probes,
)
from paths import CACHE, OUT, STL, STATIONS, TIMES
from sampling import load_lidar, sample_match_mc, station_gate, timestamps_rel_s, window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-routes", action="store_true")
    args = ap.parse_args()
    t_all = time.time()

    print("[P0] LES probes …", flush=True)
    les = load_les_probes(force=args.force)
    print(
        f"  t=[{les['t'][0]:.1f},{les['t'][-1]:.1f}] n={les['t'].size} "
        f"U{tuple(les['U'].shape)} p{tuple(les['p'].shape)}",
        flush=True,
    )

    print("[P0] URANS probes …", flush=True)
    ur = load_urans_probes(force=args.force)
    cmp_ = compare_probe_xyz(les["xyz"], ur["xyz"])
    (CACHE / "probe_coord_match.json").write_text(json.dumps(cmp_, indent=2))
    print(f"  URANS t=[{ur['t'][0]:.1f},{ur['t'][-1]:.1f}] n={ur['t'].size} match={cmp_}", flush=True)
    if not cmp_["same_layout"]:
        print("  WARNING: probe coordinates differ — later modules must match by xy, not index.", flush=True)

    print("[P0] LiDAR DBS excerpt …", flush=True)
    lidar = load_lidar(force=args.force)
    w = window(lidar)
    print(
        f"  rows={len(lidar)} window={len(w)} stations={sorted(lidar.obtid.unique())}",
        flush=True,
    )
    save_alignment_meta()

    print("[P0] STL roof index …", flush=True)
    roof = load_roof_index(STL)
    print(f"  triangles={roof.n}", flush=True)
    s1 = write_s1(roof)
    print(s1.to_string(index=False), flush=True)

    if not args.skip_routes:
        print("[P0] routes 38 × 6 …", flush=True)
        routes = load_all_routes(roof, force=args.force)
        _check_route2(routes)

    print("[P0] sampling-match smoke test (GAW111, Height=104) …", flush=True)
    g = station_gate(w, "GAW111", 104.0)
    t_rel = timestamps_rel_s(g)
    j = STATIONS["GAW111"]["i0"] + 2  # z=100
    t = les["t"]
    uh = np.hypot(les["U"][:, j, 0], les["U"][:, j, 1])
    reals = sample_match_mc(t, uh, t_rel, n=40, rng=0)
    nsamp = [np.isfinite(u).sum() for _, u in reals]
    print(f"  MC realisations={len(reals)} n_per={min(nsamp) if nsamp else 0}–{max(nsamp) if nsamp else 0}", flush=True)

    meta = dict(
        les_n=int(les["t"].size),
        urans_n=int(ur["t"].size),
        lidar_n=int(len(lidar)),
        probe_match=cmp_,
        snapshots=list(TIMES),
        elapsed_s=round(time.time() - t_all, 1),
    )
    (OUT / "p0_summary.json").write_text(json.dumps(meta, indent=2))
    print(f"[P0] done in {meta['elapsed_s']}s → {OUT}", flush=True)


def _check_route2(routes):
    ok_all = True
    for z in (30, 60, 120):
        y = routes[f"r2_z{z}_y"]
        written = routes[f"r2_z{z}_written"]
        missing_y = y[~written]
        exp = expected_route2_gaps(z)
        missing_y = np.round(missing_y).astype(int)
        exp_i = np.round(exp).astype(int)
        ok = np.array_equal(np.sort(missing_y), np.sort(exp_i))
        print(
            f"  Route2 z={z}: missing {len(missing_y)} "
            f"{'OK' if ok else 'MISMATCH '+str(missing_y.tolist())}",
            flush=True,
        )
        ok_all = ok_all and ok
    if not ok_all:
        raise SystemExit("Route 2 gap unit check failed")


if __name__ == "__main__":
    main()
