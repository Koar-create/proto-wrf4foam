#!/usr/bin/env python3
"""Synthesize second-scale 3-D wind fields from RANS statistics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    DEFAULT_SEED,
    FIELD_DT_S,
    FIELD_DURATION_S,
    HUB_HEIGHTS_M,
    LIDAR_PROBES,
    OUTPUT_ROOT,
    PROBE_DT_S,
    PROBE_DURATION_S,
    SYNTH_FIELD_SUBDIR,
    SYNTH_PROBE_SUBDIR,
    ensure_output_dirs,
    list_case_ids,
)
from analysis.synthetic_wind.grid import (  # noqa: E402
    load_enhanced_hdf5,
    nearest_indices,
)
from analysis.synthetic_wind.mean_field import MeanFieldInterpolator  # noqa: E402
from analysis.synthetic_wind.rfg import SyntheticTurbulenceGenerator  # noqa: E402
from analysis.synthetic_wind.statistics import (  # noqa: E402
    integral_scales,
    reynolds_stress_tensor,
)


def _probe_definitions(coords_x, coords_y, coords_z):
    probes = []
    meta = []
    for name, xy in LIDAR_PROBES.items():
        for z in HUB_HEIGHTS_M:
            ix, iy, iz = nearest_indices(coords_x, coords_y, coords_z, xy["x"], xy["y"], z)
            probes.append((ix, iy, iz))
            meta.append({"name": f"{name}_z{int(z)}", "x": xy["x"], "y": xy["y"], "z": z})
    return probes, meta


def synthesize_case(
    case_id: str,
    field_dt: float = FIELD_DT_S,
    field_duration: float = FIELD_DURATION_S,
    probe_dt: float = PROBE_DT_S,
    probe_duration: float = PROBE_DURATION_S,
    seed: int = DEFAULT_SEED,
    use_mean_interp: bool = False,
    probes_only: bool = False,
) -> tuple[Path | None, Path]:
    """Generate synthetic field + probe HDF5 for one hourly state."""
    ensure_output_dirs()
    data = load_enhanced_hdf5(case_id)
    U_mean = data["U"]
    k = data["k"]
    epsilon = data["epsilon"]
    nut = data["nut"]
    R = data["R"]
    L = data["L"]
    T = data["T"]
    coords_x = data["coords_x"]
    coords_y = data["coords_y"]
    coords_z = data["coords_z"]

    n_field = int(round(field_duration / field_dt)) + 1
    n_probe = int(round(probe_duration / probe_dt)) + 1

    gen = SyntheticTurbulenceGenerator(
        R=R, L=L, T=T,
        coords_x=coords_x, coords_y=coords_y, coords_z=coords_z,
        seed=seed,
    )

    t0 = time.time()
    field_path: Path | None = None
    if not probes_only:
        if use_mean_interp:
            case_data = {}
            for cid in list_case_ids():
                try:
                    case_data[cid] = load_enhanced_hdf5(cid)
                except FileNotFoundError:
                    continue
            interp = MeanFieldInterpolator(case_data)
            times = interp.window_times(case_id, field_duration, field_dt)
            series = np.zeros((n_field, 3) + U_mean.shape[1:], dtype=np.float32)
            gen._state = None
            for i, t_sec in enumerate(times[:n_field]):
                fields = interp.at_time(t_sec)
                R_t = reynolds_stress_tensor(
                    fields["U"], fields["k"], fields["nut"],
                    coords_x, coords_y, coords_z,
                )
                L_t, T_t = integral_scales(fields["k"], fields["epsilon"])
                gen.R, gen.L, gen.T = R_t, L_t, T_t
                gen.L_chol = __import__(
                    "analysis.synthetic_wind.statistics", fromlist=["cholesky_3x3"]
                ).cholesky_3x3(R_t)
                up = gen.step(field_dt)
                series[i] = (fields["U"] + up).astype(np.float32)
        else:
            print(f"  [{case_id}] generating {n_field} field frames @ {field_dt}s ...", flush=True)
            series = gen.generate_series(n_field, field_dt, mean_u=U_mean)

        field_path = OUTPUT_ROOT / SYNTH_FIELD_SUBDIR / f"synthetic_{case_id}.h5"
        field_tmp = field_path.with_suffix(".h5.tmp")
        print(f"  [{case_id}] writing field HDF5 ({series.nbytes/1e6:.0f} MB raw) ...", flush=True)
        with h5py.File(field_tmp, "w") as f:
            f.create_dataset(
                "U", data=series, compression="gzip", compression_opts=1,
                chunks=(1, 3, series.shape[2], series.shape[3], series.shape[4]),
            )
            f.create_dataset("time_s", data=np.arange(n_field) * field_dt)
            f.create_dataset("coords_x", data=coords_x.astype(np.float32))
            f.create_dataset("coords_y", data=coords_y.astype(np.float32))
            f.create_dataset("coords_z", data=coords_z.astype(np.float32))
            f.attrs["meta"] = json.dumps({
                "case_id": case_id,
                "dt_s": field_dt,
                "duration_s": field_duration,
                "seed": seed,
                "method": "RFG_Smirnov_Celik_2001_style",
                "mean_source": "hourly_RANS",
            })
        if field_path.exists():
            field_path.unlink()
        field_tmp.replace(field_path)

    probes, probe_meta = _probe_definitions(coords_x, coords_y, coords_z)
    print(f"  [{case_id}] generating {n_probe} probe samples @ {probe_dt}s ...", flush=True)
    probe_series = gen.generate_probe_series(
        probes, probe_meta, n_probe, probe_dt, mean_u=U_mean
    )
    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    probe_tmp = probe_path.with_suffix(".h5.tmp")
    with h5py.File(probe_tmp, "w") as f:
        f.create_dataset("U", data=probe_series, compression="gzip")
        f.create_dataset("time_s", data=np.arange(n_probe) * probe_dt)
        names = [m["name"] for m in probe_meta]
        f.create_dataset("probe_names", data=np.array(names, dtype="S"))
        f.attrs["meta"] = json.dumps({
            "case_id": case_id,
            "dt_s": probe_dt,
            "probes": probe_meta,
            "seed": seed,
        })
    if probe_path.exists():
        probe_path.unlink()
    probe_tmp.replace(probe_path)

    elapsed = time.time() - t0
    fname = field_path.name if field_path else "(skipped)"
    print(f"  [{case_id}] field {fname} probes {probe_path.name} ({elapsed:.1f}s)")
    return field_path, probe_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize turbulent wind fields.")
    parser.add_argument("--case", required=True, help="Case ID or 'representative'")
    parser.add_argument("--field-dt", type=float, default=FIELD_DT_S)
    parser.add_argument("--field-duration", type=float, default=FIELD_DURATION_S)
    parser.add_argument("--probe-dt", type=float, default=PROBE_DT_S)
    parser.add_argument("--probe-duration", type=float, default=PROBE_DURATION_S)
    parser.add_argument("--probes-only", action="store_true",
                        help="Skip 3-D field generation (faster; probes only)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mean-interp", action="store_true")
    args = parser.parse_args()

    if args.case == "representative":
        rep_file = OUTPUT_ROOT / "statistics" / "representative_cases.json"
        if not rep_file.is_file():
            print("Run classify_states.py first.")
            sys.exit(1)
        cases = json.loads(rep_file.read_text(encoding="utf-8"))["representative_cases"]
    elif args.case == "all":
        cases = list_case_ids()
    else:
        cases = [args.case]

    for cid in cases:
        try:
            from analysis.synthetic_wind.config import enhanced_hdf5_path
            if not enhanced_hdf5_path(cid).is_file():
                print(f"  [{cid}] SKIP (no enhanced HDF5; run extract_fields.py)", flush=True)
                continue
            synthesize_case(
                cid,
                field_dt=args.field_dt,
                field_duration=args.field_duration,
                probe_dt=args.probe_dt,
                probe_duration=args.probe_duration,
                seed=args.seed,
                use_mean_interp=args.mean_interp,
                probes_only=args.probes_only,
            )
        except Exception as exc:
            print(f"  [{cid}] FAILED: {exc}")


if __name__ == "__main__":
    main()
