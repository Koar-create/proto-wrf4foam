#!/usr/bin/env python3
"""Optional: write synthetic field frames as OpenFOAM-style time directories."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import OUTPUT_ROOT, SYNTH_FIELD_SUBDIR  # noqa: E402


def write_openfoam_velocity(
    case_id: str,
    out_case: Path,
    max_frames: int | None = 10,
) -> None:
    """Write U files for a subset of synthetic frames (ParaView/ParaFoam)."""
    src = OUTPUT_ROOT / SYNTH_FIELD_SUBDIR / f"synthetic_{case_id}.h5"
    out_case.mkdir(parents=True, exist_ok=True)
    with h5py.File(src, "r") as f:
        times = f["time_s"][:]
        n_total = f["U"].shape[0]
        n = n_total if max_frames is None else min(max_frames, n_total)
        coords_x = f["coords_x"][:]
        coords_y = f["coords_y"][:]
        coords_z = f["coords_z"][:]
        for i in range(n):
            field = f["U"][i]
            tdir = out_case / f"{i:04d}"
            tdir.mkdir(parents=True, exist_ok=True)
            flat = field.transpose(1, 2, 3, 0).reshape(-1, 3)
            with (tdir / "U").open("w", encoding="utf-8") as fo:
                fo.write(
                    "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
                )
                fo.write("dimensions [0 1 -1 0 0 0 0];\n")
                fo.write(f"internalField nonuniform List<vector>\n{flat.shape[0]}\n(\n")
                for u, v, w in flat:
                    fo.write(f"({u} {v} {w})\n")
                fo.write(");\n")
            with (tdir / "time.txt").open("w") as ft:
                ft.write(f"{times[i]}\n")
    meta = out_case / "grid_meta.npz"
    np.savez(meta, coords_x=coords_x, coords_y=coords_y, coords_z=coords_z)
    print(f"Wrote {n} frames to {out_case}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True, help="Output case directory")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()
    write_openfoam_velocity(args.case, Path(args.out), max_frames=args.max_frames)


if __name__ == "__main__":
    main()
