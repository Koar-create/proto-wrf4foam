"""Configuration for RANS-driven synthetic wind-field generation."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = REPO_ROOT / "steady_experiments_finer_ABL"
CASE_SUFFIX = "_two_boundaries_as_outlet"
IGNORE_DIRS = {"archive_experiments", "patch_verification", "sensitivity"}

# Regular grid (matches existing processed_hdf5 meta)
DX_M = 40.0
CORE_HALF_M = 2600.0
Z_LEVELS = [5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 150.0, 200.0,
            300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
C_MU = 0.09
NU_LAM = 1.0e-6

# Synthesis defaults (plan recommendations)
FIELD_DT_S = 0.5          # 2 Hz for full 3-D field
FIELD_DURATION_S = 600.0    # 10 min per state
PROBE_DT_S = 0.05         # 20 Hz at probes
PROBE_DURATION_S = 600.0

# Output roots
OUTPUT_ROOT = REPO_ROOT / "data" / "synthetic_wind"
ENHANCED_HDF5_SUBDIR = "processed_hdf5_full"
SYNTH_FIELD_SUBDIR = "fields"
SYNTH_PROBE_SUBDIR = "probes"
STATS_SUBDIR = "statistics"
VALIDATION_SUBDIR = "validation"

# LiDAR / hub-height probe locations (m, domain-relative)
LIDAR_PROBES = {
    "GAW103": {"x": 975.0, "y": -320.0},
    "GAW104": {"x": 450.0, "y": 350.0},
    "GAW111": {"x": 75.0, "y": 30.0},
}
HUB_HEIGHTS_M = [80.0, 100.0, 150.0]

LIDAR_CSV = REPO_ROOT / "data" / "260409" / "processed" / "merged_lidar_simulation_final.csv"

DEFAULT_SEED = 42


def list_case_ids() -> list[str]:
    """Return sorted baseline case IDs (exclude archive/sensitivity)."""
    if not CASES_ROOT.is_dir():
        return []
    out = []
    for p in sorted(CASES_ROOT.iterdir()):
        if not p.is_dir():
            continue
        if p.name in IGNORE_DIRS:
            continue
        if p.name.endswith(CASE_SUFFIX):
            out.append(p.name)
    return out


def case_dir(case_id: str) -> Path:
    return CASES_ROOT / case_id


def legacy_hdf5_path(case_id: str) -> Path:
    return case_dir(case_id) / "processed_hdf5" / f"{case_id}.h5"


def enhanced_hdf5_path(case_id: str) -> Path:
    return case_dir(case_id) / ENHANCED_HDF5_SUBDIR / f"{case_id}.h5"


def grid_meta() -> dict:
    nx = int(round(2 * CORE_HALF_M / DX_M)) + 1
    coords_x = [(-CORE_HALF_M + i * DX_M) for i in range(nx)]
    coords_y = coords_x.copy()
    return {
        "dx_m": DX_M,
        "core_half_m": CORE_HALF_M,
        "z_levels": Z_LEVELS,
        "coords_x": coords_x,
        "coords_y": coords_y,
        "coords_z": Z_LEVELS,
        # Extent actually covered by the regular sampling grid (use this for any
        # downstream consumer / LUT converter). Horizontal is uniform (dx=DX_M),
        # vertical (z) is NON-uniform — see Z_LEVELS.
        "grid_bbox": {
            "x_lo": -CORE_HALF_M, "x_hi": CORE_HALF_M,
            "y_lo": -CORE_HALF_M, "y_hi": CORE_HALF_M,
            "z_lo": float(Z_LEVELS[0]), "z_hi": float(Z_LEVELS[-1]),
            "z_uniform": False,
        },
        # Full OpenFOAM CFD domain extent (incl. sponge zones); informational only,
        # does NOT describe the sampling grid above. Kept for provenance.
        "bbox_clean": {
            "x_lo": -5100, "x_hi": 5100,
            "y_lo": -5600, "y_hi": 5600,
            "z_lo": 0.0, "z_hi": 2100.0,
        },
    }


def ensure_output_dirs() -> None:
    for sub in (SYNTH_FIELD_SUBDIR, SYNTH_PROBE_SUBDIR, STATS_SUBDIR,
                VALIDATION_SUBDIR):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
