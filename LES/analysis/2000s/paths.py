"""Canonical local paths for the LES 2000 s analysis (P0–P4)."""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

LES_CASE = REPO / "LES" / "experiments" / "20250901_1000_long_DFI"
LES_MEAN100 = REPO / "LES" / "experiments" / "20250901_1000"
URANS_CASE = REPO / "experiments" / "urans" / "20250901_1000_pimplefoam"
RANS_CASE = (
    REPO
    / "steady_experiments_finer_ABL"
    / "20250901_1000_two_boundaries_as_outlet"
)

STL = LES_MEAN100 / "constant" / "triSurface" / "buildings.stl"
LIDAR_DBS = REPO / "data" / "260409" / "raw" / "lidar" / "lidar_dbs095_20250901_0900-1100UTC.csv"
LIDAR_1MIN = REPO / "data" / "260409" / "raw" / "lidar" / "lidar_transient.csv"
MERGED_HOURLY = REPO / "data" / "260409" / "processed" / "merged_lidar_simulation_final.csv"
STATION_JSON = REPO / "util" / "lidar_station_info.json"
INBOX = REPO / "LES" / "docs" / "inbox"
PROTO = INBOX / "proto"

CACHE = HERE / "cache"
OUT = HERE / "out"
FIGS = REPO / "results" / "les" / "2000s"
for _d in (CACHE, OUT, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

TIMES = tuple(range(150, 2001, 50))  # 38 snapshots
PROBE_Z = (60, 80, 100, 120, 150, 200, 232, 300, 400, 500, 600)
SLICE_Z = (30, 60, 100, 120)

STATIONS = {
    "GAW103": dict(xy=(975.0, -320.0), h0=7.2, i0=0, i1=11),
    "GAW104": dict(xy=(450.0, 350.0), h0=11.6, i0=11, i1=22),
    "GAW111": dict(xy=(75.0, 30.0), h0=30.9, i0=22, i1=33),
}

# Route 2 points deleted by OpenFOAM because they sit in building solid
# (postUAV briefing §6.3). Coordinates in metres, 20 m axis.
ROUTE2_GAPS = {
    30: (-960, -940, -860, -840, -780, -720, -660, -640, -580, -560, 580, 600, 720, 740, 800, 820),
    60: (580, 600, 720, 740, 800, 820),
    120: (580, 600),
}

ROUTE1_X = tuple(range(-1500, 1501, 20))
ROUTE2_Y = tuple(range(-1000, 1001, 20))

OBS_WINDOW = ("2025-09-01 09:30:00", "2025-09-01 10:30:00")
OBS_WINDOW_B = ("2025-09-01 10:00:00", "2025-09-01 11:00:00")
OBS_WINDOW_C = ("2025-09-01 10:00:00", "2025-09-01 10:30:00")

LES_WIN_A = (1400.0, 2000.0)  # primary transient window
LES_WIN_B = (1000.0, 1400.0)
LES_WIN_FULL = (1000.0, 2000.0)
URANS_WIN = (1200.0, 1800.0)

INNER_LIM = 2500.0
INNER_DX = 10.0
OUTER_DX = 50.0
DOMAIN_LIM = 5000.0
