"""Sample CFD velocity along the two UAV routes at shear neighbour heights.

One OpenFOAM read per case. Points are the grid-snapped route nodes, repeated
at z = 10, 40, 50, 80, 100, 140 m (the ±20 m neighbours of 30/60/120 m).

    pvbatch analysis/260409/uav-route_wind-exposure/sample_route_shear_pvbatch.py \\
        /path/to/myExpxx.foam /path/to/probe_points.csv /path/to/out.csv
"""

from paraview.simple import *  # noqa: F403
import os
import sys

if len(sys.argv) != 4:
    print(
        "Usage: pvbatch sample_route_shear_pvbatch.py foam.foam probe_points.csv out.csv",
        file=sys.stderr,
    )
    sys.exit(1)

foam_path, points_csv, out_csv = (os.path.abspath(a) for a in sys.argv[1:])
if not os.path.isfile(foam_path):
    print(f"Error: foam not found: {foam_path}", file=sys.stderr)
    sys.exit(1)
if not os.path.isfile(points_csv):
    print(f"Error: points not found: {points_csv}", file=sys.stderr)
    sys.exit(1)
if os.path.isfile(out_csv) and os.path.getsize(out_csv) > 1000:
    print(f"Skip existing: {out_csv}")
    sys.exit(0)

src = OpenFOAMReader(FileName=foam_path)
src.UpdatePipelineInformation()
times = list(src.TimestepValues) if src.TimestepValues is not None else []
if times:
    src.UpdatePipeline(time=times[-1])
    print(f"Using time: {times[-1]}")
else:
    src.UpdatePipeline()

c2p = CellDatatoPointData(Input=src)
c2p.ProcessAllArrays = 1
if times:
    c2p.UpdatePipeline(time=times[-1])
else:
    c2p.UpdatePipeline()

table = CSVReader(FileName=points_csv)
table.UpdatePipeline()
pts = TableToPoints(Input=table)
pts.XColumn = "x"
pts.YColumn = "y"
pts.ZColumn = "z"
if times:
    pts.UpdatePipeline(time=times[-1])
else:
    pts.UpdatePipeline()

# ParaView 5.11+ names the arguments SourceDataArrays / DestinationMesh.
# Older builds use Input / Source. Try the new names first.
try:
    sampled = ResampleWithDataset(SourceDataArrays=c2p, DestinationMesh=pts)
except TypeError:
    sampled = ResampleWithDataset(Input=c2p, Source=pts)
if times:
    sampled.UpdatePipeline(time=times[-1])
else:
    sampled.UpdatePipeline()

os.makedirs(os.path.dirname(out_csv), exist_ok=True)
SaveData(out_csv, proxy=sampled, Precision=6, FieldAssociation="Point Data")
print(f"Saved {out_csv}")
