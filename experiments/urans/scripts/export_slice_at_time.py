#!/usr/bin/env python
# pvbatch: export a reconstructed-case slice at a given time.
# Usage:
#   pvbatch export_slice_at_time.py <case.foam> z 100 1800
#   pvbatch export_slice_at_time.py <case.foam> y 800 1800
from paraview.simple import *  # noqa: F403
import os
import sys

if len(sys.argv) != 5:
    print("Usage: pvbatch export_slice_at_time.py <foam> {z|y} <coord> <time>")
    sys.exit(1)

foam_path = os.path.abspath(sys.argv[1])
axis = sys.argv[2].lower()
coord = float(sys.argv[3])
t_req = float(sys.argv[4])
if not os.path.isfile(foam_path):
    print(f"Error: file not found: {foam_path}")
    sys.exit(1)
if axis not in {"z", "y"}:
    print("axis must be z or y")
    sys.exit(1)

foam_dir = os.path.dirname(foam_path)
out_dir = os.path.join(foam_dir, "postProcessing")
os.makedirs(out_dir, exist_ok=True)
tag = str(int(coord)) if float(coord).is_integer() else str(coord)
out_csv = os.path.join(out_dir, f"{tag}m.csv" if axis == "z" else f"y{tag}m.csv")

src = OpenFOAMReader(FileName=foam_path)
src.CaseType = "Reconstructed Case"
src.MeshRegions = ["internalMesh"]
src.CellArrays = ["U"]
src.UpdatePipelineInformation()
times = list(src.TimestepValues)
if not times:
    print("Error: no timesteps on reader")
    sys.exit(1)
t_use = min(times, key=lambda t: abs(float(t) - t_req))
print(f"Using reconstructed time {t_use} (requested {t_req}); n_times={len(times)}")
src.UpdatePipeline(time=t_use)

c2p = CellDatatoPointData(Input=src)
c2p.ProcessAllArrays = 1
c2p.UpdatePipeline(time=t_use)

sl = Slice(Input=c2p)
sl.SliceType = "Plane"
if axis == "z":
    sl.SliceType.Normal = [0.0, 0.0, 1.0]
    sl.SliceType.Origin = [0.0, 0.0, coord]
else:
    sl.SliceType.Normal = [0.0, 1.0, 0.0]
    sl.SliceType.Origin = [0.0, coord, 0.0]
sl.UpdatePipeline(time=t_use)

rti = ResampleToImage(Input=sl)
rti.UseInputBounds = 0
if axis == "z":
    rti.SamplingDimensions = [1000, 1000, 1]
    rti.SamplingBounds = [-2500.0, 2500.0, -2500.0, 2500.0, coord, coord]
else:
    rti.SamplingDimensions = [1000, 1, 400]
    rti.SamplingBounds = [-2500.0, 2500.0, coord, coord, 0.0, 2000.0]
rti.UpdatePipeline(time=t_use)

calc = Calculator(Input=rti)
calc.AttributeType = "Point Data"
calc.ResultArrayName = "Coords"
calc.Function = "coords"
calc.UpdatePipeline(time=t_use)

SaveData(
    out_csv,
    proxy=calc,
    Precision=6,
    FieldAssociation="Point Data",
    AddMetaData=1,
)
print(f"Saved CSV to: {out_csv}")
