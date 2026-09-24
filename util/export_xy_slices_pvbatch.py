"""Export horizontal wind slices with the same ParaView chain as util/export_z_slice_as_csv.py.

foam -> CellDatatoPointData -> Slice -> ResampleToImage (1000x1000, [-2500, 2500] m)
-> Calculator(coords) -> CSV.

One OpenFOAM read serves every requested height. Run with pvbatch, not python:

    pvbatch util/export_xy_slices_pvbatch.py /path/to/myExpxx.foam 30 60 120
"""

from paraview.simple import *
import os
import sys

if len(sys.argv) < 3:
    print(
        "Usage: pvbatch export_xy_slices_pvbatch.py /path/to/case.foam z [z ...]",
        file=sys.stderr,
    )
    sys.exit(1)

foam_path = os.path.abspath(sys.argv[1])
if not os.path.isfile(foam_path):
    print(f"Error: file not found: {foam_path}", file=sys.stderr)
    sys.exit(1)

heights = []
for raw in sys.argv[2:]:
    try:
        heights.append(float(raw))
    except ValueError:
        print(f"Error: invalid z value: {raw}", file=sys.stderr)
        sys.exit(1)

foam_dir = os.path.dirname(foam_path)
out_dir = os.path.join(foam_dir, "postProcessing")
os.makedirs(out_dir, exist_ok=True)


def _z_tag(z: float) -> str:
    return str(int(z)) if z.is_integer() else str(z)


pending = []
for z in heights:
    out_csv = os.path.join(out_dir, f"{_z_tag(z)}m.csv")
    if os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"Skip existing: {out_csv}")
        continue
    pending.append((z, out_csv))

if not pending:
    print("Nothing to do.")
    sys.exit(0)

src = OpenFOAMReader(FileName=foam_path)
src.UpdatePipelineInformation()
times = list(src.TimestepValues) if src.TimestepValues is not None else []
print(f"TimestepValues: {times}")
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

for z, out_csv in pending:
    sl = Slice(Input=c2p)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [0.0, 0.0, 1.0]
    sl.SliceType.Origin = [0.0, 0.0, z]
    if times:
        sl.UpdatePipeline(time=times[-1])
    else:
        sl.UpdatePipeline()

    rti = ResampleToImage(Input=sl)
    rti.UseInputBounds = 0
    rti.SamplingDimensions = [1000, 1000, 1]
    rti.SamplingBounds = [-2500.0, 2500.0, -2500.0, 2500.0, z, z]
    if times:
        rti.UpdatePipeline(time=times[-1])
    else:
        rti.UpdatePipeline()

    calc = Calculator(Input=rti)
    calc.AttributeType = "Point Data"
    calc.ResultArrayName = "Coords"
    calc.Function = "coords"
    if times:
        calc.UpdatePipeline(time=times[-1])
    else:
        calc.UpdatePipeline()

    SaveData(
        out_csv,
        proxy=calc,
        Precision=6,
        FieldAssociation="Point Data",
        AddMetaData=1,
    )
    print(f"Saved CSV to: {out_csv}")
    Delete(calc)
    Delete(rti)
    Delete(sl)

print("Done.")
