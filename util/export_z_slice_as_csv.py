from paraview.simple import *
import os
import sys

if len(sys.argv) != 3:
    print("Usage: pvbatch export_z_slice_as_csv.py /path/to/xxx.foam <z>")
    print("for example: pvbatch export_z_slice_as_csv.py /public/home/zhixianyang/WRF-OpenFOAM-Coupling/steady_experiments_finer_ABL/20250903_0000/myExpxx.foam 10")
    sys.exit(1)

foam_path = os.path.abspath(sys.argv[1])

if not os.path.isfile(foam_path):
    print(f"Error: file not found: {foam_path}")
    sys.exit(1)

try:
    z = float(sys.argv[2])
except ValueError:
    print(f"Error: invalid z value: {sys.argv[2]}")
    sys.exit(1)

foam_dir = os.path.dirname(foam_path)
out_dir = os.path.join(foam_dir, "postProcessing")
os.makedirs(out_dir, exist_ok=True)

# 输出文件名：10 -> 10m.csv, 100 -> 100m.csv, 10.5 -> 10.5m.csv
if z.is_integer():
    z_str = str(int(z))
else:
    z_str = str(z)

out_csv = os.path.join(out_dir, f"{z_str}m.csv")

# 读 foam
src = OpenFOAMReader(FileName=foam_path)
src.UpdatePipeline()

# Cell Data -> Point Data
c2p = CellDatatoPointData(Input=src)
c2p.ProcessAllArrays = 1
c2p.UpdatePipeline()

# Slice: z = input z
sl = Slice(Input=c2p)
sl.SliceType = "Plane"
sl.SliceType.Normal = [0.0, 0.0, 1.0]
sl.SliceType.Origin = [0.0, 0.0, z]
sl.UpdatePipeline()

# Resample to Image
rti = ResampleToImage(Input=sl)
rti.UseInputBounds = 0
rti.SamplingDimensions = [1000, 1000, 1]
rti.SamplingBounds = [-2500.0, 2500.0, -2500.0, 2500.0, z, z]
rti.UpdatePipeline()

# Calculator: Coords = coords
calc = Calculator(Input=rti)
calc.AttributeType = "Point Data"
calc.ResultArrayName = "Coords"
calc.Function = "coords"
calc.UpdatePipeline()

# 输出 CSV
SaveData(
    out_csv,
    proxy=calc,
    Precision=6,
    FieldAssociation="Point Data",
    AddMetaData=1,
)

print(f"Saved CSV to: {out_csv}")

