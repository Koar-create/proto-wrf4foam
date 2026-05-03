from paraview.simple import *
import os
import sys

if len(sys.argv) != 3:
    print("Usage: pvbatch export_y_slice_as_csv.py /path/to/xxx.foam <y>")
    print("for example: pvbatch export_y_slice_as_csv.py /public/home/zhixianyang/WRF-OpenFOAM-Coupling/steady_experiments_finer_ABL/20250903_0000/myExpxx.foam 800")
    sys.exit(1)

foam_path = os.path.abspath(sys.argv[1])

if not os.path.isfile(foam_path):
    print(f"Error: file not found: {foam_path}")
    sys.exit(1)

try:
    y = float(sys.argv[2])
except ValueError:
    print(f"Error: invalid y value: {sys.argv[2]}")
    sys.exit(1)

foam_dir = os.path.dirname(foam_path)
out_dir = os.path.join(foam_dir, "postProcessing")
os.makedirs(out_dir, exist_ok=True)

# 输出文件名：800 -> y800m.csv, 800.5 -> y800.5m.csv
if y.is_integer():
    y_str = str(int(y))
else:
    y_str = str(y)

out_csv = os.path.join(out_dir, f"y{y_str}m.csv")

# 读 foam
src = OpenFOAMReader(FileName=foam_path)
src.UpdatePipeline()

# Cell Data -> Point Data
c2p = CellDatatoPointData(Input=src)
c2p.ProcessAllArrays = 1
c2p.UpdatePipeline()

# Slice: y = input y
sl = Slice(Input=c2p)
sl.SliceType = "Plane"
sl.SliceType.Normal = [0.0, 1.0, 0.0]
sl.SliceType.Origin = [0.0, y, 0.0]
sl.UpdatePipeline()

# Resample to Image
rti = ResampleToImage(Input=sl)
rti.UseInputBounds = 0
rti.SamplingDimensions = [1000, 1, 400]
rti.SamplingBounds = [-2500.0, 2500.0, y, y, 0.0, 2000.0]
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