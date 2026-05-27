from paraview.simple import *
import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a y-normal slice from OpenFOAM to CSV at a given time step."
    )
    parser.add_argument("foam", help="Path to the .foam case file")
    parser.add_argument("--y", type=float, required=True, help="Y coordinate of the slice (m)")
    parser.add_argument(
        "--time",
        type=float,
        required=True,
        help="Time step to export (e.g. 10, 100, 1000)",
    )
    return parser.parse_args()


def resolve_timestep(reader, requested_time):
    timesteps = list(reader.TimestepValues) if reader.TimestepValues else []
    if not timesteps:
        print("Warning: no timesteps reported by reader; using requested time as-is.")
        return requested_time

    if requested_time in timesteps:
        return requested_time

    closest = min(timesteps, key=lambda t: abs(t - requested_time))
    if closest != requested_time:
        print(
            f"Warning: time {requested_time} not in case; using closest available: {closest}"
        )
        print(f"Available timesteps: {timesteps}")
    return closest


def format_coord(value):
    return str(int(value)) if value == int(value) else str(value)


def main():
    args = parse_args()

    foam_path = os.path.abspath(args.foam)
    if not os.path.isfile(foam_path):
        print(f"Error: file not found: {foam_path}")
        sys.exit(1)

    y = args.y
    foam_dir = os.path.dirname(foam_path)
    out_dir = os.path.join(foam_dir, "postProcessing")
    os.makedirs(out_dir, exist_ok=True)

    time_str = format_coord(args.time)
    y_str = format_coord(y)
    out_csv = os.path.join(out_dir, f"y{y_str}m_t{time_str}.csv")

    src = OpenFOAMReader(FileName=foam_path)
    t = resolve_timestep(src, args.time)
    UpdatePipeline(t, proxy=src)

    c2p = CellDatatoPointData(Input=src)
    c2p.ProcessAllArrays = 1
    UpdatePipeline(t, proxy=c2p)

    sl = Slice(Input=c2p)
    sl.SliceType = "Plane"
    sl.SliceType.Normal = [0.0, 1.0, 0.0]
    sl.SliceType.Origin = [0.0, y, 0.0]
    UpdatePipeline(t, proxy=sl)

    rti = ResampleToImage(Input=sl)
    rti.UseInputBounds = 0
    rti.SamplingDimensions = [1000, 1, 400]
    rti.SamplingBounds = [-2500.0, 2500.0, y, y, 0.0, 2000.0]
    UpdatePipeline(t, proxy=rti)

    calc = Calculator(Input=rti)
    calc.AttributeType = "Point Data"
    calc.ResultArrayName = "Coords"
    calc.Function = "coords"
    UpdatePipeline(t, proxy=calc)

    SaveData(
        out_csv,
        proxy=calc,
        Precision=6,
        FieldAssociation="Point Data",
        AddMetaData=1,
    )

    print(f"Exported time={t}, y={y}")
    print(f"Saved CSV to: {out_csv}")


if __name__ == "__main__":
    main()
