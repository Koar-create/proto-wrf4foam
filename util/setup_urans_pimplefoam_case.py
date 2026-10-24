#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scaffold experiments/urans/*_pimplefoam from steady reference + hot start."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
PATCHES_INLET_OUTLET_FIX = ("west", "north")
FIELDS_5000 = ("U", "p", "k", "epsilon", "nut")
ZERO_TEMPLATE = ("C", "Cx", "Cy", "Cz")


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def copy_reference(ref: Path, case: Path) -> None:
    case.mkdir(parents=True, exist_ok=True)
    excludes = [
        "processor*",
        "postProcessing",
        "VTK",
        "CFD_*",
        "log.*",
        "logs",
        "batchScript",
        "*.out",
        "processed_hdf5*",
        "myExpxx.foam",
        "5000",
        "constant/boundaryData",
    ]
    rsync = ["rsync", "-a", "--delete"]
    for ex in excludes:
        rsync.extend(["--exclude", ex])
    rsync.append(f"{ref}/")
    rsync.append(f"{case}/")
    run(rsync, _REPO)


def hot_start_from_5000(ref: Path, case: Path) -> None:
    zero = case / "0"
    zero.mkdir(exist_ok=True)
    for name in FIELDS_5000:
        shutil.copy2(ref / "5000" / name, zero / name)
    for name in ZERO_TEMPLATE:
        src = ref / "0" / name
        dst = zero / name
        if not src.is_file():
            continue
        if dst.exists() and dst.resolve() == src.resolve():
            continue
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst, follow_symlinks=False)
    for fld in ("U", "k", "epsilon"):
        path = zero / fld
        text = path.read_text()
        text = re.sub(
            r"value\s+\$internalField;",
            "value           uniform (0 0 0);" if fld == "U" else "value           uniform 0;",
            text,
        )
        path.write_text(text)


def write_system(case: Path, delta_t: float, u_watchdog: float) -> None:
    sysd = case / "system"
    probes_src = _REPO / "LES/experiments/20250901_1000_long_DFI/system/probesLidar"
    probes_body = probes_src.read_text() if probes_src.is_file() else ""
    probes_body = probes_body.replace("writeInterval   0.2;", "writeInterval   1;")
    probes_body = probes_body.replace(
        "writeControl    adjustableRunTime;",
        "writeControl    timeStep;",
    )

    (sysd / "controlDict").write_text(
        f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2412                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     pimpleFoam;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         1800;
deltaT          {delta_t};
writeControl    adjustableRunTime;
writeInterval   60;
purgeWrite      0;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    #include "runTimeControls"

    solverInfo1
    {{
        type            solverInfo;
        libs            (utilityFunctionObjects);
        writeControl    timeStep;
        writeInterval   1;
        fields          (p U);
    }}

    fieldMinMax1
    {{
        type            fieldMinMax;
        libs            (fieldFunctionObjects);
        fields          (U k epsilon nut p);
        mode            magnitude;
        location        no;
        writeControl    timeStep;
        writeInterval   1;
        log             true;
    }}

    CourantNo1
    {{
        type            CourantNo;
        libs            (utilityFunctionObjects);
        writeControl    adjustableRunTime;
        writeInterval   60;
    }}

{probes_body}

    phi_west
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeFields     false;
        regionType      patch;
        name            west;
        operation       sum;
        fields          (phi);
        writeControl    timeStep;
        writeInterval   10;
        log             true;
    }}
    phi_east
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeFields     false;
        regionType      patch;
        name            east;
        operation       sum;
        fields          (phi);
        writeControl    timeStep;
        writeInterval   10;
        log             true;
    }}
    phi_south
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeFields     false;
        regionType      patch;
        name            south;
        operation       sum;
        fields          (phi);
        writeControl    timeStep;
        writeInterval   10;
        log             true;
    }}
    phi_north
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeFields     false;
        regionType      patch;
        name            north;
        operation       sum;
        fields          (phi);
        writeControl    timeStep;
        writeInterval   10;
        log             true;
    }}
    phi_upper
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeFields     false;
        regionType      patch;
        name            upper;
        operation       sum;
        fields          (phi);
        writeControl    timeStep;
        writeInterval   10;
        log             true;
    }}
}}
"""
    )

    (sysd / "runTimeControls").write_text(
        f"""runTimeControl_abort
{{
    type            runTimeControl;
    libs            (utilityFunctionObjects);
    conditions
    {{
        U_runaway
        {{
            type            minMax;
            mode            maximum;
            functionObject  fieldMinMax1;
            fields          ("max(mag(U))");
            value           {u_watchdog:.4g};
        }}
        k_runaway
        {{
            type            minMax;
            mode            maximum;
            functionObject  fieldMinMax1;
            fields          ("max(mag(k))");
            value           50;
        }}
        nut_runaway
        {{
            type            minMax;
            mode            maximum;
            functionObject  fieldMinMax1;
            fields          ("max(mag(nut))");
            value           2000;
        }}
    }}
    satisfiedAction abort;
    nWriteStep      0;
}}
"""
    )

    (sysd / "fvSchemes").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
ddtSchemes
{
    default         Euler;
}
gradSchemes
{
    default         cellLimited Gauss linear 0.5;
    grad(U)         cellLimited Gauss linear 0.5;
}
divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div(phi,v2)     Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
    div(nonlinearStress) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear limited 0.5;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         limited 0.5;
}
wallDist
{
    method meshWave;
}
"""
    )

    (sysd / "fvSolution").write_text(
        """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    "(U|k|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
    "(U|k|epsilon)Final"
    {
        $U;
        relTol          0;
    }
}
PIMPLE
{
    momentumPredictor   yes;
    nOuterCorrectors    15;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    turbOnFinalIterOnly false;
    residualControl
    {
        p
        {
            tolerance   1e-3;
            relTol      0;
        }
        U
        {
            tolerance   1e-4;
            relTol      0;
        }
        "(k|epsilon)"
        {
            tolerance   1e-4;
            relTol      0;
        }
    }
}
relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.7;
        epsilon         0.7;
    }
}
"""
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--case", type=Path, required=True)
    ap.add_argument("--delta-t", type=float, required=True)
    ap.add_argument("--job-name", default="URANS_pimple")
    ap.add_argument("--walltime", default="08:00:00")
    ap.add_argument("--meta", type=Path, default=None)
    args = ap.parse_args()
    ref = args.ref.resolve()
    case = args.case.resolve()
    u_watch = 40.0
    if args.meta and args.meta.is_file():
        meta = json.loads(args.meta.read_text())
        u_watch = float(meta.get("watchdog_Umax_suggest", u_watch))

    copy_reference(ref, case)
    hot_start_from_5000(ref, case)
    write_system(case, args.delta_t, u_watch)
    (case / "check").mkdir(exist_ok=True)
    print(f"Case ready at {case}", flush=True)
    print(
        "Submit with: sbatch experiments/urans/scripts/run.slurm <YYYYMMDD_HHMM>",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
