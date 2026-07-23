"""OpenFOAM ASCII field readers (no fluidfoam dependency)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_VEC_HEADER = re.compile(
    r"internalField\s+nonuniform\s+List<vector>\s*\n\s*(\d+)\s*\n\s*\(",
    re.MULTILINE,
)
_SCALAR_HEADER = re.compile(
    r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\(",
    re.MULTILINE,
)
_VEC_UNIFORM = re.compile(r"internalField\s+uniform\s+\(([^)]+)\)")
_SCALAR_UNIFORM = re.compile(r"internalField\s+uniform\s+([-\d.eE+]+)")
_VEC_TOKEN = re.compile(
    r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)"
)


def _read_text(path: Path) -> str:
    with path.open("r", errors="replace") as f:
        return f.read()


def read_foam_vector_field(path: Path) -> np.ndarray:
    """Return (n_cells, 3) array."""
    content = _read_text(path)
    m = _VEC_HEADER.search(content)
    if m:
        raw = content[m.end():]
        end = raw.find("\n)")
        if end < 0:
            raise ValueError(f"Malformed vector field: {path}")
        block = raw[:end]
        vecs = _VEC_TOKEN.findall(block)
        if not vecs:
            raise ValueError(f"No vector tokens in {path}")
        return np.asarray([[float(a), float(b), float(c)] for a, b, c in vecs],
                          dtype=np.float64)
    m = _VEC_UNIFORM.search(content)
    if m:
        vals = list(map(float, m.group(1).split()))
        return np.asarray(vals, dtype=np.float64)
    raise ValueError(f"Cannot parse vector field: {path}")


def read_foam_scalar_field(path: Path) -> np.ndarray:
    """Return (n_cells,) array."""
    content = _read_text(path)
    m = _SCALAR_HEADER.search(content)
    if m:
        n_expected = int(m.group(1))
        raw = content[m.end():]
        end = raw.find("\n)")
        if end < 0:
            raise ValueError(f"Malformed scalar field: {path}")
        block = raw[:end].strip()
        vals = np.fromstring(block.replace("\n", " "), sep=" ", dtype=np.float64)
        if vals.size != n_expected:
            # fallback line-by-line for irregular whitespace
            vals = np.asarray([float(x) for x in block.split()], dtype=np.float64)
        if vals.size != n_expected:
            raise ValueError(
                f"Expected {n_expected} scalars, got {vals.size} in {path}"
            )
        return vals
    m = _SCALAR_UNIFORM.search(content)
    if m:
        return np.asarray([float(m.group(1))], dtype=np.float64)
    raise ValueError(f"Cannot parse scalar field: {path}")


def read_cell_centres(case_dir: Path) -> np.ndarray:
    """Return (n_cells, 3) cell-centre coordinates."""
    for candidate in (
        case_dir / "0" / "C",
        case_dir / "constant" / "cellCentres",
    ):
        if candidate.is_file():
            return read_foam_vector_field(candidate)
    raise FileNotFoundError(
        f"No cell centres in {case_dir} (expected 0/C or constant/cellCentres)"
    )


def read_fields_at_timestep(case_dir: Path, timestep: str = "5000") -> dict[str, np.ndarray]:
    """Read U, k, epsilon, nut at given time directory."""
    tdir = case_dir / timestep
    U = read_foam_vector_field(tdir / "U")
    k = read_foam_scalar_field(tdir / "k")
    eps = read_foam_scalar_field(tdir / "epsilon")
    nut = read_foam_scalar_field(tdir / "nut")
    if U.ndim == 1:
        U = np.tile(U, (k.size, 1))
    for name, arr in ("k", k), ("epsilon", eps), ("nut", nut):
        if arr.size == 1 and k.size > 1:
            arr = np.full(k.size, arr.item())
        if name == "k":
            k = arr
        elif name == "epsilon":
            eps = arr
        else:
            nut = arr
    return {"U": U, "k": k, "epsilon": eps, "nut": nut}
