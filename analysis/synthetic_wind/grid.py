"""Regular-grid utilities: load legacy HDF5, voxel binning, interpolation."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .config import DX_M, Z_LEVELS, enhanced_hdf5_path, grid_meta, legacy_hdf5_path


def make_coords() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    meta = grid_meta()
    return (
        np.asarray(meta["coords_x"], dtype=np.float64),
        np.asarray(meta["coords_y"], dtype=np.float64),
        np.asarray(meta["coords_z"], dtype=np.float64),
    )


def voxel_bin_scalar(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
) -> np.ndarray:
    """Bin unstructured scalars onto a regular (nx, ny, nz) lattice."""
    nx, ny, nz = len(coords_x), len(coords_y), len(coords_z)
    dx = float(coords_x[1] - coords_x[0]) if nx > 1 else DX_M
    dy = float(coords_y[1] - coords_y[0]) if ny > 1 else DX_M

    ix = np.rint((x - coords_x[0]) / dx).astype(np.int64)
    iy = np.rint((y - coords_y[0]) / dy).astype(np.int64)
    iz = np.searchsorted(coords_z, z, side="right") - 1

    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    ix, iy, iz, values = ix[valid], iy[valid], iz[valid], values[valid]

    out_sum = np.zeros((nx, ny, nz), dtype=np.float64)
    out_cnt = np.zeros((nx, ny, nz), dtype=np.float64)
    np.add.at(out_sum, (ix, iy, iz), values)
    np.add.at(out_cnt, (ix, iy, iz), 1.0)
    with np.errstate(invalid="ignore"):
        out = np.where(out_cnt > 0, out_sum / out_cnt, np.nan)
    return out


def voxel_bin_vector(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vectors: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
) -> np.ndarray:
    """Return (3, nx, ny, nz)."""
    comps = []
    for c in range(3):
        comps.append(
            voxel_bin_scalar(x, y, z, vectors[:, c], coords_x, coords_y, coords_z)
        )
    return np.stack(comps, axis=0)


def fill_nan_nearest_3d(field: np.ndarray) -> np.ndarray:
    """Fill NaN voxels with nearest valid neighbour (for empty bins)."""
    from scipy.ndimage import distance_transform_edt

    out = field.copy()
    mask = np.isfinite(out)
    if mask.all():
        return out
    if not mask.any():
        return np.zeros_like(out)
    idx = distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return out[tuple(idx)]


def load_legacy_hdf5(case_id: str) -> dict:
    path = legacy_hdf5_path(case_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        data = {
            "U": f["U"][:].astype(np.float64),
            "k": f["k"][:].astype(np.float64),
            "coords_x": f["coords_x"][:].astype(np.float64),
            "coords_y": f["coords_y"][:].astype(np.float64),
            "coords_z": f["coords_z"][:].astype(np.float64),
            "meta": json.loads(f.attrs.get("meta", "{}")),
        }
    return data


def write_enhanced_hdf5(
    case_id: str,
    U: np.ndarray,
    k: np.ndarray,
    epsilon: np.ndarray,
    nut: np.ndarray,
    R: np.ndarray,
    L: np.ndarray,
    T: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    extra_meta: dict | None = None,
) -> Path:
    """Write full statistics bundle to processed_hdf5_full/."""
    out = enhanced_hdf5_path(case_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    meta = grid_meta()
    meta["case_id"] = case_id
    if extra_meta:
        meta.update(extra_meta)
    with h5py.File(tmp, "w") as f:
        f.create_dataset("U", data=U.astype(np.float32), compression="gzip")
        f.create_dataset("k", data=k.astype(np.float32), compression="gzip")
        f.create_dataset("epsilon", data=epsilon.astype(np.float32), compression="gzip")
        f.create_dataset("nut", data=nut.astype(np.float32), compression="gzip")
        f.create_dataset("R", data=R.astype(np.float32), compression="gzip")
        f.create_dataset("L", data=L.astype(np.float32), compression="gzip")
        f.create_dataset("T", data=T.astype(np.float32), compression="gzip")
        f.create_dataset("coords_x", data=coords_x.astype(np.float32))
        f.create_dataset("coords_y", data=coords_y.astype(np.float32))
        f.create_dataset("coords_z", data=coords_z.astype(np.float32))
        f.attrs["meta"] = json.dumps(meta)
    if out.exists():
        out.unlink()
    tmp.replace(out)
    return out


def load_enhanced_hdf5(case_id: str) -> dict:
    path = enhanced_hdf5_path(case_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        return {
            "U": f["U"][:].astype(np.float64),
            "k": f["k"][:].astype(np.float64),
            "epsilon": f["epsilon"][:].astype(np.float64),
            "nut": f["nut"][:].astype(np.float64),
            "R": f["R"][:].astype(np.float64),
            "L": f["L"][:].astype(np.float64),
            "T": f["T"][:].astype(np.float64),
            "coords_x": f["coords_x"][:].astype(np.float64),
            "coords_y": f["coords_y"][:].astype(np.float64),
            "coords_z": f["coords_z"][:].astype(np.float64),
            "meta": json.loads(f.attrs.get("meta", "{}")),
        }


def nearest_indices(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    x: float,
    y: float,
    z: float,
) -> tuple[int, int, int]:
    ix = int(np.argmin(np.abs(coords_x - x)))
    iy = int(np.argmin(np.abs(coords_y - y)))
    iz = int(np.argmin(np.abs(coords_z - z)))
    return ix, iy, iz
