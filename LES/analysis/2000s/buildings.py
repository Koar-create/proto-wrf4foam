"""Binary-STL roof-height index (same algorithm as the RANS UAV script)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


class BuildingRoofIndex:
    """Approximate vertical-ray roof height from a binary STL mesh."""

    def __init__(self, triangles: np.ndarray, grid_size: float = 25.0):
        self.triangles = triangles.astype(np.float64, copy=False)
        self.n = len(self.triangles)
        self.grid_size = float(grid_size)
        xy = self.triangles[:, :, :2]
        self.xy = xy
        self.z_max = self.triangles[:, :, 2].max(axis=1)
        self.xmin = xy[:, :, 0].min(axis=1)
        self.xmax = xy[:, :, 0].max(axis=1)
        self.ymin = xy[:, :, 1].min(axis=1)
        self.ymax = xy[:, :, 1].max(axis=1)
        self._hash: dict[tuple[int, int], list[int]] = {}
        gs = self.grid_size
        for i in range(self.n):
            ix0 = int(np.floor(self.xmin[i] / gs))
            ix1 = int(np.floor(self.xmax[i] / gs))
            iy0 = int(np.floor(self.ymin[i] / gs))
            iy1 = int(np.floor(self.ymax[i] / gs))
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    self._hash.setdefault((ix, iy), []).append(i)

    @staticmethod
    def _point_in_triangle_2d(px: float, py: float, tri_xy: np.ndarray) -> bool:
        (x0, y0), (x1, y1), (x2, y2) = tri_xy
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            return False
        a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
        b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
        c = 1.0 - a - b
        return (a >= -1e-9) and (b >= -1e-9) and (c >= -1e-9)

    def query_roof_height(self, x: float, y: float) -> float:
        gs = self.grid_size
        ix = int(np.floor(x / gs))
        iy = int(np.floor(y / gs))
        candidates = self._hash.get((ix, iy), [])
        if not candidates:
            return 0.0
        h_max = 0.0
        for i in candidates:
            if not (self.xmin[i] <= x <= self.xmax[i] and self.ymin[i] <= y <= self.ymax[i]):
                continue
            if self._point_in_triangle_2d(x, y, self.xy[i]):
                if self.z_max[i] > h_max:
                    h_max = float(self.z_max[i])
        return h_max

    def query_many(self, xs, ys) -> np.ndarray:
        xs = np.asarray(xs, float).ravel()
        ys = np.asarray(ys, float).ravel()
        out = np.zeros(len(xs), dtype=float)
        for i, (x, y) in enumerate(zip(xs, ys)):
            out[i] = self.query_roof_height(float(x), float(y))
        return out

    def max_in_radius(self, x: float, y: float, radius: float, step: float = 5.0) -> float:
        n = max(int(np.ceil(2.0 * radius / step)), 1)
        xs = np.linspace(x - radius, x + radius, n + 1)
        ys = np.linspace(y - radius, y + radius, n + 1)
        r2 = radius * radius
        h = 0.0
        for xx in xs:
            for yy in ys:
                if (xx - x) ** 2 + (yy - y) ** 2 <= r2:
                    hh = self.query_roof_height(float(xx), float(yy))
                    if hh > h:
                        h = hh
        return h


def load_binary_stl_triangles(stl_path: Path) -> np.ndarray:
    with open(stl_path, "rb") as f:
        header = f.read(80)
        n_raw = f.read(4)
        if len(n_raw) < 4:
            raise ValueError(f"Invalid STL: {stl_path}")
        n_tri = struct.unpack("<I", n_raw)[0]
        if header.lstrip().lower().startswith(b"solid") and n_tri == 0:
            raise ValueError(f"ASCII STL not supported: {stl_path}")
        blob = np.fromfile(f, dtype=np.uint8, count=n_tri * 50)
    if blob.size != n_tri * 50:
        raise ValueError(f"Truncated STL: {stl_path}")
    dt = np.dtype({
        "names": ["normal", "verts", "attr"],
        "formats": [("<f4", (3,)), ("<f4", (9,)), "<u2"],
        "itemsize": 50,
        "aligned": False,
    })
    rec = np.frombuffer(blob.tobytes(), dtype=dt)
    return rec["verts"].reshape(n_tri, 3, 3).astype(np.float64, copy=True)


def load_roof_index(stl_path: Path, grid_size: float = 25.0) -> BuildingRoofIndex:
    tris = load_binary_stl_triangles(stl_path)
    return BuildingRoofIndex(tris, grid_size=grid_size)
