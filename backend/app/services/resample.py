from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..datasources.base import BBoxData


def resample_points_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bbox: BBoxData,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x = x.astype(np.float32).reshape(-1)
    y = y.astype(np.float32).reshape(-1)
    u = u.astype(np.float32).reshape(-1)
    v = v.astype(np.float32).reshape(-1)

    grid_u = np.full((ny, nx), np.nan, dtype=np.float32)
    grid_v = np.full((ny, nx), np.nan, dtype=np.float32)
    count = np.zeros((ny, nx), dtype=np.int32)

    w = float(bbox.max_x - bbox.min_x)
    h = float(bbox.max_y - bbox.min_y)
    if w <= 0 or h <= 0:
        return grid_u, grid_v, {"resample_mode": "invalid_bbox"}

    ix = ((x - bbox.min_x) / w * nx).astype(np.int32)
    iy = ((y - bbox.min_y) / h * ny).astype(np.int32)

    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[ok]
    iy = iy[ok]
    uu = u[ok]
    vv = v[ok]

    flat = iy * nx + ix
    sum_u = np.zeros(ny * nx, dtype=np.float64)
    sum_v = np.zeros(ny * nx, dtype=np.float64)
    cnt = np.zeros(ny * nx, dtype=np.int32)

    np.add.at(sum_u, flat, uu.astype(np.float64))
    np.add.at(sum_v, flat, vv.astype(np.float64))
    np.add.at(cnt, flat, 1)

    cnt2 = cnt.reshape(ny, nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid_u = (sum_u.reshape(ny, nx) / np.maximum(cnt2, 1)).astype(np.float32)
        grid_v = (sum_v.reshape(ny, nx) / np.maximum(cnt2, 1)).astype(np.float32)

    empty = cnt2 == 0
    grid_u[empty] = np.nan
    grid_v[empty] = np.nan

    fill_passes = 4
    for _ in range(fill_passes):
        nan_mask = ~np.isfinite(grid_u) | ~np.isfinite(grid_v)
        if not nan_mask.any():
            break

        u2 = grid_u.copy()
        v2 = grid_v.copy()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            u_shift = np.roll(grid_u, shift=dy, axis=0)
            u_shift = np.roll(u_shift, shift=dx, axis=1)
            v_shift = np.roll(grid_v, shift=dy, axis=0)
            v_shift = np.roll(v_shift, shift=dx, axis=1)

            good = np.isfinite(u_shift) & np.isfinite(v_shift)
            upd = nan_mask & good
            u2[upd] = u_shift[upd]
            v2[upd] = v_shift[upd]

        grid_u = u2
        grid_v = v2

    return grid_u, grid_v, {
        "resample_mode": "bin_average_nn_fill",
        "points_used": int(uu.size),
        "empty_cells_initial": int(empty.sum()),
    }
