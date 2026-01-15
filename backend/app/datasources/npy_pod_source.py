from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, cast

from numpy.typing import NDArray
import numpy as np

from .base import DatasetMeta, WindDataSource, BBoxData, WindQuery, WindField


def _safe_load(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def _infer_height_from_dir(dirname: str) -> Optional[int]:
    d = dirname.strip().lower()
    if d.endswith("m"):
        try:
            return int(d[:-1])
        except ValueError:
            return None
    return None

@dataclass
class _LoadedHeightSlice:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    A: np.ndarray
    Xmean: np.ndarray
    wdNorm: np.ndarray

    Psi: Optional[np.ndarray] = None

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0


class NpyPodFilesystemSource(WindDataSource):

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._cache: Dict[Tuple[str, int], _LoadedHeightSlice] = {}

    def list_datasets(self) -> list[DatasetMeta]:
        datasets: list[DatasetMeta] = []

        if not os.path.isdir(self._data_dir):
            raise RuntimeError(f"UWV_DATA_DIR does not exist or is not a directory: {self._data_dir}")

        for area in sorted(os.listdir(self._data_dir)):
            area_dir = os.path.join(self._data_dir, area)
            if not os.path.isdir(area_dir):
                continue

            heights: list[int] = []
            for hdir in sorted(os.listdir(area_dir)):
                full = os.path.join(area_dir, hdir)
                if not os.path.isdir(full):
                    continue
                h = _infer_height_from_dir(hdir)
                if h is not None:
                    heights.append(h)

            if not heights:
                continue

            sample_h = heights[0]
            sl = self._load_slice(area, sample_h)
            datasets.append(
                DatasetMeta(
                    id=area,
                    name=f"{area} (NPY/POD filesystem)",
                    bbox=BBoxData(sl.x_min, sl.y_min, sl.x_max, sl.y_max),
                    heights_m=heights,
                )
            )

        if not datasets:
            raise RuntimeError(f"No datasets found under UWV_DATA_DIR={self._data_dir}. Expected: <area>/<height>m/*.npy")
        return datasets

    def get_wind(self, q: WindQuery) -> WindField:
        sl = self._load_slice(q.dataset_id, q.height_m)

        bbox = q.bbox
        overlaps = not (
            bbox.max_x < sl.x_min or bbox.min_x > sl.x_max or 
            bbox.max_y < sl.y_min or bbox.min_y > sl.y_max
        )

        if not overlaps:
            bbox = BBoxData(min_x=sl.x_min, min_y=sl.y_min, max_x=sl.x_max, max_y=sl.y_max)

        mask = (sl.x >= bbox.min_x) & (sl.x <= bbox.max_x) & (sl.y >= bbox.min_y) & (sl.y <= bbox.max_y)
        idx = np.where(mask)[0]

        if idx.size == 0:
            return WindField(
                u=np.full((q.ny, q.nx), np.nan, dtype=np.float32),
                v=np.full((q.ny, q.nx), np.nan, dtype=np.float32),
                speed_min=float("nan"), 
                speed_max=float("nan"),
                debug={"subset_points": 0}
            )

        u = sl.Xmean[idx, 0].astype(np.float32)
        v = sl.Xmean[idx, 1].astype(np.float32)

        from ..services.resample import resample_points_to_grid
        grid_u, grid_v, debug_resample = resample_points_to_grid(
            x=sl.x[idx], y=sl.y[idx], u=u, v=v,
            bbox=bbox, nx=q.nx, ny=q.ny
        )

        speed = np.hypot(grid_u, grid_v)
        speed_min = float(np.nanmin(speed)) if np.isfinite(speed).any() else float("nan")
        speed_max = float(np.nanmax(speed)) if np.isfinite(speed).any() else float("nan")

        return WindField(
            u=grid_u, v=grid_v,
            speed_min=speed_min, 
            speed_max=speed_max,
            debug={"subset_points": int(idx.size), **debug_resample}
        )
    
    def _reconstruct_uv_points(
        self, sl: _LoadedHeightSlice, idx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        u = sl.Xmean[idx, 0].astype(np.float32)
        v = sl.Xmean[idx, 1].astype(np.float32)
        return u, v, {"recon_mode": "xmean_uv"}

    def _load_slice(self, area: str, height_m: int) -> _LoadedHeightSlice:
        key = (area, height_m)
        if key in self._cache:
            return self._cache[key]

        base = os.path.join(self._data_dir, area, f"{height_m}m")

        def pick(name1: str, name2: str) -> str:
            p1 = os.path.join(base, name1)
            p2 = os.path.join(base, name2)
            return p1 if os.path.exists(p1) else p2

        x = _safe_load(pick(f"x_{height_m}.npy", "x.npy")).astype(np.float32).reshape(-1)
        y = _safe_load(pick(f"y_{height_m}.npy", "y.npy")).astype(np.float32).reshape(-1)
        z = _safe_load(pick(f"z_{height_m}.npy", "z.npy")).astype(np.float32).reshape(-1)

        A = _safe_load(pick(f"A_{height_m}.npy", "A.npy")).astype(np.float32)
        wdNorm = _safe_load(pick(f"wdNorm_{height_m}.npy", "wdNorm.npy")).astype(np.float32).reshape(-1)

        Xmean_raw = cast(NDArray[np.float32], _safe_load(pick(f"Xmean_{height_m}.npy", "Xmean.npy")).astype(np.float32))

        if Xmean_raw.ndim == 1:
            Xmean_flat = Xmean_raw.reshape(-1)
            if Xmean_flat.size == x.size * 4:
                Xmean = Xmean_flat.reshape(x.size, 4)
            else:
                Xmean = np.zeros((x.size, 4), dtype=np.float32)

        elif Xmean_raw.ndim == 2:
            if Xmean_raw.shape[0] == x.size and Xmean_raw.shape[1] >= 2:
                if Xmean_raw.shape[1] == 2:
                    Xmean = np.concatenate([Xmean_raw, np.zeros((x.size, 2), dtype=np.float32)], axis=1)
                else:
                    Xmean = Xmean_raw[:, :4]
            else:
                Xmean = np.zeros((x.size, 4), dtype=np.float32)

        else:
            Xmean = np.zeros((x.size, 4), dtype=np.float32)

        psi = None
        for cand in [f"Psi_{height_m}.npy", f"Psi_{height_m}m.npy", "Psi.npy"]:
            p = os.path.join(base, cand)
            if os.path.exists(p):
                psi = np.load(p).astype(np.float32)
                break

        sl = _LoadedHeightSlice(
            x=x,
            y=y,
            z=z,
            A=A,
            Xmean=Xmean,
            wdNorm=wdNorm,
            Psi=psi,
        )

        sl.x_min = float(np.min(x))
        sl.x_max = float(np.max(x))
        sl.y_min = float(np.min(y))
        sl.y_max = float(np.max(y))

        self._cache[key] = sl
        return sl

    