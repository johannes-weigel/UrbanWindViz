from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

from .base import DatasetMeta, WindDataSource, BBoxData, WindQueryPoints, WindFieldPoints
from ..utils.pod_reconstruction import reconstruct_pod_field


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

    Psi: np.ndarray

    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0


class NpyPodFilesystemSource(WindDataSource):

    def __init__(self, data_dir: str):
        self._data_dir = data_dir
        self._cache: Dict[Tuple[str, int], _LoadedHeightSlice] = {}


    def list_datasets(self) -> list[DatasetMeta]:
        """Scans filesystem for available POD datasets and returns metadata."""
        datasets: list[DatasetMeta] = []

        if not os.path.isdir(self._data_dir):
            raise RuntimeError(f"UWV_DATA_DIR does not exist or is not a directory: {self._data_dir}")

        for area in sorted(os.listdir(self._data_dir)):
            area_dir = os.path.join(self._data_dir, area)
            if not os.path.isdir(area_dir):
                continue

            heights = self._find_heights_for_area(area_dir)
            if not heights:
                continue

            sl = self._load_slice(area, heights[0])
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


    def get_wind_points(self, q: WindQueryPoints) -> WindFieldPoints:
        """Returns wind data at irregular CFD points."""
        sl = self._load_slice(q.dataset_id, q.height_m)
        
        # Filter points in bbox
        mask = (sl.x >= q.bbox.min_x) & (sl.x <= q.bbox.max_x) & \
               (sl.y >= q.bbox.min_y) & (sl.y <= q.bbox.max_y)
        idx = np.where(mask)[0]
        
        # POD reconstruction
        ux, uy, uz = reconstruct_pod_field(
            N=len(sl.x), Psi=sl.Psi, A=sl.A, Xmean=sl.Xmean,
            wdNorm=sl.wdNorm, idx=idx, 
            ws_ref=q.ws_ref, wd_ref=q.wd_ref
        )
        
        return WindFieldPoints(
            x=sl.x[idx], y=sl.y[idx],
            u=ux, v=uy, w=uz
        )
    

    def _load_slice(self, area: str, height_m: int) -> _LoadedHeightSlice:
        """
        Loads and caches POD data for a specific area and height from disk
        """
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

        Xmean = _safe_load(pick(f"Xmean_{height_m}.npy", "Xmean.npy")).astype(np.float32).reshape(-1)

        psi = _safe_load(pick(f"Psi_{height_m}.npy", "Psi.npy")).astype(np.float32)
        
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
    

    def _find_heights_for_area(self, area_dir: str) -> list[int]:
        """Finds all height levels (e.g., 70m, 100m) in an area directory."""
        heights: list[int] = []
        for hdir in sorted(os.listdir(area_dir)):
            full = os.path.join(area_dir, hdir)
            if not os.path.isdir(full):
                continue
            h = _infer_height_from_dir(hdir)
            if h is not None:
                heights.append(h)
        return heights
    