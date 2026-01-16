from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np


@dataclass(frozen=True)
class BBoxData:
    """Bounding box in data coordinates (e.g., UTM)."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class DatasetMeta:
    """Internal dataset metadata."""
    id: str
    name: str
    bbox: BBoxData
    heights_m: Sequence[int]


@dataclass(frozen=True)
class WindQueryPoints:
    """Query for wind data at irregular points."""
    dataset_id: str
    height_m: int
    bbox: BBoxData
    ws_ref: float
    wd_ref: float


@dataclass(frozen=True)
class WindFieldPoints:
    """Wind data at irregular CFD points (not gridded)."""
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray | None = None


@dataclass(frozen=True)
class WindField:
    """Gridded wind field with metadata."""
    u: np.ndarray
    v: np.ndarray
    speed_min: float
    speed_max: float
    debug: dict
    
    lon: np.ndarray | None = None
    lat: np.ndarray | None = None
    
    x_points: np.ndarray | None = None
    y_points: np.ndarray | None = None
    u_points: np.ndarray | None = None
    v_points: np.ndarray | None = None
    w_points: np.ndarray | None = None


class WindDataSource(Protocol):
    """Contract for wind data sources."""
    def list_datasets(self) -> list[DatasetMeta]: ...
    def get_wind_points(self, q: WindQueryPoints) -> WindFieldPoints: ...
