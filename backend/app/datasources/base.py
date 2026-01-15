from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class BBoxData:
    min_x: float
    min_y: float
    max_x: float
    max_y: float


@dataclass(frozen=True)
class DatasetMeta:
    id: str
    name: str
    bbox: BBoxData
    heights_m: Sequence[int]


@dataclass(frozen=True)
class WindQuery:
    dataset_id: str
    height_m: int
    bbox: BBoxData
    nx: int
    ny: int


@dataclass(frozen=True)
class WindField:
    u: np.ndarray
    v: np.ndarray
    speed_min: float
    speed_max: float
    debug: dict


class WindDataSource(Protocol):
    def list_datasets(self) -> list[DatasetMeta]: ...
    def get_wind(self, q: WindQuery) -> WindField: ...
