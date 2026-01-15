from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


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


class WindDataSource(Protocol):
    def list_datasets(self) -> list[DatasetMeta]: ...
