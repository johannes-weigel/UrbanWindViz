from __future__ import annotations

from pydantic import BaseModel
from typing import List


class BBoxWgs84(BaseModel):
    minLon: float
    maxLon: float
    minLat: float
    maxLat: float


class DatasetInfo(BaseModel):
    id: str
    name: str

    datasetExtent: BBoxWgs84

    availableHeightsMeters: List[int]
    