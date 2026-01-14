from __future__ import annotations

from pydantic import BaseModel
from typing import List


class RequestBox(BaseModel):
    minLon: float
    minLat: float
    maxLon: float
    maxLat: float


class DatasetInfo(BaseModel):
    id: str
    name: str

    bbox: RequestBox

    availableHeightsMeters: List[int]
    