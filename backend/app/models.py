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


class WindFieldResponse(BaseModel):
    datasetId: str
    heightMeters: int
    bbox: BBoxWgs84
    nx: int
    ny: int
    
    u_b64: str
    v_b64: str
    
    speedMin: float
    speedMax: float
    