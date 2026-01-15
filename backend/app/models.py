from pydantic import BaseModel


class BBoxWgs84(BaseModel):
    """Bounding box in WGS84 (lat/lon) for API."""
    minLon: float
    maxLon: float
    minLat: float
    maxLat: float


class DatasetInfo(BaseModel):
    """API response for dataset metadata."""
    id: str
    name: str
    datasetExtent: BBoxWgs84
    availableHeightsMeters: list[int]


class WindFieldResponse(BaseModel):
    """API response for gridded wind field."""
    datasetId: str
    heightMeters: int
    bbox: BBoxWgs84
    nx: int
    ny: int
    u_b64: str
    v_b64: str
    speedMin: float
    speedMax: float
