from dotenv import load_dotenv
load_dotenv()

import base64
import numpy as np

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import DatasetInfo, WindFieldResponse, BBoxWgs84
from .dataset_registry import load_config, build_source
from .services.wind_service import WindService
from .services.crs_transform import bbox_utm_to_wgs84, bbox_wgs84_to_utm


app = FastAPI(title="UrbanWindViz Backend (NPY/POD)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_cfg = load_config()
_service = WindService(build_source(_cfg))


@app.exception_handler(RuntimeError)
def handle_runtime_error(_: Request, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "source": _cfg.source_kind,
        "dataDir": _cfg.data_dir,
    }


@app.get("/api/datasets", response_model=list[DatasetInfo])
def list_datasets():
    return [
        DatasetInfo(
            id=m.id,
            name=m.name,
            datasetExtent=bbox_utm_to_wgs84(m.bbox),
            availableHeightsMeters=list(m.heights_m),
        )
        for m in _service.list_datasets()
    ]


@app.get("/api/wind", response_model=WindFieldResponse)
def get_wind(
    dataset_id: str = Query(..., alias="datasetId"),
    height_meters: int = Query(..., alias="heightMeters"),
    min_lon: float = Query(..., alias="minLon"),
    min_lat: float = Query(..., alias="minLat"),
    max_lon: float = Query(..., alias="maxLon"),
    max_lat: float = Query(..., alias="maxLat"),
    nx: int = Query(48, ge=4, le=1024),
    ny: int = Query(36, ge=4, le=1024),
    ws_ref: float = Query(10.0, alias="wsRef"),
    wd_ref: float = Query(270.0, alias="wdRef"),
    include_coords: bool = Query(True, alias="includeCoords"),
) -> WindFieldResponse:
    if not (min_lon < max_lon and min_lat < max_lat):
        raise HTTPException(status_code=400, detail="Invalid bbox")

    bbox_wgs84 = BBoxWgs84(minLon=min_lon, minLat=min_lat, 
                           maxLon=max_lon, maxLat=max_lat)
    bbox_data = bbox_wgs84_to_utm(bbox_wgs84)


    field = _service.get_wind(
        dataset_id=dataset_id,
        height_m=height_meters,
        bbox=bbox_data,
        nx=nx, ny=ny,
        ws_ref=ws_ref,
        wd_ref=wd_ref,
        include_coords=include_coords,
    )

    def to_b64_f32(arr: np.ndarray) -> str:
        arr32 = np.asarray(arr, dtype=np.float32)
        return base64.b64encode(arr32.tobytes(order="C")).decode("ascii")

    return WindFieldResponse(
        datasetId=dataset_id,
        heightMeters=height_meters,
        bbox=bbox_wgs84,
        nx=nx,
        ny=ny,
        u_b64=to_b64_f32(field.u),
        v_b64=to_b64_f32(field.v),
        speedMin=field.speed_min,
        speedMax=field.speed_max,
        lon_b64=to_b64_f32(field.lon) if field.lon is not None else None,
        lat_b64=to_b64_f32(field.lat) if field.lat is not None else None,
    )
