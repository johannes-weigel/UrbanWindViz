from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import DatasetInfo, BBoxWgs84
from .dataset_registry import load_config, build_source
from .services.wind_service import WindService
from .services.crs_transform import bbox_utm_to_wgs84

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