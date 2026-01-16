from ..datasources.base import WindDataSource, WindQueryPoints, WindField, BBoxData
from .resample import resample_points_to_grid
from .crs_transform import require_env
import numpy as np
from pyproj import CRS, Transformer

class WindService:
    """Business logic for wind data operations."""
    
    def __init__(self, source: WindDataSource):
        self._source = source
    
    def list_datasets(self):
        """Pass-through to data source."""
        return self._source.list_datasets()
    
    def get_wind(
        self, 
        dataset_id: str,
        height_m: int,
        bbox: BBoxData,
        nx: int,
        ny: int,
        ws_ref: float,
        wd_ref: float,
        include_coords: bool = True,
    ) -> WindField:
        """Get gridded wind field (with resampling)."""
        
        # Load points
        points = self._source.get_wind_points(WindQueryPoints(
            dataset_id=dataset_id,
            height_m=height_m,
            bbox=bbox,
            ws_ref=ws_ref,
            wd_ref=wd_ref
        ))
        
        # Interpolate grid
        grid_u, grid_v, debug = resample_points_to_grid(
            x=points.x, y=points.y, 
            u=points.u, v=points.v,
            bbox=bbox, nx=nx, ny=ny
        )
        
        # Compute statistics
        speed = np.hypot(grid_u, grid_v)
        speed_min = float(np.nanmin(speed)) if np.isfinite(speed).any() else float("nan")
        speed_max = float(np.nanmax(speed)) if np.isfinite(speed).any() else float("nan")
        
        # Compute WGS84 grid coordinates
        lon_grid = None
        lat_grid = None
        
        if include_coords:
            w = bbox.max_x - bbox.min_x
            h = bbox.max_y - bbox.min_y
            
            xs = bbox.min_x + (np.arange(nx, dtype=np.float32) + 0.5) / nx * w
            ys = bbox.min_y + (np.arange(ny, dtype=np.float32) + 0.5) / ny * h
            
            xx, yy = np.meshgrid(xs, ys)
            
            data_crs_str = require_env("UWV_CRS_WIND")
            utm_crs = CRS.from_user_input(data_crs_str)
            transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
            
            lon_grid, lat_grid = transformer.transform(xx.ravel(), yy.ravel())
            lon_grid = np.array(lon_grid, dtype=np.float32)
            lat_grid = np.array(lat_grid, dtype=np.float32)
        
        return WindField(
            u=grid_u.ravel(), 
            v=grid_v.ravel(),
            speed_min=speed_min, 
            speed_max=speed_max,
            debug=debug,
            lon=lon_grid,
            lat=lat_grid,
        )