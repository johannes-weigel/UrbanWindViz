from ..datasources.base import WindDataSource, WindQueryPoints, WindField, BBoxData
from .resample import resample_points_to_grid
import numpy as np

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
        wd_ref: float
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
        
        return WindField(
            u=grid_u, v=grid_v,
            speed_min=speed_min, speed_max=speed_max,
            debug=debug,
            x_points=points.x, y_points=points.y,
            u_points=points.u, v_points=points.v,
            w_points=points.w
        )