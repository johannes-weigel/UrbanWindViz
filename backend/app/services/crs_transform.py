from __future__ import annotations

import os
from dataclasses import dataclass

from pyproj import CRS, Transformer

from ..datasources.base import BBoxData
from ..models import BBoxWgs84

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def point_utm_to_wgs84(*, x: float, y: float, utm_zone: int, northern: bool = True) -> tuple[float, float]:
    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'south': not northern
    })
    
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lon), float(lat)

def point_wgs84_to_utm(*, lon: float, lat: float, utm_zone: int, northern: bool = True) -> tuple[float, float]:
    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'south': not northern
    })
    
    transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return float(x), float(y)

def bbox_utm_to_wgs84(b: BBoxData) -> BBoxWgs84:
    data_crs_str = require_env("UWV_DATA_CRS")
    utm_crs = CRS.from_user_input(data_crs_str)
    
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    
    corners = [(b.min_x, b.min_y), (b.max_x, b.min_y), 
               (b.min_x, b.max_y), (b.max_x, b.max_y)]
    
    lons, lats = [], []
    for x, y in corners:
        lon, lat = transformer.transform(x, y)
        lons.append(float(lon))
        lats.append(float(lat))
    
    return BBoxWgs84(minLon=min(lons), maxLon=max(lons),
                     minLat=min(lats), maxLat=max(lats))

def bbox_wgs84_to_utm(b: BBoxWgs84) -> BBoxData:
    data_crs_str = require_env("UWV_DATA_CRS")
    utm_crs = CRS.from_user_input(data_crs_str)
    
    transformer = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
    
    corners = [(b.minLon, b.minLat), (b.maxLon, b.minLat), 
               (b.minLon, b.maxLat), (b.maxLon, b.maxLat)]
    
    xs, ys = [], []
    for lon, lat in corners:
        x, y = transformer.transform(lon, lat)
        xs.append(float(x))
        ys.append(float(y))
    
    return BBoxData(min_x=min(xs), max_x=max(xs),
                    min_y=min(ys), max_y=max(ys))