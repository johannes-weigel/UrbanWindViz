from __future__ import annotations

from ..datasources.base import WindDataSource, WindQuery


class WindService:
    def __init__(self, source: WindDataSource):
        self._source = source

    def list_datasets(self):
        return self._source.list_datasets()
    
    def get_wind(self, q: WindQuery):
        return self._source.get_wind(q)
