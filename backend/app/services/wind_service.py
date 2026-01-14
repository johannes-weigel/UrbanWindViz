from __future__ import annotations

from ..datasources.base import WindDataSource


class WindService:
    def __init__(self, source: WindDataSource):
        self._source = source

    def list_datasets(self):
        return self._source.list_datasets()
