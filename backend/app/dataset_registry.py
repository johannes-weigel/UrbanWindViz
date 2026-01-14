from __future__ import annotations

import os
from dataclasses import dataclass

from .datasources.npy_pod_source import NpyPodFilesystemSource
from .datasources.base import WindDataSource


@dataclass(frozen=True)
class AppConfig:
    data_dir: str
    source_kind: str


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config() -> AppConfig:
    data_dir = _require_env("UWV_DATA_DIR")
    source_kind = _require_env("UWV_SOURCE")
    return AppConfig(data_dir=data_dir, source_kind=source_kind)


def build_source(cfg: AppConfig) -> WindDataSource:
    if cfg.source_kind == "npy_pod":
        return NpyPodFilesystemSource(data_dir=cfg.data_dir)

    raise RuntimeError(f"Unsupported UWV_SOURCE='{cfg.source_kind}'. Supported: npy_pod")
