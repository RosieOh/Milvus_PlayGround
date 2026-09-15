"""설정 로딩. 단일 진입점 — 다른 모듈은 여기서만 설정을 읽는다."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "vecshift.yml"


class Config:
    def __init__(self, data: dict[str, Any], path: Path):
        self._d = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, dotted: str) -> Any:
        val = self.get(dotted, None)
        if val is None:
            raise KeyError(f"설정 키가 없습니다: {dotted} ({self.path})")
        return val

    @property
    def active_tier_size(self) -> int:
        tier = self.require("dataset.active_tier")
        return int(self.require(f"dataset.tiers.{tier}"))

    def variant(self, name: str) -> dict[str, Any]:
        """임베딩 변형(v1/v2) 설정."""
        v = self.get(f"embedding.{name}")
        if not v:
            raise KeyError(f"embedding.{name} 설정이 없습니다")
        return dict(v)


def load(path: str | os.PathLike[str] | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {p}")
    with open(p, encoding="utf-8") as f:
        return Config(yaml.safe_load(f) or {}, p)
