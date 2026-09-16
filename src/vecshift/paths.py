"""캐시·산출물 경로 단일화.

**왜 모듈이 따로 있는가** — HuggingFace 와 uv 는 기본 캐시가 `~/.cache/...`,
즉 내장 디스크다. 이 프로젝트는 저장소만 외장에 두고 있어서, 아무 생각 없이
코퍼스와 모델 가중치를 받으면 내장이 찬다. 실제로 벤치가 내장을 0바이트까지
채워 Docker VM 이 read-only 로 떨어진 적이 있다 (D-011).

그래서 **HF 를 import 하기 전에** 환경변수를 박는다. import 순서가 곧 안전장치다.
`vecshift.dataset` / `vecshift.embed` 는 맨 위에서 이 모듈을 부른다.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CACHE = ROOT / "cache"          # 외장. 코퍼스·모델 가중치
DATA = ROOT / "data"            # 외장. 샘플링된 코퍼스·골든셋
RESULTS = ROOT / "results"      # 외장. 평가 결과

HF_HOME = CACHE / "huggingface"


def setup_hf_cache() -> Path:
    """HF 캐시를 외장으로 돌린다. **HF 를 import 하기 전에** 불러야 한다.

    사용자가 HF_HOME 을 이미 지정했다면 존중한다 — 의도적으로 다른 디스크를
    쓰는 경우가 있다.
    """
    if not os.environ.get("HF_HOME"):
        HF_HOME.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(HF_HOME)
    # datasets 3.x 는 HF_HOME 을 따르지만, 구버전 호환으로 함께 박아둔다.
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(Path(os.environ["HF_HOME"])))
    return Path(os.environ["HF_HOME"])


def ensure_dirs() -> None:
    for d in (CACHE, DATA, RESULTS):
        d.mkdir(parents=True, exist_ok=True)


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def require_free(path: Path, need_gb: float, what: str) -> None:
    """공간이 부족하면 **받기 전에** 멈춘다.

    받다가 중간에 터지면 어디까지 받았는지 알 수 없고, 정리도 어렵다.
    """
    have = free_gb(path)
    if have < need_gb:
        raise RuntimeError(
            f"{what}: {path} 여유 {have:.1f} GB < 필요 {need_gb:.1f} GB. "
            f"티어를 줄이거나(config: dataset.active_tier) 공간을 비우세요."
        )
