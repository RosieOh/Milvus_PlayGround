"""v1 → v2 무중단 재색인과 alias 스왑.

**핵심은 alias 다.** 서비스는 `docs_live` 같은 alias 만 보고, 실제 컬렉션은 그 뒤에서
바뀐다. Milvus 3.0.1 의 `alter_alias` 는 원자적으로 재지정되고 alias 검색이 즉시 새
컬렉션을 친다 — 실측으로 확인했다.

순서가 전부다.
  1. v2 컬렉션을 **따로** 만들고 채운다 (서비스는 여전히 v1 을 본다)
  2. **품질 게이트** — v2 를 골든셋으로 평가한다. 회귀하면 여기서 멈춘다.
     스왑 뒤에 발견하면 이미 사용자가 맞은 뒤다.
  3. alias 를 v2 로 돌린다 (원자적)
  4. 문제가 생기면 alias 를 v1 로 되돌린다 — 데이터는 그대로 있으므로 초 단위다
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pymilvus import MilvusClient


@dataclass
class GateResult:
    passed: bool
    metric: str
    baseline: float
    candidate: float
    delta: float
    threshold: float

    def reason(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        verdict = "통과" if self.passed else "중단"
        return (f"{verdict} — {self.metric} {self.baseline:.4f} → {self.candidate:.4f} "
                f"({sign}{self.delta:.4f}, 허용 {-abs(self.threshold):.4f})")


def current_target(client: MilvusClient, alias: str) -> str | None:
    """alias 가 지금 가리키는 컬렉션. 없으면 None."""
    try:
        return client.describe_alias(alias=alias).get("collection_name") or None
    except Exception:
        return None


def point_alias(client: MilvusClient, alias: str, collection: str) -> float:
    """alias 를 collection 으로 지정한다. 이미 있으면 재지정(alter), 없으면 생성.

    소요 시간을 돌려준다 — 이 값이 곧 '스왑 소요'이고, 기획서의 성공 기준이다.
    """
    t0 = time.perf_counter()
    if current_target(client, alias) is None:
        client.create_alias(collection_name=collection, alias=alias)
    else:
        client.alter_alias(collection_name=collection, alias=alias)
    return time.perf_counter() - t0


def quality_gate(baseline: float, candidate: float, threshold: float,
                 metric: str = "ndcg@10") -> GateResult:
    """회귀 판정. threshold 는 **허용 가능한 하락폭**(양수로 준다).

    D-007 — 여기 들어오는 값은 반드시 **qrels 기준** 점수다. ANN recall 이 아니다.
    인덱스가 아니라 모델이 바뀌는 상황이므로 판정 기준도 모델 품질이어야 한다.
    """
    delta = candidate - baseline
    return GateResult(
        passed=delta >= -abs(threshold),
        metric=metric, baseline=baseline, candidate=candidate,
        delta=delta, threshold=threshold,
    )
