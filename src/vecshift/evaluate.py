"""qrels 기준 검색 품질 평가.

**여기서 계산하는 것은 임베딩 모델의 품질이다** (D-007 의 M2 지표).
인덱스 품질(ANN recall — FLAT 브루트포스 대비)과 섞어 읽으면 안 된다.
인덱스 파라미터를 바꿔도 이 숫자는 거의 움직이지 않는다. 모델이 그대로이기 때문이다.
"""

from __future__ import annotations

import math


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], judged: dict[str, int], k: int) -> float:
    gains = [judged.get(d, 0) for d in ranked[:k]]
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], judged: dict[str, int], k: int) -> float:
    positives = {d for d, r in judged.items() if r > 0}
    if not positives:
        return 0.0
    return len(positives & set(ranked[:k])) / len(positives)


def aggregate(runs: dict[str, list[str]], rel: dict[str, dict[str, int]],
              k: int = 10) -> dict[str, float]:
    """질의별 점수의 산술평균. 질의 수가 적으므로(dev 213개) 표본오차를 함께 낸다."""
    nd, rc = [], []
    for qid, ranked in runs.items():
        judged = rel.get(qid, {})
        if not judged:
            continue
        nd.append(ndcg_at_k(ranked, judged, k))
        rc.append(recall_at_k(ranked, judged, k))
    n = len(nd) or 1
    mean_nd = sum(nd) / n
    mean_rc = sum(rc) / n
    return {
        "queries": len(nd),
        f"ndcg@{k}": mean_nd,
        f"recall@{k}": mean_rc,
        f"ndcg@{k}_stderr": _stderr(nd, mean_nd),
        f"recall@{k}_stderr": _stderr(rc, mean_rc),
    }


def _stderr(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))
