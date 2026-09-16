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


def per_query(runs: dict[str, list[str]], rel: dict[str, dict[str, int]],
              k: int = 10) -> dict[str, dict[str, float]]:
    """질의별 점수. **두 설정을 비교하려면 이게 있어야 한다** — 같은 질의를 쓰므로
    대응비교(paired)가 옳고, 독립 신뢰구간을 겹쳐 보는 것은 검정력을 버리는 짓이다."""
    out: dict[str, dict[str, float]] = {}
    for qid, ranked in runs.items():
        judged = rel.get(qid, {})
        if not judged:
            continue
        out[qid] = {f"ndcg@{k}": ndcg_at_k(ranked, judged, k),
                    f"recall@{k}": recall_at_k(ranked, judged, k)}
    return out


def aggregate(runs: dict[str, list[str]], rel: dict[str, dict[str, int]],
              k: int = 10) -> dict[str, float]:
    """질의별 점수의 산술평균. 질의 수가 적으므로(dev 213개) 표본오차를 함께 낸다."""
    pq = per_query(runs, rel, k)
    nd = [v[f"ndcg@{k}"] for v in pq.values()]
    rc = [v[f"recall@{k}"] for v in pq.values()]
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


def paired_delta(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]],
                 metric: str) -> dict[str, float]:
    """a − b 의 대응 차이. 같은 질의에서만 계산한다."""
    qids = sorted(set(a) & set(b))
    d = [a[q][metric] - b[q][metric] for q in qids]
    n = len(d)
    if n < 2:
        return {"n": n, "mean": 0.0, "stderr": 0.0, "t": 0.0,
                "wins": 0, "losses": 0, "ties": n}
    mean = sum(d) / n
    se = _stderr(d, mean)
    return {
        "n": n, "mean": mean, "stderr": se,
        "t": mean / se if se else 0.0,
        "wins": sum(1 for x in d if x > 1e-12),
        "losses": sum(1 for x in d if x < -1e-12),
        "ties": sum(1 for x in d if abs(x) <= 1e-12),
    }


def _stderr(xs: list[float], mean: float) -> float:
    if len(xs) < 2:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var / len(xs))
