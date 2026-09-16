"""M1 — 인덱스 파라미터 스윕.

**여기서 재는 것은 인덱스 품질이다** (D-007 의 ANN recall). 같은 임베딩의 FLAT
브루트포스를 정답으로 놓고, 근사 인덱스가 그 top-k 를 얼마나 회수하는지 본다.
qrels 기준 nDCG 와 섞으면 안 된다 — 모델이 그대로이므로 그 점수는 거의 안 움직인다.

설계의 핵심은 **빌드와 검색의 분리**다. `nlist`·`M`·`efConstruction` 은 재빌드가
필요하지만 `nprobe`·`ef`·`search_list` 는 아니다. (빌드 1회) × (검색 N회) 로 묶어야
20종 이상이 현실적인 시간에 들어온다.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

from pymilvus import MilvusClient

SEARCH_KEYS = ("nprobe", "ef", "search_list", "reorder_k")


def _rebuild_index(client: MilvusClient, name: str, index_type: str,
                   metric: str, params: dict[str, Any]) -> float:
    """인덱스를 갈아끼운다. 로드를 먼저 풀지 않으면 drop 이 거부된다."""
    t0 = time.perf_counter()
    try:
        client.release_collection(collection_name=name)
    except Exception:
        pass
    try:
        client.drop_index(collection_name=name, index_name="vector")
    except Exception:
        pass

    ip = client.prepare_index_params()
    ip.add_index(field_name="vector", index_type=index_type,
                 metric_type=metric, params=dict(params))
    client.create_index(collection_name=name, index_params=ip)
    client.load_collection(collection_name=name)
    return time.perf_counter() - t0


def _search(client: MilvusClient, name: str, qvecs, k: int,
            search_params: dict[str, Any]) -> tuple[list[list[str]], float]:
    t0 = time.perf_counter()
    res = client.search(collection_name=name, data=qvecs, limit=k,
                        search_params={"params": dict(search_params)},
                        output_fields=["docid"])
    dt = time.perf_counter() - t0
    # PK 필드명이 최상위 키로 온다 — h["id"] 가 아니다.
    return [[h.get("docid") or h["entity"]["docid"] for h in hits] for hits in res], dt


def ground_truth(client: MilvusClient, name: str, qvecs, k: int, metric: str,
                 index_type: str = "FLAT") -> list[list[str]]:
    """FLAT 정답. 근사가 아니므로 이 결과가 recall 의 분모다."""
    _rebuild_index(client, name, index_type, metric, {})
    ranked, _ = _search(client, name, qvecs, k, {})
    return ranked


def ann_recall(got: list[list[str]], truth: list[list[str]]) -> float:
    """질의별 |교집합| / k 의 평균. 순서는 보지 않는다 — 회수율이 정의다."""
    if not truth:
        return 0.0
    vals = [len(set(g) & set(t)) / len(t) for g, t in zip(got, truth) if t]
    return sum(vals) / len(vals) if vals else 0.0


def run_build(client: MilvusClient, name: str, build: dict, qvecs, k: int,
              metric: str, truth: list[list[str]], repeats: int, warmup: int,
              on_result=None) -> list[dict]:
    """빌드 하나에 대해 검색 파라미터를 훑는다."""
    build_s = _rebuild_index(client, name, build["type"], metric,
                             build.get("params") or {})
    rows: list[dict] = []
    for sp in build.get("search") or [{}]:
        for _ in range(max(0, warmup)):
            _search(client, name, qvecs, k, sp)
        got, times = None, []
        for _ in range(max(1, repeats)):
            got, dt = _search(client, name, qvecs, k, sp)
            times.append(dt)
        med = statistics.median(times)
        n = len(qvecs)
        qps_all = sorted(n / t for t in times if t)
        # **대표값은 최대값(best-of-N)이다.** 중앙값이 아니다.
        # 반복 내 QPS 폭이 중앙 43 % 까지 벌어지는데, 분포를 보면 느린 쪽으로만 튄다
        # (중앙값/최대값 비율 0.934, 38개 중 25개가 0.9 이상). 간섭은 시간을 더할 뿐
        # 빼지 않으므로, 가장 빠른 실행이 간섭 없는 값에 가장 가깝다 — D-022.
        # 중앙값도 함께 남겨 둘을 맞대볼 수 있게 한다.
        row = {
            "build": build["name"],
            "index_type": build["type"],
            "build_params": build.get("params") or {},
            "search_params": sp,
            "label": f"{build['name']}/{_sp_label(sp)}",
            "build_seconds": round(build_s, 2),
            "ann_recall@%d" % k: ann_recall(got or [], truth),
            "queries": n,
            "repeats": len(times),
            "batch_seconds_median": round(med, 4),
            "qps": round(qps_all[-1], 1) if qps_all else 0.0,          # best-of-N
            "qps_median": round(n / med, 1) if med else 0.0,
            "qps_min": round(qps_all[0], 1) if qps_all else 0.0,
            "qps_max": round(qps_all[-1], 1) if qps_all else 0.0,
            "qps_spread_pct": round((qps_all[-1] - qps_all[0]) / qps_all[0] * 100, 1)
            if qps_all and qps_all[0] else 0.0,
            "ms_per_query": round(med / n * 1000, 3) if n else 0.0,
        }
        rows.append(row)
        if on_result:
            on_result(row)
    return rows


def _sp_label(sp: dict) -> str:
    """검색 파라미터를 라벨로. 두 개 이상이면 전부 보여야 한다 —
    nprobe 만 찍으면 reorder_k 가 있는 설정과 없는 설정이 같은 이름이 된다."""
    parts = [f"{k}={sp[k]}" for k in SEARCH_KEYS if k in sp]
    return ",".join(parts) if parts else "default"


def pareto(rows: list[dict], x: str, y: str) -> list[dict]:
    """x(높을수록 좋음)·y(높을수록 좋음) 기준 파레토 프론티어.

    다른 설정에 **양쪽 다** 눌리지 않는 점만 남긴다. 한쪽만 좋으면 프론티어다 —
    "무엇을 포기하고 무엇을 얻는가"가 M1 의 질문이기 때문이다.
    """
    out = []
    for r in rows:
        dominated = any(
            o is not r and o[x] >= r[x] and o[y] >= r[y]
            and (o[x] > r[x] or o[y] > r[y])
            for o in rows
        )
        if not dominated:
            out.append(r)
    return sorted(out, key=lambda r: r[x])
