"""부하 생성기 — 동시 클라이언트로 검색을 때리며 지연과 오류를 기록한다.

M1 의 QPS 는 213 질의를 **한 번의 호출**로 보낸 배치 처리량이라 용량 산정에 쓸 수
없었다(D-017). 여기서는 클라이언트마다 단건 검색을 보내므로 동시성과 네트워크 왕복이
포함된다 — 이쪽이 서비스가 보는 숫자다.

다운타임 측정의 정의:
  요청이 **실패하거나** 성공 요청 사이 간격이 임계값을 넘으면 그 구간을 정지로 본다.
  "오류 0" 만 보면 응답이 멈춰 있어도 통과해버린다.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Sample:
    t: float          # 시작 시각 (monotonic 기준 경과 초)
    ms: float         # 지연
    ok: bool
    err: str = ""


@dataclass
class LoadResult:
    samples: list[Sample] = field(default_factory=list)
    started: float = 0.0
    duration: float = 0.0

    def oks(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]

    def percentiles(self, ps=(50, 95, 99)) -> dict[str, float]:
        xs = sorted(s.ms for s in self.oks())
        if not xs:
            return {f"p{p}": 0.0 for p in ps}
        out = {}
        for p in ps:
            i = min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))
            out[f"p{p}"] = round(xs[i], 3)
        return out

    def qps(self) -> float:
        return round(len(self.oks()) / self.duration, 1) if self.duration else 0.0

    def error_rate(self) -> float:
        return round(1 - len(self.oks()) / len(self.samples), 6) if self.samples else 0.0

    def gaps(self, threshold_ms: float) -> list[tuple[float, float]]:
        """성공 응답이 threshold 이상 끊긴 구간. 이것이 체감 다운타임이다."""
        ts = sorted(s.t for s in self.oks())
        out = []
        for a, b in zip(ts, ts[1:]):
            if (b - a) * 1000 >= threshold_ms:
                out.append((round(a, 4), round(b, 4)))
        return out

    def window_stats(self, width: float = 1.0) -> list[dict]:
        """구간별 요약. 스왑 순간에 무슨 일이 있었는지는 전체 평균으로는 안 보인다."""
        if not self.samples:
            return []
        end = max(s.t for s in self.samples)
        out = []
        w = 0.0
        while w < end:
            group = [s for s in self.samples if w <= s.t < w + width]
            if group:
                ok = [g.ms for g in group if g.ok]
                xs = sorted(ok)
                out.append({
                    "t": round(w, 1),
                    "n": len(group),
                    "errors": sum(1 for g in group if not g.ok),
                    "p50": round(xs[len(xs) // 2], 3) if xs else 0.0,
                    "p99": round(xs[min(len(xs) - 1, int(0.99 * (len(xs) - 1)))], 3)
                    if xs else 0.0,
                })
            w += width
        return out


def zipf_picker(n: int, s: float = 1.1, seed: int = 42) -> Callable[[], int]:
    """Zipf 분포로 질의를 고른다. 실서비스의 질의는 균등하지 않다 —
    소수의 인기 질의가 대부분을 차지하고, 그게 캐시 거동을 바꾼다."""
    weights = [1.0 / ((i + 1) ** s) for i in range(n)]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    rng = random.Random(seed)

    def pick() -> int:
        x = rng.random()
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    return pick


def run(make_client: Callable[[], object], target: str, qvecs: list[list[float]],
        k: int, duration: float, workers: int,
        search_params: dict | None = None) -> LoadResult:
    """`duration` 초 동안 `workers` 개 스레드가 단건 검색을 계속 보낸다.

    스레드마다 클라이언트를 따로 만든다 — 하나를 공유하면 커넥션에서 직렬화돼
    동시성을 재는 게 아니라 락을 재게 된다.
    """
    res = LoadResult()
    lock = threading.Lock()
    stop = threading.Event()
    pick = zipf_picker(len(qvecs))
    t0 = time.perf_counter()

    def worker() -> None:
        client = make_client()
        local: list[Sample] = []
        while not stop.is_set():
            i = pick()
            s0 = time.perf_counter()
            try:
                client.search(collection_name=target, data=[qvecs[i]], limit=k,
                              search_params={"params": dict(search_params or {})},
                              output_fields=["docid"])
                local.append(Sample(s0 - t0, (time.perf_counter() - s0) * 1000, True))
            except Exception as e:
                local.append(Sample(s0 - t0, (time.perf_counter() - s0) * 1000,
                                    False, f"{type(e).__name__}: {str(e)[:80]}"))
        with lock:
            res.samples.extend(local)

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
    res.started = time.time()
    for t in threads:
        t.start()
    time.sleep(duration)
    stop.set()
    for t in threads:
        t.join(timeout=30)
    res.duration = time.perf_counter() - t0
    res.samples.sort(key=lambda s: s.t)
    return res
