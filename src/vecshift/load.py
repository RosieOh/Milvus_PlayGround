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
    target: str = ""      # 이 요청이 실제로 어느 컬렉션을 쳤는가 — 스왑 검증용


@dataclass
class LoadResult:
    samples: list[Sample] = field(default_factory=list)
    started: float = 0.0
    duration: float = 0.0
    # 스왑을 감지하고 질의 벡터를 다시 만드는 데 걸린 시간들. 진짜 다운타임의 정체다.
    reresolve_ms: list[float] = field(default_factory=list)

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

    def target_timeline(self, width: float = 1.0) -> list[dict]:
        """구간별로 어느 컬렉션에 요청이 갔는지. **스왑이 실제로 일어났다는 증거**다.
        재해석 횟수만 세면 대상이 안 바뀌어도 숫자가 올라간다 — 한 번 속았다."""
        if not self.samples:
            return []
        end = max(s.t for s in self.samples)
        out, w = [], 0.0
        while w < end:
            group = [s for s in self.samples if w <= s.t < w + width]
            if group:
                counts: dict[str, int] = {}
                for g in group:
                    counts[g.target or "?"] = counts.get(g.target or "?", 0) + 1
                out.append({"t": round(w, 1), "targets": counts})
            w += width
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
        search_params: dict | None = None,
        resolve: Callable[[object], tuple[str, list[list[float]]]] | None = None,
        resolve_every: float = 0.0) -> LoadResult:
    """`duration` 초 동안 `workers` 개 스레드가 단건 검색을 계속 보낸다.

    스레드마다 클라이언트를 따로 만든다 — 하나를 공유하면 커넥션에서 직렬화돼
    동시성을 재는 게 아니라 락을 재게 된다.

    `resolve` 를 주면 **모델을 인지하는 클라이언트**가 된다. 검색이 실패했을 때
    이 함수를 불러 `(대상 컬렉션, 질의 벡터)` 를 다시 얻는다. alias 가 다른 모델의
    컬렉션으로 넘어갔을 때 클라이언트가 스스로 따라가는 경로다(D-025).
    스왑을 미리 알려주는 채널이 없어도 **에러가 곧 신호**다.

    대상을 alias 가 아니라 **구체 컬렉션 이름**으로 되돌려 주는 것이 중요하다 —
    클라이언트가 alias 를 계속 때리면 낡은 컬렉션 메타 캐시에 묶인다(D-026).

    `resolve_every` 를 주면 그 주기(초)마다 **선제적으로** 다시 해석한다.
    이게 없으면 구체 컬렉션에 고정돼 스왑을 영원히 못 본다 — 옛 컬렉션은 여전히
    정상 응답하므로 에러도 안 난다. **다운타임 0 이 아니라 그냥 옛 모델을 계속 쓰는 것**이다.
    """
    res = LoadResult()
    lock = threading.Lock()
    stop = threading.Event()
    pick = zipf_picker(len(qvecs))
    t0 = time.perf_counter()

    def worker() -> None:
        client = make_client()
        vecs = qvecs
        tgt = target
        local: list[Sample] = []
        reresolves: list[float] = []
        next_resolve = time.perf_counter() + resolve_every if resolve_every else float("inf")
        while not stop.is_set():
            now = time.perf_counter()
            if resolve is not None and now >= next_resolve:
                r0 = now
                try:
                    tgt, vecs = resolve(client)
                    reresolves.append((time.perf_counter() - r0) * 1000)
                except Exception:
                    pass
                next_resolve = time.perf_counter() + resolve_every
            i = pick()
            s0 = time.perf_counter()
            try:
                client.search(collection_name=tgt, data=[vecs[i % len(vecs)]],
                              limit=k,
                              search_params={"params": dict(search_params or {})},
                              output_fields=["docid"])
                local.append(Sample(s0 - t0, (time.perf_counter() - s0) * 1000,
                                    True, "", tgt))
            except Exception as e:
                local.append(Sample(s0 - t0, (time.perf_counter() - s0) * 1000,
                                    False, f"{type(e).__name__}: {str(e)[:80]}", tgt))
                if resolve is not None:
                    r0 = time.perf_counter()
                    try:
                        tgt, vecs = resolve(client)
                        reresolves.append((time.perf_counter() - r0) * 1000)
                    except Exception:
                        time.sleep(0.05)   # 해석도 실패하면 잠깐 물러선다
        with lock:
            res.samples.extend(local)
            res.reresolve_ms.extend(reresolves)

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
