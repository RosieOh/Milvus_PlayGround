"""재색인 중 들어온 쓰기를 따라잡는다.

**저널은 벡터가 아니라 원문을 저장한다.** 벡터는 모델에 묶여 있어서, v1 으로 만든
벡터를 v2 컬렉션에 넣으면 차원부터 안 맞는다. 차원이 같은 모델이었다면 들어가긴
하지만 **다른 벡터 공간의 값이라 조용히 오염된다** — D-025 와 같은 종류의 함정이다.
원문을 들고 있다가 대상 모델로 다시 임베딩하는 것만이 안전하다.

순서:
  1. 재색인 시작 시점에 마크를 찍는다
  2. 그 뒤 모든 쓰기는 **라이브 컬렉션에 반영 + 저널에 기록**
  3. 스왑 직전 저널을 v2 로 재생한다 (v2 모델로 임베딩)
  4. 따라잡기가 끝나야 스왑한다. 못 따라잡으면 스왑하지 않는다.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class Entry:
    docid: str
    title: str
    text: str
    t: float          # 기록 시각 (경과 초)


@dataclass
class Journal:
    """재색인 시작 이후의 쓰기 기록. 스레드 안전하다."""
    entries: list[Entry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(self, e: Entry) -> None:
        with self._lock:
            self.entries.append(e)

    def snapshot(self) -> list[Entry]:
        with self._lock:
            return list(self.entries)

    def docids(self) -> set[str]:
        return {e.docid for e in self.snapshot()}


def run_prefix() -> str:
    """실행마다 고유한 docid 접두사.

    고정 접두사를 쓰면 이전 실행이 남긴 문서가 "있는 것"으로 잡혀서 **유실을 놓친다.**
    실제로 음성 대조군에서 33건 중 7건만 누락으로 나온 적이 있다 — 나머지 26건은
    앞선 실행의 잔재였다(D-029).
    """
    return f"live-{int(time.time())}"


def writer(client_factory, journal: Journal, target_of, encode, stop: threading.Event,
           rate: float, t0: float, prefix: str = "live") -> dict:
    """라이브 컬렉션에 쓰면서 저널에 남긴다.

    `target_of()` 가 매번 현재 라이브 컬렉션과 그 차원을 돌려준다 — 스왑 뒤에는
    새 컬렉션에 써야 하기 때문이다. `encode(texts, dim)` 은 해당 모델로 임베딩한다.
    인코더는 스레드 안전하지 않으므로(MPS) 호출부에서 락을 건다.
    """
    client = client_factory()
    n, fails = 0, 0
    interval = 1.0 / rate if rate > 0 else 0.0
    while not stop.is_set():
        i = n
        docid = f"{prefix}-{i:06d}"
        title = f"재색인 중 삽입 {i}"
        text = f"{title}. 이 문서는 재색인이 진행되는 동안 들어온 쓰기다. seq={i}"
        try:
            name, dim = target_of(client)
            vec = encode([f"{title}\n{text}"], dim)[0]
            client.insert(collection_name=name,
                          data=[{"docid": docid, "vector": vec, "title": title[:512]}])
            journal.append(Entry(docid, title, text, time.perf_counter() - t0))
            n += 1
        except Exception:
            fails += 1
        if interval:
            time.sleep(interval)
    return {"written": n, "failed": fails}


def catch_up(client, target: str, entries: list[Entry], encode, dim: int,
             batch: int = 128) -> dict:
    """저널을 대상 컬렉션으로 재생한다. **대상 모델로 다시 임베딩한다.**"""
    t0 = time.perf_counter()
    done = 0
    for i in range(0, len(entries), batch):
        chunk = entries[i:i + batch]
        vecs = encode([f"{e.title}\n{e.text}" for e in chunk], dim)
        client.insert(collection_name=target, data=[
            {"docid": e.docid, "vector": v, "title": e.title[:512]}
            for e, v in zip(chunk, vecs)
        ])
        done += len(chunk)
    client.flush(collection_name=target)
    return {"replayed": done, "seconds": round(time.perf_counter() - t0, 3)}


def verify(client, target: str, expected: set[str], batch: int = 256) -> dict:
    """대상 컬렉션에 저널의 docid 가 전부 있는지 확인한다.

    개수만 세면 안 된다 — 다른 문서가 들어와 수가 맞는 경우가 있다. **id 로 대조한다.**
    """
    missing: set[str] = set()
    ids = sorted(expected)
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        got = client.get(collection_name=target, ids=chunk, output_fields=["docid"])
        found = {r.get("docid") for r in got}
        missing |= set(chunk) - found
    return {"expected": len(expected), "missing": len(missing),
            "sample_missing": sorted(missing)[:5]}
