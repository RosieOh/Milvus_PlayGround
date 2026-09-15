#!/usr/bin/env python3
"""iobench.sh 가 남긴 fio JSON 을 마크다운 표로 요약한다.

절대 수치보다 **두 대상의 상대 비교**가 이 저장소의 서술 형식이다(README 「측정의 한계」 2).
그래서 두 대상이 모두 있으면 마지막에 배수를 낸다. 가드레일이 내장 대상을 건너뛴
경우에는 있는 것만으로 표를 그린다 — 비교가 없다고 측정까지 버리지는 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TARGETS = ["bind-external", "volume-internal"]
LABELS = {"bind-external": "외장 bind", "volume-internal": "내장 volume"}
PATTERNS = ["randread-4k", "randread-16k", "randwrite-4k"]


def stats(path: Path) -> dict[str, float] | None:
    """(IOPS, 대역폭 MiB/s, p99 지연 ms). 읽을 수 없거나 실패한 런은 None."""
    try:
        job = json.loads(path.read_text())["jobs"][0]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    # fio 는 실패해도 JSON 을 뱉는다. error 를 보지 않으면 0 을 실측치로 착각한다.
    if job.get("error"):
        print(f"  ! {path.name}: fio error={job['error']}", file=sys.stderr)
        return None
    side = job["read"] if job["read"]["io_bytes"] else job["write"]
    return {
        "iops": side["iops"],
        "bw": side["bw"] / 1024,
        "p99": side["clat_ns"]["percentile"].get("99.000000", 0) / 1e6,
    }


def main(out_dir: str) -> int:
    d = Path(out_dir)
    data: dict[str, dict[str, dict[str, float]]] = {}
    for pat in PATTERNS:
        for t in TARGETS:
            s = stats(d / f"{t}-{pat}.json")
            if s:
                data.setdefault(pat, {})[t] = s

    if not data:
        print("측정 결과가 없다.", file=sys.stderr)
        return 1

    present = [t for t in TARGETS if any(t in v for v in data.values())]
    if len(present) == 1:
        print(f"\n  ({LABELS[present[0]]} 단독 — 비교 대상이 없어 배수는 내지 않는다)")

    head = "| 패턴 | " + " | ".join(f"{LABELS[t]} IOPS" for t in present) + " |"
    head += " 외장/내장 |" if len(present) == 2 else ""
    head += " " + " | ".join(f"{LABELS[t]} p99" for t in present) + " |"
    print()
    print(head)
    print("|---|" + "---:|" * (len(present) * 2 + (1 if len(present) == 2 else 0)))
    for pat in PATTERNS:
        row = data.get(pat)
        if not row or not all(t in row for t in present):
            continue
        cells = [f"{row[t]['iops']:,.0f}" for t in present]
        if len(present) == 2:
            e, i = row["bind-external"]["iops"], row["volume-internal"]["iops"]
            cells.append(f"{e / i:.2f}×" if i else "—")
        cells += [f"{row[t]['p99']:.2f} ms" for t in present]
        print(f"| {pat} | " + " | ".join(cells) + " |")

    print()
    print("| 패턴 | " + " | ".join(f"{LABELS[t]} MiB/s" for t in present) + " |")
    print("|---|" + "---:|" * len(present))
    for pat in PATTERNS:
        row = data.get(pat)
        if not row or not all(t in row for t in present):
            continue
        print(f"| {pat} | " + " | ".join(f"{row[t]['bw']:,.1f}" for t in present) + " |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/iobench"))
