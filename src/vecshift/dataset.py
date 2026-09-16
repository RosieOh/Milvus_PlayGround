"""MIRACL Korean 코퍼스·qrels 로딩과 티어 샘플링.

**HF 로더 스크립트를 쓰지 않는다.** `datasets` 5.x 가 스크립트 기반 데이터셋 지원을
제거해서 `load_dataset("miracl/miracl-corpus", "ko")` 가 더는 동작하지 않는다(D-013).
원본 파일이 JSONL.gz 3샤드와 TREC 형식 TSV 로 단순해서, 직접 읽는 쪽이 오히려
재현성이 높다 — 라이브러리 버전에 묶이지 않는다.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from .config import Config
from .paths import DATA, require_free, setup_hf_cache

CORPUS_REPO = "miracl/miracl-corpus"
QRELS_REPO = "miracl/miracl"
CORPUS_SHARDS = 3


def _hub_download(repo: str, filename: str) -> Path:
    setup_hf_cache()  # HF import 전에 캐시를 외장으로 — D-012
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename, repo_type="dataset"))


def corpus_shards(lang: str = "ko") -> list[Path]:
    """코퍼스 샤드를 받는다. 압축 약 226 MB."""
    require_free(DATA, 3.0, "코퍼스 다운로드")
    return [
        _hub_download(CORPUS_REPO, f"miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz")
        for i in range(CORPUS_SHARDS)
    ]


def iter_corpus(shards: list[Path]) -> Iterator[dict[str, str]]:
    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def load_qrels(split: str = "dev", lang: str = "ko") -> tuple[
    dict[str, dict[str, int]], dict[str, str]
]:
    """(판정, 질의) 를 돌려준다.

    판정은 `{query_id: {docid: relevance}}`. relevance 0 은 "판정했으나 정답 아님"
    이라는 뜻이므로 버리지 않는다 — 정답만 남기면 판정 규모를 알 수 없다.
    """
    qrels_p = _hub_download(
        QRELS_REPO, f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv"
    )
    topics_p = _hub_download(
        QRELS_REPO, f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv"
    )

    rel: dict[str, dict[str, int]] = defaultdict(dict)
    with open(qrels_p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            qid, _, docid, r = parts
            rel[qid][docid] = int(r)

    topics: dict[str, str] = {}
    with open(topics_p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            qid, text = line.rstrip("\n").split("\t", 1)
            topics[qid] = text

    return dict(rel), topics


def positive_docids(rel: dict[str, dict[str, int]]) -> set[str]:
    return {d for judged in rel.values() for d, r in judged.items() if r > 0}


def tier_path(cfg: Config) -> Path:
    tier = cfg.require("dataset.active_tier")
    return DATA / f"corpus-{tier}.jsonl"


def build_tier(cfg: Config, seed: int = 42, force: bool = False) -> tuple[Path, dict]:
    """티어 샘플을 만든다.

    **정답 문서를 무조건 포함한다.** 앞에서부터 N개를 자르면 골든셋의 정답이
    샘플 밖으로 나가고, recall 은 구조적으로 0 에 가까워진다. 그렇게 나온 숫자는
    인덱스도 모델도 아닌 샘플링을 재고 있는 것이다.

    나머지는 고정 시드로 무작위 추출한다 — 앞쪽 문서만 쓰면 위키 문서 순서(=주제
    편향)가 그대로 들어온다.
    """
    out = tier_path(cfg)
    size = cfg.active_tier_size  # 0 = 전량
    rel, _ = load_qrels("dev")
    keep = positive_docids(rel)

    if out.exists() and not force:
        n = sum(1 for _ in open(out, encoding="utf-8"))
        return out, {"rows": n, "positives": len(keep), "reused": True}

    shards = corpus_shards()
    rng = random.Random(seed)

    # 한 번 훑으면서 정답은 전부 담고, 나머지는 저수지 표본추출로 뽑는다.
    # 전량(1.4 M)을 메모리에 올리지 않기 위해서다.
    fill = max(0, size - len(keep)) if size else 0
    reservoir: list[dict] = []
    seen_other = 0
    kept: list[dict] = []

    for doc in iter_corpus(shards):
        if doc["docid"] in keep:
            kept.append(doc)
            continue
        if not size:  # 전량
            reservoir.append(doc)
            continue
        seen_other += 1
        if len(reservoir) < fill:
            reservoir.append(doc)
        else:
            j = rng.randrange(seen_other)
            if j < fill:
                reservoir[j] = doc

    rows = kept + reservoir
    rng.shuffle(rows)

    DATA.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(
                {"docid": r["docid"], "title": r.get("title", ""), "text": r["text"]},
                ensure_ascii=False,
            ) + "\n")

    return out, {
        "rows": len(rows),
        "positives": len(kept),
        "missing_positives": len(keep) - len(kept),
        "reused": False,
    }


def iter_tier(path: Path) -> Iterator[dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
