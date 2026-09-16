"""Milvus 컬렉션 스키마와 적재.

docid 를 그대로 기본키로 쓴다(VARCHAR). 정수로 바꿔 매핑을 따로 들고 다니면
평가 단계에서 qrels 와 맞추는 코드가 한 겹 늘고, 그 겹에서 사고가 난다.
본문은 저장하지 않는다 — 평가에 필요한 것은 docid 뿐이고, 재색인 때는 어차피
코퍼스에서 다시 읽는다.
"""

from __future__ import annotations

import json

from pymilvus import DataType, MilvusClient

from .config import Config

DOCID_MAX = 64
TITLE_MAX = 512


def name_for(cfg: Config, variant: str) -> str:
    return f"{cfg.require('collection.prefix')}_{variant}"


def create(client: MilvusClient, name: str, dim: int, metric: str,
           drop_existing: bool = False, model: str = "") -> None:
    """컬렉션을 만든다.

    `model` 을 받으면 **컬렉션 설명에 심는다.** alias 는 컬렉션을 가리킬 뿐 모델을
    가리키지 않는다 — 그래서 클라이언트가 "지금 이 컬렉션은 어떤 모델로 만들어졌는가"를
    서버에서 읽을 수 있어야 한다(D-025). 별도 레지스트리를 두면 그게 또 하나의
    동기화 대상이 된다.
    """
    if client.has_collection(name):
        if not drop_existing:
            return
        client.drop_collection(name)

    desc = json.dumps({"model": model, "dim": dim, "metric": metric},
                      ensure_ascii=False) if model else ""
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False,
                                  description=desc)
    schema.add_field("docid", DataType.VARCHAR, is_primary=True, max_length=DOCID_MAX)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("title", DataType.VARCHAR, max_length=TITLE_MAX)

    index = client.prepare_index_params()
    # W1 은 베이스라인이다. 인덱스 파라미터 탐색은 M1(W2)의 몫이므로 여기서는
    # 기본 HNSW 하나만 건다. 여기서 튜닝하면 W2 의 비교 기준이 흐려진다.
    index.add_index(field_name="vector", index_type="HNSW", metric_type=metric,
                    params={"M": 16, "efConstruction": 200})

    client.create_collection(collection_name=name, schema=schema,
                             index_params=index)


def insert(client: MilvusClient, name: str, docids: list[str], titles: list[str],
           vectors) -> int:
    rows = [
        {"docid": d, "vector": v.tolist(), "title": t[:TITLE_MAX]}
        for d, t, v in zip(docids, titles, vectors)
    ]
    client.insert(collection_name=name, data=rows)
    return len(rows)


def meta_of(client: MilvusClient, name: str) -> dict:
    """컬렉션이 어떤 모델로 만들어졌는지 읽는다.

    설명에 심어둔 값을 먼저 보고, 없으면 벡터 필드의 차원으로라도 답한다 —
    예전에 만든 컬렉션이나 남이 만든 컬렉션에도 최소한의 판단 근거는 있어야 한다.
    """
    d = client.describe_collection(collection_name=name)
    out: dict = {}
    raw = (d.get("description") or "").strip()
    if raw.startswith("{"):
        try:
            out = dict(json.loads(raw))
        except ValueError:
            out = {}
    if "dim" not in out:
        for f in d.get("fields") or []:
            dim = (f.get("params") or {}).get("dim")
            if dim:
                out["dim"] = int(dim)
                break
    out["collection"] = name
    return out


def resolve_alias(client: MilvusClient, alias: str) -> dict:
    """alias → 컬렉션 → 모델. 서비스 질의 경로가 매번 물어야 하는 것."""
    target = client.describe_alias(alias=alias).get("collection_name")
    if not target:
        raise RuntimeError(f"alias 가 아무것도 가리키지 않습니다: {alias}")
    return meta_of(client, target)
