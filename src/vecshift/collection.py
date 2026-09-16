"""Milvus 컬렉션 스키마와 적재.

docid 를 그대로 기본키로 쓴다(VARCHAR). 정수로 바꿔 매핑을 따로 들고 다니면
평가 단계에서 qrels 와 맞추는 코드가 한 겹 늘고, 그 겹에서 사고가 난다.
본문은 저장하지 않는다 — 평가에 필요한 것은 docid 뿐이고, 재색인 때는 어차피
코퍼스에서 다시 읽는다.
"""

from __future__ import annotations

from pymilvus import DataType, MilvusClient

from .config import Config

DOCID_MAX = 64
TITLE_MAX = 512


def name_for(cfg: Config, variant: str) -> str:
    return f"{cfg.require('collection.prefix')}_{variant}"


def create(client: MilvusClient, name: str, dim: int, metric: str,
           drop_existing: bool = False) -> None:
    if client.has_collection(name):
        if not drop_existing:
            return
        client.drop_collection(name)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
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
