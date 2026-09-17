#!/usr/bin/env python3
"""임베딩 처리량 측정 — 재색인 소요의 병목을 찾는다 (이슈 #16).

D-028 은 "병목은 Milvus 적재가 아니라 MPS 임베딩"이라고 단정했다. **근거 없이 단정한
문장이다.** 여기서 둘을 분리해 실제로 확인하고, 조절 가능한 손잡이(배치 크기, 정밀도)가
얼마나 효과가 있는지 잰다.

사용: .venv/bin/python scripts/embed_bench.py [--n 2000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vecshift import dataset as ds
from vecshift.config import load
from vecshift.embed import passage_text
from vecshift.paths import setup_hf_cache


def load_texts(n: int) -> list[str]:
    cfg = load()
    path = ds.tier_path(cfg)
    if not path.exists():
        raise SystemExit(f"티어 샘플이 없습니다: {path} — 먼저 `make ingest`")
    out = []
    for doc in ds.iter_tier(path):
        out.append(passage_text(doc))
        if len(out) >= n:
            break
    return out


def bench_embed(model_id: str, texts: list[str], batch: int, fp16: bool,
                device: str) -> dict:
    setup_hf_cache()
    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_id, device=device)
    if fp16:
        m = m.half()
    # 예열 — 첫 배치는 커널 컴파일·메모리 할당이 섞인다
    m.encode(texts[: min(batch, len(texts))], batch_size=batch,
             normalize_embeddings=True, convert_to_numpy=True,
             show_progress_bar=False)
    if device == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    v = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    if device == "mps":
        torch.mps.synchronize()
    dt = time.perf_counter() - t0
    del m
    if device == "mps":
        torch.mps.empty_cache()
    return {"rows": len(texts), "seconds": round(dt, 2),
            "rows_per_s": round(len(texts) / dt, 1), "dim": int(v.shape[1])}


def token_lengths(model_id: str, texts: list[str]):
    """토큰 길이 분포. max_seq_length 를 줄이는 게 레버가 될지 판단하는 근거다."""
    setup_hf_cache()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    return np.array([len(tok("passage: " + t, truncation=False)["input_ids"]) for t in texts])


def fp16_fidelity(model_id: str, texts: list[str], k: int = 10, probes: int = 100) -> dict:
    """fp16 이 벡터를 얼마나 바꾸는가. 코사인만 보면 거의 1 이라 속는다 —
    검색은 순위로 동작하므로 **이웃 겹침**을 같이 본다."""
    v32 = _encode(model_id, texts, 64, False)
    v16 = _encode(model_id, texts, 64, True)
    cos = (v32 * v16).sum(axis=1)
    q = np.arange(min(probes, len(texts)))
    t32 = np.argsort(-(v32[q] @ v32.T), axis=1)[:, 1:k + 1]
    t16 = np.argsort(-(v16[q] @ v16.T), axis=1)[:, 1:k + 1]
    overlap = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(t32, t16)]))
    return {"cos_min": float(cos.min()), "overlap": overlap}


def _encode(model_id: str, texts: list[str], batch: int, fp16: bool):
    setup_hf_cache()
    import torch
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_id, device="mps")
    if fp16:
        m = m.half()
    v = m.encode(["passage: " + t for t in texts], batch_size=batch,
                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    del m
    torch.mps.empty_cache()
    return v.astype("float32")


def bench_insert(rows: int, dim: int, chunk: int = 1024) -> dict:
    """Milvus 적재만 따로 잰다. 임베딩과 비교해야 병목을 말할 수 있다."""
    import numpy as np
    from pymilvus import DataType

    from vecshift.client import connect

    cfg = load()
    c = connect(cfg)
    name = "_bench_insert"
    if c.has_collection(name):
        c.drop_collection(name)
    sch = c.create_schema(auto_id=False, enable_dynamic_field=False)
    sch.add_field("docid", DataType.VARCHAR, is_primary=True, max_length=64)
    sch.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    ip = c.prepare_index_params()
    ip.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE",
                 params={"M": 16, "efConstruction": 200})
    c.create_collection(collection_name=name, schema=sch, index_params=ip)

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(rows, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    t0 = time.perf_counter()
    for i in range(0, rows, chunk):
        part = vecs[i:i + chunk]
        c.insert(collection_name=name, data=[
            {"docid": f"b-{i+j}", "vector": part[j].tolist()}
            for j in range(len(part))])
    c.flush(collection_name=name)
    dt = time.perf_counter() - t0
    c.drop_collection(name)
    return {"rows": rows, "seconds": round(dt, 2),
            "rows_per_s": round(rows / dt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--variant", default="v2")
    a = ap.parse_args()

    cfg = load()
    v = cfg.variant(a.variant)
    model_id, dim = v["id"], int(v["dim"])
    texts = load_texts(a.n)
    print(f"\n임베딩 처리량 — {model_id} ({dim}d) · {len(texts):,} passage\n")

    print(f"  {'설정':<22} {'rows/s':>9} {'소요':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*8}")
    results = []
    # 위로만 재면 64가 최적처럼 보인다. 아래(16·32)까지 재야 64 이하가 평평한
    # 천장이라는 게 드러난다 — 배치는 레버가 아니다(D-032).
    for batch in (16, 32, 64, 128, 256, 512):
        for fp16 in (False, True):
            label = f"batch={batch} {'fp16' if fp16 else 'fp32'}"
            try:
                r = bench_embed(model_id, texts, batch, fp16, "mps")
                results.append((label, r))
                print(f"  {label:<22} {r['rows_per_s']:>9,.1f} {r['seconds']:>7.1f}s")
            except Exception as e:
                print(f"  {label:<22} {'FAIL':>9}  {str(e)[:40]}")

    print(f"\n  Milvus 적재만 (임베딩 제외, {dim}d)")
    ins = bench_insert(a.n, dim)
    print(f"  {'insert only':<22} {ins['rows_per_s']:>9,.1f} {ins['seconds']:>7.1f}s")

    lens = token_lengths(model_id, texts)
    print(f"\n  문단 토큰 길이 — 중앙 {int(np.median(lens))} · p90 {int(np.percentile(lens, 90))}"
          f" · 256 초과 {(lens > 256).mean()*100:.1f}% · 512 초과 {(lens > 512).mean()*100:.1f}%")

    fid = fp16_fidelity(model_id, texts)
    print(f"  fp16 충실도 — cos 최소 {fid['cos_min']:.6f} · top-10 이웃 겹침 {fid['overlap']*100:.1f}%"
          "  (100% 가 아니면 품질 게이트로 판정해야 한다)")

    if results:
        best_label, best = max(results, key=lambda x: x[1]["rows_per_s"])
        base = next((r for l, r in results if l == "batch=64 fp32"), None)
        print(f"\n  최고: {best_label} — {best['rows_per_s']:,.1f} rows/s")
        if base:
            print(f"  기준(batch=64 fp32) 대비 {best['rows_per_s']/base['rows_per_s']:.2f}배")
        # 적재는 청크마다 임베딩 뒤에 직렬로 붙으므로 행당 시간이 더해진다.
        # fp32 기준 이 모델의 예측은 88.7 rows/s, 실측 v2 적재는 84 rows/s 였다.
        e2e = 1 / (1 / best["rows_per_s"] + 1 / ins["rows_per_s"])
        need = 1_400_000 / 1800
        print(f"  전 구간 예측(임베딩+적재 직렬): {e2e:,.1f} rows/s → 1.4M {1_400_000/e2e/3600:.1f} 시간")
        print(f"  1.4M 30분 목표({need:.0f} rows/s)까지 {need/e2e:.1f}배 더 필요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
