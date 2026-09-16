"""vecshift CLI. W1 범위: doctor / smoke / ingest / goldenset / eval."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import typer

from . import __version__
from .config import ROOT, load

app = typer.Typer(
    add_completion=False,
    help="Milvus 무중단 임베딩 재색인 & 인덱스 튜닝 하네스",
)

OK = "\033[32mOK  \033[0m"
BAD = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def _line(status: str, label: str, detail: str = "") -> None:
    typer.echo(f"  [{status}] {label:<26} {detail}")


@app.command()
def doctor(config: str = typer.Option(None, "--config", "-c")) -> None:
    """환경 점검. 여기서 초록불이 아니면 그 다음 단계는 의미가 없다."""
    typer.echo(f"\nvecshift {__version__} — 환경 점검\n")
    failures = 0

    # 1. Python
    v = sys.version_info
    pyok = (v.major, v.minor) == (3, 12)
    _line(OK if pyok else WARN, "python", f"{v.major}.{v.minor}.{v.micro}"
          + ("" if pyok else "  (3.12 권장 — D-002)"))

    # 2. pymilvus
    try:
        import pymilvus

        _line(OK, "pymilvus", pymilvus.__version__)
    except Exception as e:  # pragma: no cover
        _line(BAD, "pymilvus", str(e))
        failures += 1

    # 3. 설정
    try:
        cfg = load(config)
        tier = cfg.require("dataset.active_tier")
        _line(OK, "config", f"{cfg.path.relative_to(ROOT)}  tier={tier}"
              f" ({cfg.active_tier_size or 'full'})")
    except Exception as e:
        _line(BAD, "config", str(e))
        typer.echo("")
        raise typer.Exit(1)

    # 4. docker
    if shutil.which("docker"):
        import subprocess

        r = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            _line(OK, "docker daemon", r.stdout.strip())
        else:
            _line(BAD, "docker daemon", "미실행 — Docker Desktop 을 켜세요")
            failures += 1
    else:
        _line(BAD, "docker", "설치되지 않음")
        failures += 1

    # 5. Milvus
    try:
        from .client import connect, server_version

        t0 = time.perf_counter()
        client = connect(cfg)
        cols = client.list_collections()
        ms = (time.perf_counter() - t0) * 1000
        _line(OK, "milvus", f"{server_version(client)}  "
                            f"collections={len(cols)}  {ms:.0f}ms")
    except Exception as e:
        _line(BAD, "milvus", f"{type(e).__name__}: {str(e)[:70]}")
        failures += 1

    # 6. 디스크 — 이 프로젝트에서는 1급 제약이다 (D-004)
    du = shutil.disk_usage(ROOT)
    free_gb = du.free / 1e9
    _line(OK if free_gb >= 20 else WARN, "disk free",
          f"{free_gb:.1f} GB" + ("" if free_gb >= 20 else "  (티어 축소 필요)"))

    typer.echo("")
    if failures:
        typer.secho(f"{failures}개 항목 실패.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("환경 정상.", fg=typer.colors.GREEN)


@app.command()
def smoke(config: str = typer.Option(None, "--config", "-c")) -> None:
    """생성 → 삽입 → 인덱스 → 검색 → 삭제 왕복 검증."""
    import numpy as np

    from .client import connect

    cfg = load(config)
    client = connect(cfg)
    name = "_vecshift_smoke"
    dim, n = 32, 1000

    typer.echo(f"\n왕복 검증  collection={name}  dim={dim}  n={n}\n")
    try:
        if client.has_collection(name):
            client.drop_collection(name)

        t0 = time.perf_counter()
        client.create_collection(collection_name=name, dimension=dim,
                                 metric_type="COSINE", auto_id=False)
        _line(OK, "create_collection", f"{(time.perf_counter()-t0)*1000:.0f}ms")

        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(n, dim)).astype("float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        rows = [{"id": i, "vector": vecs[i].tolist(), "tag": f"t{i % 5}"}
                for i in range(n)]

        t0 = time.perf_counter()
        client.insert(collection_name=name, data=rows)
        client.flush(collection_name=name)
        _line(OK, "insert + flush", f"{n} rows  {(time.perf_counter()-t0)*1000:.0f}ms")

        t0 = time.perf_counter()
        res = client.search(collection_name=name, data=[vecs[0].tolist()],
                            limit=5, output_fields=["tag"])
        ms = (time.perf_counter() - t0) * 1000
        hits = res[0]
        top = hits[0]
        self_hit = top["id"] == 0
        _line(OK if self_hit else BAD, "search",
              f"top1=id:{top['id']} dist={top['distance']:.4f}  {ms:.1f}ms")

        t0 = time.perf_counter()
        fres = client.search(collection_name=name, data=[vecs[0].tolist()],
                             limit=5, filter="tag == 't1'", output_fields=["tag"])
        tags = {h["entity"]["tag"] for h in fres[0]}
        okf = tags == {"t1"}
        _line(OK if okf else BAD, "filtered search",
              f"tags={sorted(tags)}  {(time.perf_counter()-t0)*1000:.1f}ms")

        client.drop_collection(name)
        _line(OK, "drop_collection", "")

        typer.echo("")
        if self_hit and okf:
            typer.secho("왕복 검증 통과.", fg=typer.colors.GREEN)
        else:
            typer.secho("왕복 검증 실패 — 결과가 기대와 다릅니다.", fg=typer.colors.RED)
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo("")
        typer.secho(f"{type(e).__name__}: {e}", fg=typer.colors.RED)
        raise typer.Exit(1)


@app.command()
def ingest(
    config: str = typer.Option(None, "--config", "-c"),
    variant: str = typer.Option("v1", "--variant", help="embedding.v1 / v2"),
    rebuild: bool = typer.Option(False, "--rebuild", help="티어 샘플과 컬렉션을 다시 만든다"),
) -> None:
    """MIRACL 코퍼스를 티어만큼 샘플링해 임베딩하고 Milvus 에 적재한다."""
    import json

    from . import collection as coll
    from . import dataset as ds
    from .client import connect
    from .embed import Encoder, batched, passage_text
    from .paths import RESULTS, ensure_dirs

    cfg = load(config)
    ensure_dirs()
    tier = cfg.require("dataset.active_tier")
    v = cfg.variant(variant)

    typer.echo(f"\n적재  tier={tier}({cfg.active_tier_size or 'full'})  "
               f"variant={variant}  model={v['id']}  dim={v['dim']}\n")

    t0 = time.perf_counter()
    path, info = ds.build_tier(cfg, force=rebuild)
    _line(OK, "티어 샘플",
          f"{info['rows']:,} rows  정답포함 {info['positives']}개"
          + ("  (재사용)" if info["reused"] else f"  {time.perf_counter()-t0:.0f}s"))
    if info.get("missing_positives"):
        _line(WARN, "정답 누락", f"{info['missing_positives']}개 — recall 상한이 낮아진다")

    enc = Encoder(v, batch_size=int(cfg.get("embedding.batch_size", 64)),
                  normalize=bool(cfg.get("embedding.normalize", True)))
    _line(OK, "인코더", f"{enc.id}  device={enc.device}  batch={enc.batch_size}")

    client = connect(cfg)
    name = coll.name_for(cfg, variant)
    coll.create(client, name, enc.dim, enc.metric, drop_existing=rebuild)
    _line(OK, "컬렉션", f"{name}  metric={enc.metric}")

    total, t0 = 0, time.perf_counter()
    with typer.progressbar(length=info["rows"], label="  임베딩+적재") as bar:
        for chunk in batched(ds.iter_tier(path), 1024):
            vecs = enc.passages([passage_text(d) for d in chunk])
            total += coll.insert(client, name, [d["docid"] for d in chunk],
                                 [d.get("title", "") for d in chunk], vecs)
            bar.update(len(chunk))
    client.flush(collection_name=name)
    dt = time.perf_counter() - t0
    _line(OK, "적재 완료", f"{total:,} rows  {dt:.0f}s  ({total/dt:.0f} rows/s)")

    (RESULTS / f"ingest-{tier}-{variant}.json").write_text(json.dumps(
        {"tier": tier, "variant": variant, "model": enc.id, "dim": enc.dim,
         "rows": total, "seconds": round(dt, 1), "device": enc.device,
         "collection": name}, ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo("")
    typer.secho(f"적재 완료 — {name}", fg=typer.colors.GREEN)


@app.command()
def goldenset(
    config: str = typer.Option(None, "--config", "-c"),
    split: str = typer.Option("dev", "--split", help="dev / train"),
) -> None:
    """qrels 에서 골든셋을 만든다. dev 는 213 질의로 고정 — 이게 전부다."""
    import json

    from . import dataset as ds
    from .paths import DATA, ensure_dirs

    cfg = load(config)
    ensure_dirs()
    rel, topics = ds.load_qrels(split)
    pos = ds.positive_docids(rel)

    want = int(cfg.get("goldenset.size", 0) or 0)
    have = len(topics)
    out = DATA / f"goldenset-{split}.json"
    out.write_text(json.dumps(
        {"split": split, "queries": topics, "qrels": rel}, ensure_ascii=False), encoding="utf-8")

    typer.echo(f"\n골든셋  split={split}\n")
    _line(OK, "질의", f"{have}개")
    _line(OK, "정답 문서", f"{pos and len(pos)}개 (고유)")
    _line(OK, "질의당 평균 정답",
          f"{sum(1 for j in rel.values() for r in j.values() if r > 0)/max(len(rel),1):.2f}개")
    if want and want > have:
        _line(WARN, "설정 goldenset.size",
              f"{want} 요청 / {have} 가능 — MIRACL {split} 은 이게 전부다")
    _line(OK, "저장", str(out.relative_to(ROOT)))
    typer.echo("")


@app.command(name="eval")
def eval_(
    config: str = typer.Option(None, "--config", "-c"),
    variant: str = typer.Option("v1", "--variant"),
    split: str = typer.Option("dev", "--split"),
    k: int = typer.Option(0, "--k", help="0 이면 설정의 goldenset.top_k"),
) -> None:
    """골든셋으로 nDCG@k · Recall@k 를 계산한다 (qrels 기준 — D-007)."""
    import json

    from . import collection as coll
    from . import dataset as ds
    from .client import connect
    from .embed import Encoder
    from .evaluate import aggregate
    from .paths import RESULTS, ensure_dirs

    cfg = load(config)
    ensure_dirs()
    k = k or int(cfg.get("goldenset.top_k", 10))
    v = cfg.variant(variant)
    rel, topics = ds.load_qrels(split)

    enc = Encoder(v, batch_size=int(cfg.get("embedding.batch_size", 64)),
                  normalize=bool(cfg.get("embedding.normalize", True)))
    client = connect(cfg)
    name = coll.name_for(cfg, variant)
    if not client.has_collection(name):
        typer.secho(f"컬렉션이 없습니다: {name} — 먼저 `make ingest`", fg=typer.colors.RED)
        raise typer.Exit(1)
    client.load_collection(collection_name=name)

    typer.echo(f"\n평가  collection={name}  split={split}  k={k}  질의 {len(topics)}개\n")

    qids = list(topics)
    qvecs = enc.queries([topics[q] for q in qids])
    t0 = time.perf_counter()
    res = client.search(collection_name=name, data=[x.tolist() for x in qvecs],
                        limit=k, output_fields=["docid"])
    ms = (time.perf_counter() - t0) * 1000

    runs = {qid: [h["id"] if "id" in h else h["entity"]["docid"] for h in hits]
            for qid, hits in zip(qids, res)}
    m = aggregate(runs, rel, k)

    _line(OK, "검색", f"{len(qids)} 질의  {ms:.0f}ms  ({ms/len(qids):.1f}ms/질의)")
    _line(OK, f"nDCG@{k}", f"{m[f'ndcg@{k}']:.4f}  ± {m[f'ndcg@{k}_stderr']:.4f}")
    _line(OK, f"Recall@{k}", f"{m[f'recall@{k}']:.4f}  ± {m[f'recall@{k}_stderr']:.4f}")

    tier = cfg.require("dataset.active_tier")
    out = RESULTS / f"eval-{tier}-{variant}-{split}.json"
    out.write_text(json.dumps(
        {"tier": tier, "variant": variant, "model": enc.id, "split": split, "k": k,
         "collection": name, "search_ms_total": round(ms, 1), **m},
        ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo("")
    typer.secho(f"저장 — {out.relative_to(ROOT)}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
