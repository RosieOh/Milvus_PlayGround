"""vecshift CLI. doctor / smoke / ingest / goldenset / eval / sweep / load / shift."""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import typer

from . import __version__
from .config import ROOT, load as load_cfg

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
        cfg = load_cfg(config)
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

    cfg = load_cfg(config)
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

    cfg = load_cfg(config)
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

    cfg = load_cfg(config)
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
    no_query_prefix: bool = typer.Option(
        False, "--no-query-prefix",
        help="질의 접두사를 빼고 평가 (D-005 의 접두사 규약 어블레이션)"),
) -> None:
    """골든셋으로 nDCG@k · Recall@k 를 계산한다 (qrels 기준 — D-007)."""
    import json

    from . import collection as coll
    from . import dataset as ds
    from .client import connect
    from .embed import Encoder
    from .evaluate import aggregate, per_query
    from .paths import RESULTS, ensure_dirs

    cfg = load_cfg(config)
    ensure_dirs()
    k = k or int(cfg.get("goldenset.top_k", 10))
    v = cfg.variant(variant)
    rel, topics = ds.load_qrels(split)

    enc = Encoder(v, batch_size=int(cfg.get("embedding.batch_size", 64)),
                  normalize=bool(cfg.get("embedding.normalize", True)))
    if no_query_prefix:
        enc.query_prefix = ""  # 문서는 접두사가 붙은 채로 색인돼 있다 — 불일치를 만든다
    client = connect(cfg)
    name = coll.name_for(cfg, variant)
    if not client.has_collection(name):
        typer.secho(f"컬렉션이 없습니다: {name} — 먼저 `make ingest`", fg=typer.colors.RED)
        raise typer.Exit(1)
    client.load_collection(collection_name=name)

    tag = "  [접두사 없음 — 어블레이션]" if no_query_prefix else ""
    typer.echo(f"\n평가  collection={name}  split={split}  k={k}  질의 {len(topics)}개{tag}\n")

    qids = list(topics)
    qvecs = enc.queries([topics[q] for q in qids])
    t0 = time.perf_counter()
    res = client.search(collection_name=name, data=[x.tolist() for x in qvecs],
                        limit=k, output_fields=["docid"])
    ms = (time.perf_counter() - t0) * 1000

    # pymilvus 는 기본키 필드명을 **그대로 최상위 키**로 돌려준다. PK 가 docid 이므로
    # h["docid"] 다 — `h["id"]` 가 아니다. 구버전/다른 스키마 대비로 entity 도 본다.
    def hit_docid(h: dict) -> str:
        return h.get("docid") or h.get("id") or h["entity"]["docid"]

    runs = {qid: [hit_docid(h) for h in hits] for qid, hits in zip(qids, res)}
    m = aggregate(runs, rel, k)

    _line(OK, "검색", f"{len(qids)} 질의  {ms:.0f}ms  ({ms/len(qids):.1f}ms/질의)")
    _line(OK, f"nDCG@{k}", f"{m[f'ndcg@{k}']:.4f}  ± {m[f'ndcg@{k}_stderr']:.4f}")
    _line(OK, f"Recall@{k}", f"{m[f'recall@{k}']:.4f}  ± {m[f'recall@{k}_stderr']:.4f}")

    tier = cfg.require("dataset.active_tier")
    suffix = "-noprefix" if no_query_prefix else ""
    out = RESULTS / f"eval-{tier}-{variant}-{split}{suffix}.json"
    out.write_text(json.dumps(
        {"tier": tier, "variant": variant, "model": enc.id, "split": split, "k": k,
         "collection": name, "query_prefix": enc.query_prefix,
         "search_ms_total": round(ms, 1), **m,
         # 질의별 점수를 남긴다 — 두 설정의 대응비교에 필요하다.
         "per_query": per_query(runs, rel, k)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    typer.echo("")
    typer.secho(f"저장 — {out.relative_to(ROOT)}", fg=typer.colors.GREEN)


@app.command()
def sweep(
    config: str = typer.Option(None, "--config", "-c"),
    variant: str = typer.Option("v1", "--variant"),
    split: str = typer.Option("dev", "--split"),
    k: int = typer.Option(0, "--k", help="0 이면 설정의 goldenset.top_k"),
    only: str = typer.Option(None, "--only", help="빌드 이름 부분일치로 걸러 실행"),
) -> None:
    """M1 — 인덱스 파라미터를 훑고 파레토 곡선을 낸다 (ANN recall — D-007)."""
    import json

    from . import collection as coll
    from . import dataset as ds
    from . import sweep as sw
    from .client import connect
    from .embed import Encoder
    from .paths import ROOT as _ROOT, RESULTS, ensure_dirs

    cfg = load_cfg(config)
    ensure_dirs()
    k = k or int(cfg.get("goldenset.top_k", 10))
    v = cfg.variant(variant)
    metric = v.get("metric", "COSINE")
    builds = cfg.require("sweep.builds")
    if only:
        builds = [b for b in builds if only in b["name"]]
    repeats = int(cfg.get("sweep.repeats", 3))
    warmup = int(cfg.get("sweep.warmup", 1))

    _, topics = ds.load_qrels(split)
    enc = Encoder(v, batch_size=int(cfg.get("embedding.batch_size", 64)),
                  normalize=bool(cfg.get("embedding.normalize", True)))
    qvecs = [x.tolist() for x in enc.queries([topics[q] for q in topics])]

    client = connect(cfg)
    name = coll.name_for(cfg, variant)
    if not client.has_collection(name):
        typer.secho(f"컬렉션이 없습니다: {name} — 먼저 `make ingest`", fg=typer.colors.RED)
        raise typer.Exit(1)

    total = sum(len(b.get("search") or [{}]) for b in builds)
    typer.echo(f"\n스윕  collection={name}  질의 {len(qvecs)}개  k={k}  "
               f"빌드 {len(builds)}종 → 설정 {total}개\n")

    t0 = time.perf_counter()
    gt_type = cfg.get("sweep.ground_truth", "FLAT")
    truth = sw.ground_truth(client, name, qvecs, k, metric, gt_type)
    _line(OK, f"정답({gt_type})", f"{len(truth)} 질의  {time.perf_counter()-t0:.0f}s")

    rows: list[dict] = []
    rkey = f"ann_recall@{k}"
    for b in builds:
        def show(r: dict) -> None:
            _line(OK, r["label"],
                  f"recall {r[rkey]:.4f}  {r['qps']:>8,.0f} QPS  "
                  f"{r['ms_per_query']:.3f} ms/q  build {r['build_seconds']:.0f}s")
        rows += sw.run_build(client, name, b, qvecs, k, metric, truth,
                             repeats, warmup, on_result=show)

    front = sw.pareto(rows, rkey, "qps")
    typer.echo("")
    _line(OK, "파레토 프론티어", f"{len(front)} / {len(rows)} 설정")
    for f in front:
        _line(OK, f"  {f['label']}", f"recall {f[rkey]:.4f}  {f['qps']:,.0f} QPS")

    tier = cfg.require("dataset.active_tier")
    out = RESULTS / f"sweep-{tier}-{variant}.json"
    out.write_text(json.dumps(
        {"tier": tier, "variant": variant, "model": enc.id, "k": k,
         "metric": metric, "queries": len(qvecs), "ground_truth": gt_type,
         "repeats": repeats, "rows": rows,
         "pareto": [f["label"] for f in front]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    _line(OK, "결과", str(out.relative_to(_ROOT)))

    try:
        from .plot import pareto_chart, use_korean_font

        use_korean_font()
        png = pareto_chart(
            rows, front, _ROOT / "reports" / f"pareto-{tier}-{variant}.png", rkey,
            f"M1 인덱스 파레토 — {tier} 티어 / {enc.id}",
            f"질의 {len(qvecs)}개 · k={k} · {metric} · DISKANN 은 저장 계층 때문에 불리하다(D-010)")
        _line(OK, "그래프", str(png.relative_to(_ROOT)))
    except ImportError:
        _line(WARN, "그래프", "matplotlib 없음 — `uv pip install -e '.[viz]'`")

    typer.echo("")
    typer.secho("스윕 완료.", fg=typer.colors.GREEN)


def _parse_params(items: list[str] | None) -> dict:
    """`ef=64` 같은 문자열을 검색 파라미터 dict 로. 숫자는 숫자로 바꾼다 —
    문자열로 넘기면 Milvus 가 조용히 무시하거나 기본값을 쓴다."""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise typer.BadParameter(f"key=value 형식이어야 합니다: {it}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = int(v)
        except ValueError:
            try:
                out[k.strip()] = float(v)
            except ValueError:
                out[k.strip()] = v.strip()
    return out


def _query_vectors(cfg, variant: str, split: str):
    """골든셋 질의를 해당 variant 로 임베딩한다."""
    from . import dataset as ds
    from .embed import Encoder

    _, topics = ds.load_qrels(split)
    enc = Encoder(cfg.variant(variant),
                  batch_size=int(cfg.get("embedding.batch_size", 64)),
                  normalize=bool(cfg.get("embedding.normalize", True)))
    return enc, [x.tolist() for x in enc.queries([topics[q] for q in topics])]


@app.command()
def load(
    config: str = typer.Option(None, "--config", "-c"),
    variant: str = typer.Option("v1", "--variant"),
    split: str = typer.Option("dev", "--split"),
    target: str = typer.Option(None, "--target", help="컬렉션 또는 alias. 기본은 설정의 alias"),
    seconds: float = typer.Option(20.0, "--seconds"),
    workers: int = typer.Option(8, "--workers"),
    gap_ms: float = typer.Option(500.0, "--gap-ms", help="이 이상 응답이 끊기면 정지로 본다"),
    param: list[str] = typer.Option(None, "--param", "-p",
                                    help="검색 파라미터 key=value (예: -p ef=64). 반복 가능"),
) -> None:
    """동시 클라이언트로 부하를 걸고 지연·오류·정지 구간을 잰다 (진짜 QPS)."""
    import json

    from . import load as ld
    from .client import connect
    from .paths import ROOT as _ROOT, RESULTS, ensure_dirs

    cfg = load_cfg(config)
    ensure_dirs()
    k = int(cfg.get("goldenset.top_k", 10))
    tgt = target or cfg.require("collection.alias")
    sp = _parse_params(param)
    enc, qvecs = _query_vectors(cfg, variant, split)

    client = connect(cfg)
    typer.echo(f"\n부하  target={tgt}  질의 {len(qvecs)}개(Zipf)  "
               f"워커 {workers}  {seconds:.0f}s"
               + (f"  params={sp}" if sp else "  params=인덱스 기본값") + "\n")
    res = ld.run(lambda: connect(cfg), tgt, qvecs, k, seconds, workers, sp)

    pcs = res.percentiles()
    gaps = res.gaps(gap_ms)
    _line(OK, "요청", f"{len(res.samples):,}건  성공 {len(res.oks()):,}")
    _line(OK if res.error_rate() == 0 else BAD, "오류율", f"{res.error_rate()*100:.4f}%")
    _line(OK, "QPS", f"{res.qps():,.0f}  (동시 {workers} · 단건 검색)")
    _line(OK, "지연", f"p50 {pcs['p50']:.2f}ms  p95 {pcs['p95']:.2f}ms  p99 {pcs['p99']:.2f}ms")
    _line(OK if not gaps else BAD, f"정지 구간(>{gap_ms:.0f}ms)",
          f"{len(gaps)}개" + (f"  최대 {max((b-a)*1000 for a,b in gaps):.0f}ms" if gaps else ""))

    out = RESULTS / f"load-{tgt}-{int(time.time())}.json"
    out.write_text(json.dumps({
        "target": tgt, "workers": workers, "seconds": res.duration,
        "search_params": sp,
        "requests": len(res.samples), "ok": len(res.oks()),
        "error_rate": res.error_rate(), "qps": res.qps(), **pcs,
        "gaps": gaps, "windows": res.window_stats(1.0),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _line(OK, "결과", str(out.relative_to(_ROOT)))
    typer.echo("")


@app.command()
def shift(
    config: str = typer.Option(None, "--config", "-c"),
    of: str = typer.Option("v1", "--from", help="현행 variant"),
    to: str = typer.Option("v2", "--to", help="교체 대상 variant"),
    split: str = typer.Option("dev", "--split"),
    threshold: float = typer.Option(0.01, "--threshold", help="허용 가능한 품질 하락폭"),
    under_load: float = typer.Option(0.0, "--under-load",
                                     help="이 초만큼 부하를 걸고 그 중간에 스왑한다"),
    workers: int = typer.Option(8, "--workers"),
    rollback: bool = typer.Option(False, "--rollback", help="스왑 후 즉시 되돌려 소요를 잰다"),
    force: bool = typer.Option(False, "--force", help="게이트를 무시하고 스왑"),
) -> None:
    """v1 → v2 alias 스왑. 품질 게이트를 통과해야 넘어간다."""
    import json
    import threading

    from . import collection as coll
    from . import load as ld
    from . import shift as sh
    from .client import connect
    from .paths import ROOT as _ROOT, RESULTS, ensure_dirs

    cfg = load_cfg(config)
    ensure_dirs()
    alias = cfg.require("collection.alias")
    src, dst = coll.name_for(cfg, of), coll.name_for(cfg, to)
    client = connect(cfg)

    typer.echo(f"\n스왑  alias={alias}  {of}({src}) → {to}({dst})\n")
    for name in (src, dst):
        if not client.has_collection(name):
            typer.secho(f"컬렉션이 없습니다: {name} — `vecshift ingest --variant "
                        f"{of if name == src else to}`", fg=typer.colors.RED)
            raise typer.Exit(1)

    # alias 가 아직 없으면 현행을 가리키게 만든다.
    if sh.current_target(client, alias) is None:
        sh.point_alias(client, alias, src)
    _line(OK, "현재 alias", f"{alias} → {sh.current_target(client, alias)}")

    # ── 품질 게이트 — 스왑 **전에** 판정한다 ────────────────────────────────
    base = _eval_scores(cfg, of, split)
    cand = _eval_scores(cfg, to, split)
    gate = sh.quality_gate(base["ndcg"], cand["ndcg"], threshold)
    _line(OK, f"게이트 기준({of})", f"nDCG@10 {base['ndcg']:.4f}")
    _line(OK, f"게이트 후보({to})", f"nDCG@10 {cand['ndcg']:.4f}")
    _line(OK if gate.passed else BAD, "품질 게이트", gate.reason())
    if not gate.passed and not force:
        typer.echo("")
        typer.secho("스왑을 중단했습니다. --force 로 무시할 수 있습니다.", fg=typer.colors.RED)
        raise typer.Exit(2)

    # ── 부하를 걸고 그 중간에 스왑한다 ──────────────────────────────────────
    res = swap_s = rb_s = None
    if under_load > 0:
        enc, qvecs = _query_vectors(cfg, of, split)
        k = int(cfg.get("goldenset.top_k", 10))
        holder: dict = {}

        def do_swap() -> None:
            time.sleep(under_load / 2)
            holder["swap"] = sh.point_alias(client, alias, dst)
            if rollback:
                time.sleep(1.0)
                holder["rollback"] = sh.point_alias(client, alias, src)

        th = threading.Thread(target=do_swap, daemon=True)
        typer.echo(f"\n  부하 {under_load:.0f}s · 워커 {workers} — 중간에 스왑\n")
        th.start()
        res = ld.run(lambda: connect(cfg), alias, qvecs, k, under_load, workers)
        th.join(timeout=30)
        swap_s, rb_s = holder.get("swap"), holder.get("rollback")
    else:
        swap_s = sh.point_alias(client, alias, dst)
        if rollback:
            rb_s = sh.point_alias(client, alias, src)

    _line(OK, "alias 스왑", f"{swap_s*1000:.0f} ms" if swap_s else "—")
    if rb_s is not None:
        _line(OK, "롤백", f"{rb_s*1000:.0f} ms")
    _line(OK, "최종 alias", f"{alias} → {sh.current_target(client, alias)}")

    payload = {
        "alias": alias, "from": src, "to": dst,
        "gate": {"metric": gate.metric, "baseline": gate.baseline,
                 "candidate": gate.candidate, "delta": gate.delta,
                 "threshold": gate.threshold, "passed": gate.passed},
        "swap_ms": round(swap_s * 1000, 1) if swap_s else None,
        "rollback_ms": round(rb_s * 1000, 1) if rb_s is not None else None,
    }
    if res is not None:
        pcs = res.percentiles()
        gaps = res.gaps(500.0)
        _line(OK if res.error_rate() == 0 else BAD, "부하 중 오류율",
              f"{res.error_rate()*100:.4f}%  ({len(res.samples):,}건)")
        _line(OK, "부하 중 지연",
              f"p50 {pcs['p50']:.2f}ms  p95 {pcs['p95']:.2f}ms  p99 {pcs['p99']:.2f}ms")
        _line(OK if not gaps else BAD, "다운타임",
              "0" if not gaps else f"{len(gaps)}구간 최대 {max((b-a)*1000 for a,b in gaps):.0f}ms")
        payload |= {"requests": len(res.samples), "error_rate": res.error_rate(),
                    "qps": res.qps(), **pcs, "gaps": gaps,
                    "windows": res.window_stats(1.0)}

    out = RESULTS / f"shift-{of}-to-{to}-{int(time.time())}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _line(OK, "결과", str(out.relative_to(_ROOT)))
    typer.echo("")
    typer.secho("스왑 완료.", fg=typer.colors.GREEN)


def _eval_scores(cfg, variant: str, split: str) -> dict:
    """컬렉션 하나를 골든셋으로 평가해 qrels 기준 점수를 돌려준다."""
    from . import collection as coll
    from . import dataset as ds
    from .client import connect
    from .evaluate import aggregate

    k = int(cfg.get("goldenset.top_k", 10))
    rel, topics = ds.load_qrels(split)
    enc, qvecs = _query_vectors(cfg, variant, split)
    client = connect(cfg)
    name = coll.name_for(cfg, variant)
    client.load_collection(collection_name=name)
    res = client.search(collection_name=name, data=qvecs, limit=k,
                        output_fields=["docid"])
    qids = list(topics)
    runs = {q: [h.get("docid") or h["entity"]["docid"] for h in hits]
            for q, hits in zip(qids, res)}
    m = aggregate(runs, rel, k)
    return {"ndcg": m[f"ndcg@{k}"], "recall": m[f"recall@{k}"]}


if __name__ == "__main__":
    app()
