"""vecshift CLI. W1 범위: doctor / smoke."""

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


if __name__ == "__main__":
    app()
