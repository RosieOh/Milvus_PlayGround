"""파레토 곡선 출력. M1 의 산출물은 "하나의 그래프"다(기획서 07).

절대 수치보다 **무엇을 포기하고 무엇을 얻는가**가 읽혀야 한다. 그래서 축은
recall(품질)과 QPS(속도)로 고정하고, 프론티어를 선으로 잇는다.
"""

from __future__ import annotations

from pathlib import Path

INDEX_COLORS = {
    "IVF_FLAT": "#1f77b4", "IVF_SQ8": "#17becf", "IVF_PQ": "#9467bd",
    "HNSW": "#d62728", "SCANN": "#2ca02c", "DISKANN": "#ff7f0e",
    "FLAT": "#7f7f7f",
}


def pareto_chart(rows: list[dict], frontier: list[dict], out: Path,
                 recall_key: str, title: str, subtitle: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=140)

    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["index_type"], []).append(r)

    for t, group in sorted(by_type.items()):
        x = [g[recall_key] for g in group]
        y = [g["qps"] for g in group]
        color = INDEX_COLORS.get(t, "#555555")
        # QPS 는 반복 간 변동이 크다. 폭을 오차막대로 그리지 않으면 점 사이의
        # 작은 차이를 유의한 것으로 읽게 된다 (D-021).
        if all("qps_min" in g and "qps_max" in g for g in group):
            lo = [max(0.0, g["qps"] - g["qps_min"]) for g in group]
            hi = [max(0.0, g["qps_max"] - g["qps"]) for g in group]
            ax.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=color,
                        elinewidth=1.1, capsize=2.5, alpha=0.55, zorder=2)
        ax.scatter(x, y, label=t, s=58, alpha=0.9, color=color,
                   edgecolors="white", linewidths=0.8, zorder=3)

    if frontier:
        ax.plot([f[recall_key] for f in frontier], [f["qps"] for f in frontier],
                color="#333333", lw=1.4, ls="--", zorder=2,
                label="파레토 프론티어")
        for f in frontier:
            ax.annotate(f["label"], (f[recall_key], f["qps"]),
                        textcoords="offset points", xytext=(6, 5),
                        fontsize=7.5, color="#333333")

    # "ann_recall@10" → "ANN recall@10". 키를 그대로 쓰면 ANN 이 두 번 나온다.
    pretty = recall_key.replace("ann_recall", "recall")
    ax.set_xlabel(f"ANN {pretty}  (FLAT 브루트포스 대비 — 인덱스 품질)")
    ax.set_ylabel("QPS  (배치 처리량 중앙값, 막대는 반복 간 폭 — 동시 QPS 가 아니다)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.22, lw=0.6)

    # 부제를 별도 text 로 얹으면 제목과 겹친다. 제목 문자열에 줄바꿈으로 넣는다.
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title,
                 fontsize=12.5, linespacing=1.6)

    # 프론티어 주석이 오른쪽 끝에서 잘리지 않도록 여백을 준다.
    ax.margins(x=0.13)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def use_korean_font() -> str | None:
    """한글 폰트를 찾아 설정한다. 없으면 라벨이 네모로 깨진다."""
    import matplotlib
    from matplotlib import font_manager

    for cand in ("AppleSDGothicNeo", "Apple SD Gothic Neo", "NanumGothic",
                 "Malgun Gothic", "Noto Sans CJK KR"):
        for f in font_manager.fontManager.ttflist:
            if cand.replace(" ", "").lower() in f.name.replace(" ", "").lower():
                matplotlib.rcParams["font.family"] = f.name
                matplotlib.rcParams["axes.unicode_minus"] = False
                return f.name
    return None
