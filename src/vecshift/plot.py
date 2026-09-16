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
                 recall_key: str, title: str, subtitle: str = "",
                 zoom_from: float = 0.85) -> Path:
    """왼쪽은 전체, 오른쪽은 결정 구간 확대.

    프론티어 점이 recall 0.85~1.0 에 몰려서 한 패널에 라벨을 다 넣으면 겹친다.
    그런데 **실제 의사결정은 정확히 그 구간에서 일어난다.** 그래서 확대 패널을 둔다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_all, ax_zoom) = plt.subplots(
        1, 2, figsize=(15.5, 6.8), dpi=140,
        gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.18})

    def draw(ax, data, front, annotate: bool) -> None:
        by_type: dict[str, list[dict]] = {}
        for r in data:
            by_type.setdefault(r["index_type"], []).append(r)
        for t, group in sorted(by_type.items()):
            x = [g[recall_key] for g in group]
            y = [g["qps"] for g in group]
            color = INDEX_COLORS.get(t, "#555555")
            # 대표값이 best-of-N 이므로 막대는 아래로만 뻗는다 — 느린 이상치의 범위다.
            if all("qps_min" in g for g in group):
                lo = [max(0.0, g["qps"] - g["qps_min"]) for g in group]
                ax.errorbar(x, y, yerr=[lo, [0.0] * len(group)], fmt="none",
                            ecolor=color, elinewidth=1.1, capsize=2.5,
                            alpha=0.5, zorder=2)
            ax.scatter(x, y, label=t, s=58, alpha=0.9, color=color,
                       edgecolors="white", linewidths=0.8, zorder=3)
        if front:
            ax.plot([f[recall_key] for f in front], [f["qps"] for f in front],
                    color="#333333", lw=1.4, ls="--", zorder=2,
                    label="파레토 프론티어")
            if annotate:
                for i, f in enumerate(front):
                    ax.annotate(f["label"], (f[recall_key], f["qps"]),
                                textcoords="offset points",
                                xytext=(7, 9 if i % 2 == 0 else -14),
                                fontsize=7.4, color="#222222",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                          ec="none", alpha=0.8))
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.22, lw=0.6)
        ax.margins(x=0.14)

    draw(ax_all, rows, frontier, annotate=False)
    ax_all.set_title("전체 38개 설정", fontsize=10.5)
    ax_all.axvspan(zoom_from, 1.02, color="#000000", alpha=0.045, zorder=0)
    ax_all.legend(loc="lower left", fontsize=8, framealpha=0.92)

    zr = [r for r in rows if r[recall_key] >= zoom_from]
    zf = [f for f in frontier if f[recall_key] >= zoom_from]
    draw(ax_zoom, zr, zf, annotate=True)
    ax_zoom.set_title(f"결정 구간 확대 — recall ≥ {zoom_from}", fontsize=10.5)

    pretty = recall_key.replace("ann_recall", "recall")
    for ax in (ax_all, ax_zoom):
        ax.set_xlabel(f"ANN {pretty}  (FLAT 브루트포스 대비 — 인덱스 품질)", fontsize=9)
    ax_all.set_ylabel("QPS  (배치 처리량 best-of-N, 막대는 느린 쪽 폭)", fontsize=9)

    fig.suptitle(f"{title}\n{subtitle}" if subtitle else title,
                 fontsize=12.5, linespacing=1.5, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
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
