"""임베딩. e5 계열의 접두사 규약을 코드로 강제한다.

e5 는 질의에 `query: `, 문서에 `passage: ` 를 붙이는 것을 전제로 학습됐다.
빠뜨려도 에러가 나지 않고 **점수만 조용히 떨어진다** — 그래서 설정에 두고
여기서 강제한다 (D-005).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .paths import setup_hf_cache


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder:
    """설정의 variant(v1/v2) 하나에 대응하는 인코더."""

    def __init__(self, variant: dict, batch_size: int = 64, normalize: bool = True,
                 device: str | None = None):
        setup_hf_cache()  # 모델 가중치도 외장으로 — D-012
        from sentence_transformers import SentenceTransformer

        self.id = variant["id"]
        self.dim = int(variant["dim"])
        self.metric = variant.get("metric", "COSINE")
        self.query_prefix = variant.get("query_prefix", "")
        self.passage_prefix = variant.get("passage_prefix", "")
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device or pick_device()
        self.model = SentenceTransformer(self.id, device=self.device)

        got = self.model.get_sentence_embedding_dimension()
        if got != self.dim:
            raise ValueError(
                f"차원 불일치: 설정 {self.dim} vs 모델 {got} ({self.id}). "
                f"config/vecshift.yml 의 embedding.*.dim 을 고치세요."
            )

    def _encode(self, texts: list[str], prefix: str, progress: bool) -> np.ndarray:
        return self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=progress,
        ).astype("float32")

    def passages(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.passage_prefix, progress)

    def queries(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self._encode(texts, self.query_prefix, progress)


def passage_text(doc: dict) -> str:
    """제목과 본문을 합친다. MIRACL 문서는 제목이 맥락을 크게 보탠다."""
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title}\n{text}" if title else text


def batched(items: Iterable, n: int):
    buf = []
    for it in items:
        buf.append(it)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
