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
        # MPS 에서 처리량이 평평해지는 배치 상한은 모델마다 다르다 —
        # e5-base 는 64 이하, bge-m3 는 16 이하. 그래서 변형별로 덮어쓸 수 있게 한다(D-033).
        if variant.get("batch_size"):
            self.batch_size = int(variant["batch_size"])
        self.normalize = normalize
        self.device = device or pick_device()
        self.model = SentenceTransformer(self.id, device=self.device)
        # fp16 은 MPS 에서 약 21 % 빠르지만 벡터가 미세하게 바뀐다 — top-10 이웃이
        # 98.3 % 만 겹친다. 그래서 기본은 끄고, 쓰려면 variant 에 명시한다(D-032).
        # 모델마다 기본 최대 길이가 다르다(e5 512, bge-m3 8192). 비교할 때는 맞춰야 한다 —
        # 한쪽만 길게 읽으면 모델이 아니라 읽은 분량을 비교하게 된다(D-033).
        if variant.get("max_seq_length"):
            self.model.max_seq_length = int(variant["max_seq_length"])
        self.max_seq_length = self.model.max_seq_length
        self.fp16 = bool(variant.get("fp16", False))
        if self.fp16:
            self.model = self.model.half()

        # sentence-transformers 가 이름을 바꿨다. 신구 양쪽을 본다.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        got = get_dim()
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
