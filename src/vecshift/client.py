"""Milvus 접속. 연결 문자열만 바꾸면 Lite/Standalone/Distributed 가 동일 코드다."""

from __future__ import annotations

from typing import Any

from pymilvus import MilvusClient

from .config import Config


def connect(cfg: Config) -> MilvusClient:
    uri = cfg.require("milvus.uri")
    token = cfg.get("milvus.token") or ""
    kwargs: dict[str, Any] = {"uri": uri}
    if token:
        kwargs["token"] = token
    db = cfg.get("milvus.db_name")
    if db and db != "default":
        kwargs["db_name"] = db
    return MilvusClient(**kwargs)


def server_version(client: MilvusClient) -> str:
    """서버 버전 조회. pymilvus 버전에 따라 경로가 다르므로 방어적으로 처리한다."""
    try:
        return str(client.get_server_version())
    except Exception:
        pass
    try:
        from pymilvus import utility  # type: ignore

        return str(utility.get_server_version())
    except Exception:
        return "unknown"
