.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := .venv/bin/python
VECSHIFT := .venv/bin/vecshift

# 캐시를 전부 저장소(외장) 안으로 돌린다. 기본값은 ~/.cache 즉 내장이고,
# 코퍼스·모델 가중치·torch 휠이 거기 쌓이면 내장이 찬다 — D-011, D-012.
export UV_CACHE_DIR := $(CURDIR)/cache/uv
export HF_HOME := $(CURDIR)/cache/huggingface

.PHONY: help venv venv-full up down logs ps doctor smoke ingest ingest-v2 ingest-v3 goldenset eval sweep load shift iobench clean nuke

help:  ## 사용 가능한 타깃
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv:  ## Python 3.12 가상환경 + 코어 의존성
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e .
	@echo '준비 완료 — make doctor'

venv-full:  ## 코어 + 데이터셋 + 임베딩 의존성 (torch 포함, 수 GB)
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e '.[data,embed,viz]'
	@echo '준비 완료 — make ingest'

up:  ## Milvus Standalone 기동
	docker compose up -d
	@echo 'healthz 대기 중...'
	@for i in $$(seq 1 60); do \
		if curl -sf http://localhost:9091/healthz >/dev/null 2>&1; then \
			echo '  Milvus ready'; exit 0; fi; \
		sleep 3; done; \
		echo '  타임아웃 — make logs 로 확인'; exit 1

down:  ## 중지 (데이터 보존)
	docker compose down

ps:  ## 컨테이너 상태
	docker compose ps

logs:  ## Milvus 로그 추적
	docker compose logs -f standalone

doctor:  ## 환경 점검
	$(VECSHIFT) doctor

smoke:  ## 왕복 검증
	$(VECSHIFT) smoke

ingest:  ## MIRACL 적재 — 다운로드 → 샘플링 → 임베딩 → Milvus
	$(VECSHIFT) ingest

goldenset:  ## qrels 에서 골든셋 추출
	$(VECSHIFT) goldenset

eval:  ## 골든셋으로 nDCG@10 · Recall@10 계산 (qrels 기준 — D-007)
	$(VECSHIFT) eval

ingest-v2:  ## v2(e5-base 768d) 적재 — 비교용
	$(VECSHIFT) ingest --variant v2

ingest-v3:  ## v3(bge-m3 1024d) 적재 — 재색인 대상 (D-033)
	$(VECSHIFT) ingest --variant v3

load:  ## 동시 클라이언트 부하 — 진짜 QPS 와 정지 구간
	$(VECSHIFT) load

shift:  ## v1 → v3 무중단 스왑 (부하 30s 중간에 스왑 + 롤백, 쓰기 8/s)
	$(VECSHIFT) shift --from v1 --to v3 --under-load 30 --rollback --writes 8

sweep:  ## M1 — 인덱스 파라미터 스윕과 파레토 곡선
	$(VECSHIFT) sweep

iobench:  ## 컨테이너 내부 랜덤 I/O 측정 (외장 bind vs 내장 volume)
	bash scripts/iobench.sh

clean:  ## 파이썬 캐시 정리
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

nuke:  ## 컨테이너 + 볼륨 전체 삭제 (되돌릴 수 없음)
	docker compose down -v
	rm -rf volumes
