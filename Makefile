.DEFAULT_GOAL := help
SHELL := /bin/bash
PY := .venv/bin/python
VECSHIFT := .venv/bin/vecshift

.PHONY: help venv up down logs ps doctor smoke iobench clean nuke

help:  ## 사용 가능한 타깃
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv:  ## Python 3.12 가상환경 + 의존성 설치
	uv venv --python 3.12 .venv
	uv pip install --python .venv/bin/python -e .
	@echo '준비 완료 — make doctor'

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

iobench:  ## 컨테이너 내부 랜덤 I/O 측정 (외장 bind vs 내장 volume)
	bash scripts/iobench.sh

clean:  ## 파이썬 캐시 정리
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

nuke:  ## 컨테이너 + 볼륨 전체 삭제 (되돌릴 수 없음)
	docker compose down -v
	rm -rf volumes
