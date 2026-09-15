#!/usr/bin/env bash
# 컨테이너 내부에서 본 랜덤 I/O 측정. D-004 의 순차 dd 수치를 보완한다.
#
# 왜 필요한가 — DiskANN·mmap 은 랜덤 읽기가 지배적이라 순차 수치로 갈음할 수 없고,
# Docker Desktop 의 VirtioFS 오버헤드가 호스트 수치 위에 얹힌다. 그래서 반드시
# "컨테이너 안에서" 재야 한다.
#
# 두 대상을 같은 조건으로 비교한다.
#   bind-external  : 저장소의 volumes/ — 외장 APFS 를 VirtioFS 로 바인드 마운트 (현재 구성)
#   volume-internal: Docker named volume — Docker Desktop VM 디스크(내장 SSD) 안의 ext4
#
# 사용: make iobench            (기본 20초 × 3패턴 × 2대상)
#       RUNTIME=60 make iobench

set -euo pipefail

RUNTIME="${RUNTIME:-20}"
SIZE="${SIZE:-512M}"
IODEPTH="${IODEPTH:-16}"
IMAGE="${IMAGE:-alpine:3.20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIND_DIR="$ROOT/volumes/.iobench"
VOLUME_NAME="vecshift_iobench"
OUT_DIR="${OUT_DIR:-$ROOT/results/iobench}"

# 패턴: 이름|rw|블록크기
PATTERNS=(
  "randread-4k|randread|4k"
  "randread-16k|randread|16k"
  "randwrite-4k|randwrite|4k"
)

# ── 가드레일 ────────────────────────────────────────────────────────────────
# 두 대상은 서로 다른 물리 디스크를 쓴다. 각각 따로 확인한다.
#   bind-external   → 저장소가 놓인 외장 볼륨
#   volume-internal → Docker Desktop VM 디스크 이미지가 놓인 디스크. 보통 **내장**이며,
#                     외장에 여유가 아무리 많아도 상관없다.
# 이 확인 없이 돌렸다가 내장을 0바이트까지 채우고 VM 을 read-only 로 만든 적이 있다
# (D-010 · D-011 · 「막힌 기록」 참고).
size_to_gib() {  # 12G / 512M / 1024K → GiB 정수(올림)
  python3 -c "
import math,sys
raw=sys.argv[1]; unit=raw[-1].upper()
v=float(raw[:-1]) if unit in 'KMGT' else float(raw)/1073741824
print(math.ceil({'K':v/1048576,'M':v/1024,'G':v,'T':v*1024}.get(unit, v)))" "$1"
}

free_gib() { df -g "$1" 2>/dev/null | awk 'NR==2 {print $4}'; }

NEED_GIB="$(size_to_gib "$SIZE")"
MARGIN_GIB="${MARGIN_GIB:-3}"          # fio 파일 외에 파일시스템이 숨쉴 여유
WANT_GIB=$((NEED_GIB + MARGIN_GIB))

# Docker VM 디스크 이미지가 실제로 놓인 경로
VM_IMAGE_DIR="$HOME/Library/Containers/com.docker.docker/Data"
[ -d "$VM_IMAGE_DIR" ] || VM_IMAGE_DIR="/System/Volumes/Data"

check() {  # $1=대상 이름  $2=확인할 경로  $3=추가 설명
  local free; free="$(free_gib "$2")"
  if [ -n "$free" ] && [ "$free" -lt "$WANT_GIB" ]; then
    cat >&2 <<MSG
  ! $1 건너뜀 — 여유 ${free} GiB < 필요 ${WANT_GIB} GiB (워킹셋 ${NEED_GIB} + 여유 ${MARGIN_GIB})
    $3
    → SIZE 를 줄일 것 (예: SIZE=2G make iobench)
MSG
    return 1
  fi
  return 0
}

SKIP_EXTERNAL=0
SKIP_INTERNAL=0
mkdir -p "$(dirname "$BIND_DIR")"
check "bind-external" "$(dirname "$BIND_DIR")" \
  "저장소가 놓인 볼륨이 부족하다." || SKIP_EXTERNAL=1
check "volume-internal" "$VM_IMAGE_DIR" \
  "named volume 은 내장의 Docker 디스크 이미지를 키운다. 강행하면 VM 이 read-only 로 떨어진다.
    Docker 디스크 이미지를 외장으로 옮기는 것도 방법이다." || SKIP_INTERNAL=1

if [ "$SKIP_EXTERNAL" -eq 1 ] && [ "$SKIP_INTERNAL" -eq 1 ]; then
  echo "  두 대상 모두 공간이 부족하다. 중단한다." >&2
  exit 1
fi
# ────────────────────────────────────────────────────────────────────────────

mkdir -p "$BIND_DIR" "$OUT_DIR"
if [ "$SKIP_INTERNAL" -eq 0 ]; then docker volume create "$VOLUME_NAME" >/dev/null; fi

run_one() {  # $1=대상 이름  $2=docker -v 인자  $3=패턴이름  $4=rw  $5=bs
  local target="$1" mount="$2" name="$3" rw="$4" bs="$5"
  printf '  %-16s %-14s ' "$target" "$name" >&2
  local json="$OUT_DIR/$target-$name.json" log="$OUT_DIR/$target-$name.log"
  # stderr 는 버리지 않고 남긴다. fio 실패(EIO 등)는 로그 없이는 진단이 불가능하다.
  if ! docker run --rm -v "$mount" "$IMAGE" sh -c "
    apk add --no-cache fio >/dev/null 2>&1 || { echo 'fio 설치 실패' >&2; exit 1; }
    fio --name=$name --directory=/bench --filename=fio.dat --size=$SIZE \
        --rw=$rw --bs=$bs --ioengine=libaio --direct=1 --iodepth=$IODEPTH \
        --numjobs=1 --runtime=$RUNTIME --time_based --group_reporting \
        --output-format=json
  " > "$json" 2> "$log"; then
    echo "FAILED — $log 참고" >&2
    sed 's/^/      /' "$log" >&2
    return 1
  fi
  echo "done" >&2
}

echo "랜덤 I/O 측정 — ${RUNTIME}s × ${#PATTERNS[@]}패턴 × 2대상 (size=$SIZE, iodepth=$IODEPTH)" >&2
for p in "${PATTERNS[@]}"; do
  IFS='|' read -r name rw bs <<< "$p"
  if [ "$SKIP_EXTERNAL" -eq 0 ]; then
    run_one "bind-external"   "$BIND_DIR:/bench"    "$name" "$rw" "$bs"
  fi
  if [ "$SKIP_INTERNAL" -eq 0 ]; then
    run_one "volume-internal" "$VOLUME_NAME:/bench" "$name" "$rw" "$bs"
  fi
done

# 정리 — 벤치 파일은 남기지 않는다. JSON 만 results/ 에 남는다.
# 주의 — rm 만으로는 공간이 즉시 돌아오지 않는다. Docker 의 VirtioFS 프로세스가
# 삭제된 파일 핸들을 계속 붙들기 때문이다. df 가 안 줄면 Docker Desktop 을 재시작할 것.
rm -rf "$BIND_DIR"
if [ "$SKIP_INTERNAL" -eq 0 ]; then docker volume rm "$VOLUME_NAME" >/dev/null; fi

python3 "$ROOT/scripts/iobench_report.py" "$OUT_DIR"
