# 라이선스·기여 준수 점검

이 문서는 VecShift 저장소가 외부 저작물을 어떻게 쓰고 있는지, 업스트림에 무엇을 기여했는지
한자리에 모은 기록이다. 결정의 맥락은 `DECISIONS.md` D-030 · D-031 에 있다.

점검일: 2026-09-16

---

## 1. 요약

| 항목 | 상태 |
|---|---|
| 저장소 라이선스 | Apache-2.0 (`LICENSE` 전문 포함, GitHub 인식 확인) |
| vendoring 한 파일 | `docker-compose.yml` 1개 — Apache-2.0 §4 의무 충족 |
| 런타임 의존(데이터셋·모델) | 재배포하지 않음. 출처는 `NOTICE` 에 기재 |
| 상표 | 설명적 사용. Apache-2.0 §6 의 예외 범위 |
| 업스트림 기여 | milvus-io/milvus PR #53431 에 댓글 1건 |

---

## 2. vendoring — `docker-compose.yml`

Milvus v3.0.1 의 공식 `milvus-standalone-docker-compose.yml` 을 가져와 두 군데 수정했다.
원본은 Apache-2.0 이고 우리 저장소도 Apache-2.0 이다.

### Apache-2.0 §4 의무 확인

| 조항 | 요구 | 상태 |
|---|---|---|
| §4(a) | 라이선스 사본 제공 | `LICENSE` 전문 추가 |
| §4(b) | **수정 사실 고지** | 파일 헤더에 `[수정됨] … D-008 / D-009` 인라인 표기 + `NOTICE` |
| §4(c) | 원본 저작권 고지 유지 | **원본에 저작권 헤더가 없다** (`version: '3.5'` 로 시작) — 유지할 대상 없음 |
| §4(d) | NOTICE 파일 포함 | **Milvus 저장소에 NOTICE 파일이 없다** — 의무 없음 |

### 수정 내역

| 무엇 | 왜 | 근거 |
|---|---|---|
| `minio/minio` → `quay.io/minio/minio` (태그 동일) | MinIO 가 Docker Hub 배포를 내려서 **원본 그대로는 기동 불가** | D-008 |
| 최상위 `version` 속성 제거 | Compose v2 가 무시하며 매번 경고를 출력 (기능 영향 없음) | D-009 |

### 참고

D-003 에서 "공식 파일을 수정하면 무엇을 왜 바꿨는지 문서에 남긴다"는 원칙을 세웠다.
라이선스를 의식하고 만든 규칙이 아니었는데, 결과적으로 **§4(b) 를 그대로 충족**했다.

---

## 3. 런타임 의존 — 재배포하지 않음

아래는 저장소에 포함되지 않는다. `.gitignore` 대상이며 사용자가 실행 시 내려받는다.
**재배포하지 않으므로 라이선스 의무가 발생하지 않지만**, 출처는 `NOTICE` 에 적었다.
의무가 없는 것과 밝히지 않아도 되는 것은 다르다.

| 대상 | 라이선스 | 용도 |
|---|---|---|
| `miracl/miracl-corpus` (ko) | Apache-2.0 | 코퍼스 |
| `miracl/miracl` (ko) qrels | Apache-2.0 | 사람이 라벨링한 정답 |
| `intfloat/multilingual-e5-small` | MIT | v1 임베딩(384d) |
| `intfloat/multilingual-e5-base` | MIT | v2 임베딩(768d) |

---

## 4. 상표

Apache-2.0 **§6 은 상표권을 부여하지 않는다.** 다만 같은 조항이 예외를 둔다.

> except as required for **reasonable and customary use in describing the origin of the Work**

Milvus 는 LF AI & Data Foundation 산하 프로젝트이고 Zilliz 가 주요 기여자다.

이 저장소는 "Milvus 를 다루는 개인 학습·포트폴리오 프로젝트"를 **설명하기 위해** 이름을
쓴다. 공식 프로젝트를 사칭하거나 후원·제휴를 암시하지 않고, 로고를 자기 브랜드로 쓰지
않으며, 이름을 걸고 무언가를 팔지 않는다. §6 의 예외 범위에 든다.

명확성을 위해 저장소 설명에 독립 프로젝트임을 명시했다.

---

## 5. 업스트림 기여

### 경위

D-008 에서 겪은 MinIO 이미지 문제를 Milvus 에 보고하려 했다. **먼저 검색했더니 이미 있었다.**

| | |
|---|---|
| 이슈 [#53430](https://github.com/milvus-io/milvus/issues/53430) | 같은 문제, Docker Hub API 검증까지 동일 |
| PR [#53431](https://github.com/milvus-io/milvus/pull/53431) | 8개 파일 수정 — 우리가 찾은 3개를 전부 포함 |

중복 이슈나 중복 PR 은 기여가 아니라 소음이다. 초안에 있던
"`3.0` · `2.6` 브랜치 cherry-pick 필요" 지적도 버렸다 — Milvus 는 `[cp 3.0]` PR 을
일상적으로 돌린다.

### 실제로 낸 것

아무도 짚지 않은 **한 가지**만 남겼다.

| 출처 | minio 태그 |
|---|---|
| `v3.0.1` git 태그 안의 파일 | `RELEASE.2024-05-28T17-19-04Z` |
| v3.0.1 **배포된 릴리스 첨부파일** | `RELEASE.2024-12-18T13-15-44Z` |

설치 문서가 `wget` 으로 받으라는 파일은 저장소 파일이 아니다. 저장소 전체 검색에서
해당 태그도 파일명도 **0건**이다. → **저장소 파일 8개를 고쳐도 릴리스 산출물 생성 경로가
그대로면 다음 릴리스에서 재발한다.**

게시: [PR #53431 댓글](https://github.com/milvus-io/milvus/pull/53431#issuecomment-5694250402)

### 교차 참조에 대하여

우리 저장소의 PR 이 위 댓글을 링크해서 업스트림 스레드에 교차 참조가 생겼다.
이는 GitHub 이 자동 생성하는 타임라인 이벤트이며, 공개 저장소끼리의 링크는 플랫폼의
기본 동작이다. 라이선스·법적 쟁점과 무관하다.

다만 업스트림 스레드는 영어권이므로 **참조되는 PR 제목은 영문으로** 둔다.
처음에 한국어 제목이었는데 번역 시 자기 평가처럼 읽혀 중립적 영문으로 바꿨다.

### 남은 후속

#53431 이 머지된 뒤에도 릴리스 첨부파일이 그대로면, 그때는 별도 이슈가 중복이 아니다.
확인은 한 줄이면 된다.

```bash
curl -sL "https://github.com/milvus-io/milvus/releases/download/<새버전>/milvus-standalone-docker-compose.yml" \
  | grep "image:.*minio"
```

`minio/minio` 면 별도 이슈, `quay.io/minio/minio` 면 해결된 것이다.

---

## 6. 이 문서의 한계

법률 자문이 아니다. 공개된 라이선스 전문과 저장소 상태를 대조한 기록이다.
실제 분쟁 가능성을 판단해야 하는 상황이라면 변호사에게 확인해야 한다.
