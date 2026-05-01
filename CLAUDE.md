# CLAUDE.md

> 가족같이 곁에 있는 텍스트 기반 AI 비서 — 사람·관계·일상을 기억하고 먼저 챙기는 동반자

---

## 0. TL;DR (Claude Code 에게)

이 프로젝트는 **로컬 단일 사용자 웹 채팅 비서**다. 두 축이 전부다:

1. **Text-first**: JARVIS 풍의 3-컬럼 웹 UI 가 유일한 사용자 표면. 음성 채널은 폐기됨.
2. **Memory-first**: 모든 대화·관계·이벤트는 누적·연결되며, 시간이 갈수록 똑똑해진다.

이 두 축이 흔들리는 결정은 거부하라.

### 0.1 커뮤니케이션 스타일 (Claude Code 응답 규칙)

- **사용자는 한국어로 명령한다. Claude 는 영어로 응답한다.** 토큰 소비를 최소화하기 위함.
- 단, 다음 경우엔 한국어로 응답한다:
  - 사용자가 명시적으로 한국어 응답을 요청한 경우.
  - **UI 에 표시될 자연어 문장**(에이전트 시스템 프롬프트, 응답 메시지 템플릿 등) 을 작성하는 경우 — 이건 응답이 아니라 **산출물**이라 §2.1 이 우선한다.
- 응답은 **간결하게**. 불필요한 머리말·요약·확인 멘트 금지.
- 코드/파일 경로/명령어 등 고유명사는 그대로 둔다 (번역하지 말 것).

---

## 1. 프로젝트 개요

### 1.1 무엇을 만드는가
사용자와 가족 구성원을 기억하고 챙겨주는 **로컬 웹 채팅 비서**. 브라우저에서 `127.0.0.1:8765` 로 접속해 채팅한다. 영화 *Her* 의 사만다처럼 가족 같은 친밀감을 추구하되, 표면은 음성이 아닌 텍스트.

### 1.2 핵심 사용 시나리오
- 채팅창에 "어머니 생신이 다음 주야. 케이크 예약해줘" → 작년 어머니가 좋아하셨던 단호박 케이크를 메모리에서 떠올려 제안. 우측 패널에 떠올린 fact 카드가 뜬다.
- "오늘 저녁에 뭐 먹지?" → 가족 알러지·최근 식단·날씨까지 종합 추천.
- (Consolidator 가 발견한 "매주 금요일 외식" 패턴이 자연스럽게 다음 답변에 반영됨)

### 1.3 비목표 (Out of Scope)
- 음성 인터페이스 (STT/TTS/Wake Word/마이크 동의) — 폐기.
- 풀 데스크톱/모바일 GUI 앱 (Tauri/Electron 등). 웹 UI 가 유일한 표면.
- 클라우드 다중 사용자 SaaS — 가정/개인 단위 **로컬 앱** 으로 한정.
- launchd 데몬 / 백그라운드 상시 실행. 사용자가 명시적으로 `make web` 실행할 때만 동작.

---

## 2. 핵심 설계 원칙

### 2.1 Text-first
- 단일 사용자 표면은 `apps/web` — FastAPI + WebSocket 서버 + 정적 프론트엔드 (3-컬럼 JARVIS 풍).
- 응답은 **마크다운 친화**. 코드블록·표·링크 자유.
- 첨부파일은 텍스트 채널의 멀티모달 메시지로 전송 (이미지·PDF·텍스트·md).

### 2.2 Memory as Living Tissue
메모리는 단순 로그가 아니라 **사람·시간·맥락의 그래프**.
- **Person**(사람) 이 중심 노드. 그 주변에 Facts·Events·Preferences·Notes 가 매달림.
- **Session** = 한 채팅 세션 (제목·요약). **Message** = 한 턴 (role/content/ts).
- **Note** = 사람에 매이지 않는 자유 메모 / 결정 사항 / 할일.
- 매일 03:00 (수동/스케줄러로) **메모리 통합(Consolidation)**: 직전 24h Session 들 → 장기 fact / event / note 로 승격.

### 2.3 Local-First & Privacy
- 가족 정보는 외부로 나가지 않는 것이 기본.
- 모든 영구 저장은 **로컬 SQLite + sqlite-vec**.
- LLM 호출 시에는 **익명화된 컨텍스트만** 전송 (실명 → 별칭 매핑).
- **익명화 경계**: Memory Layer 는 항상 실명으로 동작. 익명화는 **Agent Core 의 LLM 어댑터 경계**에서만 수행한다. 매핑 테이블은 메모리 내에서만 유지하고, 응답 수신 후 별칭 → 실명 으로 역매핑한다. 이 경계가 흐려지면 PR 거부.
- **웹 서버 바인딩**: `127.0.0.1` 만 허용. 모든 REST 라우트는 loopback 클라이언트만 응답한다.

### 2.4 Proactive 제안 (Phase 4 — 추후)
Consolidator 가 발견한 패턴(예: "매주 금요일 저녁엔 외식")을 채팅 응답에 자연스럽게 반영. 사용자가 채팅을 안 할 때 **선제적으로 알림을 보내지는 않는다** (백그라운드 데몬 폐기됨, §1.3).

---

## 3. 기술 스택 (기본 선택)

### 3.1 LLM 정책 — Gemini 전용 (Hard Rule)

- **모든 LLM·임베딩 호출은 Gemini 패밀리 모델만 사용한다.** OpenAI / Anthropic / 기타 LLM 제공자 금지.
- **모델 선택은 태스크별 최적화**: 단일 모델로 통일하지 않는다. 추론·임베딩·요약·통합 등 각 태스크에 가장 적합한 Gemini 모델을 Claude Code 가 **제안**하고 **사용자 승인 후** 확정한다.
- **승인 워크플로**:
  1. Claude Code 가 새 LLM 호출 지점을 추가하거나 기존 모델을 바꾸기 전, **후보 모델 + 근거**(품질·지연·비용·컨텍스트 길이) 를 제시한다.
  2. 사용자가 승인하면 §3.2 표에 기록 + §12 결정 로그에 추가.
  3. 미승인 상태로 새 모델 ID 를 코드/설정에 박지 않는다.

### 3.2 스택 표

| 영역 | 선택 | 이유 / 대체안 |
|---|---|---|
| 언어/런타임 | Python 3.12 + asyncio | 비동기 + 생태계 |
| 백엔드 | FastAPI + WebSocket + uvicorn | 실시간 토큰 스트리밍 |
| LLM (메인 추론) | `gemini-3.1-pro-preview` | 한국어·멀티모달·도구 사용 강함 |
| LLM (배치·요약·Consolidator) | `gemini-2.5-flash` | 야간 배치 — Pro 대비 ~10-20× 저렴 |
| 임베딩 | `gemini-embedding-001`, **768d**, `task_type` 비대칭 사용 | 한국어·MRL (768→1536→3072 무재학습 확장). DOCUMENT 쓰기 / QUERY 검색 비대칭 |
| 구조 메모리 | SQLite + sqlite-vec | 로컬·임베디드·벡터 동시. 단일 파일 백업 |
| 프론트엔드 | Vanilla HTML + ES modules | 빌드 단계 없음. `apps/web/web/` 정적 자원 |
| 패키지 매니저 | uv | 빠름·결정적 |
| 환경 변수 | direnv (`.envrc`) | 비밀키 로컬 격리 (`GEMINI_API_KEY` 만 필수) |

> **변경 권한**: 표의 "선택" 컬럼 변경은 **사용자 승인 필요**. "대체안" 범위 내 변경은 자율.

**임베딩 운영 규칙** (sqlite-vec 사용 시 강제):
- 한 인덱스 안에서 **차원·모델 ID 혼합 금지**. 차원 변경은 마이그레이션으로 처리.
- 매 벡터 행에 `model_id`, `dim`, `task_type` 메타 컬럼 함께 저장 (미래 모델 교체 시 모호성 제거).
- 쓰기: `task_type=RETRIEVAL_DOCUMENT`. 검색 쿼리: `task_type=RETRIEVAL_QUERY`.

---

## 4. 아키텍처

```
 [Browser]
   │
   ├── HTTP   GET / GET /static/* GET /api/*  ──┐
   ├── WS     /ws/chat (token stream + recall)  │
   │                                            ▼
   │                                 FastAPI app (apps/web/server.py)
   │                                       │
   │                                       ▼
   │                                Agent Core (apps/agent/core.py)
   │                                       │
   │                                       ▼
   │                                Memory Layer (apps/memory/)
   │                                       ▲
   │                                       │
   └────────────────── Consolidator (수동/스케줄러로 03:00 배치 가능)
                          (apps/consolidator/)
```

### 4 핵심 모듈
1. **Web App** (`apps/web/`): FastAPI 서버 + WebSocket 채팅 + 정적 프론트엔드. 단일 사용자 표면.
2. **Agent Core** (`apps/agent/`): LLM 추론 (Gemini) + 도구 호출 + 회상 + 익명화 경계.
3. **Memory Layer** (`apps/memory/`): 사람·세션·메시지·사실·메모·이벤트·첨부 + 벡터 검색. SQLite + sqlite-vec.
4. **Consolidator** (`apps/consolidator/`): 단기→장기 메모리 정리. 수동 실행 (`make consolidate`) 또는 외부 스케줄러로 호출.

각 모듈은 명확한 인터페이스로 분리. 모듈 경계를 가로지르는 의존성은 PR 에서 거부.

---

## 5. Memory System (이 프로젝트의 심장)

### 5.1 3단 계층

| 계층 | 보관 | 형태 | 비유 |
|---|---|---|---|
| **Working** | 현재 세션 | messages 테이블 | 지금 떠올린 것 |
| **Episodic** | 30일 (정책) | 세션 요약 + 메시지 임베딩 | 일기 |
| **Semantic** | 영구 | 구조화된 facts / events / notes | 그 사람에 대한 앎 |

### 5.2 핵심 엔티티 (스키마 v2, 발췌)

```sql
people(id, name, relation, birthday, preferences_json, ...)

sessions(id, started_at, last_active_at, title, summary, archived_at)
messages(id, session_id, role, content, ts)

events(id, person_id, type, title, when_at, recurrence, source, status)

facts(id, subject_person_id, predicate, object, confidence,
      source_session_id, valid_from, archived_at)

preferences(person_id NULLable, domain, value, last_seen_at)

notes(id, content, tags JSON, source_session_id, created_at, updated_at, archived_at)

attachments(id, session_id, sha256, mime, ext, byte_size, path, description, ingested_at)

vec_sessions(session_id, embedding[768])
vec_messages(message_id, embedding[768])
session_embedding_meta(session_id, model_id, dim, task_type)
message_embedding_meta(message_id, model_id, dim, task_type)
```

**Schema version**: `PRAGMA user_version = 2`. v1 (episodes-only) DB 가 발견되면 자동으로 `db.sqlite.bak-<ts>` 로 백업 + v2 새로 생성 + 한국어 stderr 안내.

### 5.3 통합(Consolidation) 규칙
- 매일 03:00 권장 (스케줄러는 사용자/외부 cron 책임).
- 직전 24시간 sessions → 메시지 본문을 합쳐 LLM 에 보냄.
- `gemini-2.5-flash` 가 fact / event / note 후보를 JSON 으로 추출.
- Fact: `confidence ≥ 0.7` 만 semantic 으로 승격. 같은 subject+predicate, 다른 object → 이전 fact `archived_at` 기록 (삭제 X).
- Event: 그대로 events 테이블에 추가.
- Note: 동일 content 가 이미 있으면 dedup, 아니면 추가.
- 결과 로그는 `data/consolidation_log/<YYYY-MM-DD>.json` 누적.

### 5.4 회상(Recall) 전략
질문 들어오면 다섯 갈래 동시 조회:
1. **Structured** — 사람 이름 키워드 매치 → SQL facts + upcoming events.
2. **Semantic** — 메시지 임베딩 → vec_messages KNN → 부모 session 으로 조인. 최근 7일 session 은 가중 ×1.5.
3. **Recency fallback** — semantic 결과가 비면 가장 최근 5개 active session.
4. **Notes** — content LIKE 키워드 매치 (임베딩 기반은 이후 PR).
5. **Attachments** — 활성 session 의 최근 첨부 (있을 때만).

---

## 6. 웹 표면 세부

### 6.1 Routes (FastAPI)
```
GET    /                            → web/index.html
GET    /static/*                    → 정적 자원 (no-store)
GET    /healthz                     → liveness
GET    /api/sessions                → 최근 50개
POST   /api/sessions                → 신규
GET    /api/sessions/{id}/messages  → 메시지 풀 히스토리
DELETE /api/sessions/{id}           → soft archive (archived_at)
GET    /api/memory/{notes,people,facts,probe}
WS     /ws/chat                     → 토큰 스트림 + recall sidechannel + done/error
```

모든 REST 라우트는 loopback 클라이언트만 응답 (403 otherwise). 외부 노출 금지.

### 6.2 WS 프로토콜 (`/ws/chat`)
- 서버 → 클라 hello: `{type:"hello", schema_version:2, ts}`
- 클라 → 서버: `{type:"message", session_id?:int, content:str}`
- 서버 → 클라:
  - `{type:"recall", session_id, facts, notes, events, sessions}` (한 번)
  - `{type:"token", session_id, text}` (스트리밍)
  - `{type:"done", session_id, chars}`
  - `{type:"error", message}`

### 6.3 Frontend (3-컬럼 JARVIS)
```
┌──────────────────────────────────────────────────────┐
│ HEADER: her 로고 + 현재 세션 제목 + 작은 thinking pulse │
├──────────────┬──────────────────────┬────────────────┤
│ LEFT (240px) │ CENTER (flex)        │ RIGHT (320px)  │
│ Sessions     │ Chat (msgs + input)  │ Memory probe   │
└──────────────┴──────────────────────┴────────────────┘
```

`apps/web/web/`:
- `index.html` — 3-컬럼 grid, 다크 톤
- `styles.css` — 짙은 네이비/사이안 글로우, 모노스페이스 라벨
- `state.js` — 세션 ID 등 가벼운 pub/sub
- `chat.js` — WS 연결, 토큰 append, autoscroll. Enter 전송 / Shift+Enter 줄바꿈
- `sessions.js` — 좌측 패널, 새 세션 / 클릭 시 이전 메시지 로드
- `memory.js` — 우측 패널: 이번 답변 recall 카드 + 최근 메모 + 사람

빌드 단계 없음. `apps/web` 가 정적 디렉토리를 직접 mount.

---

## 7. 단계 (반드시 순서대로)

1. **v0 — Web chat MVP** (현재): 3-컬럼 UI + WS 채팅 + Memory v2 + Consolidator. 통합 회귀 테스트 통과.
2. **v1 — Memory 패널 강화**: 우측 패널에 사실/메모 인라인 편집, 첨부 입력 흐름.
3. **v2 — Consolidator 자동화**: 외부 스케줄러로 03:00 호출 가이드 + 통합 결과 채팅창에 알림 카드.
4. **v3 — 임베딩 기반 notes 검색**: notes 에 vec0 인덱스 추가, recall §5.4 #4 LIKE → KNN 으로 승격.

> 각 단계 종료 시 **회귀 테스트 통과 + 사용자 데모** 후 다음 단계.

### 7.2 테스트
- Memory Layer 모든 연산: **단위 테스트 필수**.
- LLM 호출: 녹음된 대화 fixture 로 회귀 테스트 (현재 미구현; 텍스트 채널만 있으므로 추후 도입).
- WS 프로토콜: TestClient + 가짜 agent 로 토큰 스트림 / 에러 / 세션 라우팅 검증.

### 7.3 커밋 규칙
- Conventional Commits: `feat / fix / refactor / docs / test / chore`.
- 영향 영역 명시: `feat(memory): add note dedup logic`.
- 한 커밋 한 의도.

---

## 8. 디렉토리 구조

```
project/
├── CLAUDE.md
├── pyproject.toml
├── .envrc                 # direnv (비밀키)
├── .envrc.example
├── Makefile
├── apps/
│   ├── web/               # FastAPI + WebSocket + 정적 프론트엔드 (단일 사용자 표면)
│   │   ├── server.py
│   │   ├── eventbus.py    # 인-프로세스 pub/sub (상태 이벤트)
│   │   └── web/           # index.html, styles.css, *.js
│   ├── agent/             # LLM (Gemini) + 도구 + 익명화 + 회상
│   ├── memory/            # 메모리 v2 (sessions/messages/notes/...)
│   ├── consolidator/      # 일일 배치
│   └── tools/             # Calendar, Weather, …
├── data/
│   ├── db.sqlite          # 영구 메모리 (백업 대상)
│   ├── attachments/       # 텍스트 채널 첨부 원본 (session 별)
│   └── consolidation_log/
├── tests/
│   ├── unit/
│   └── fixtures/
└── scripts/
    ├── smoke_llm.py
    └── smoke_embedding.py
```

---

## 9. 코딩 컨벤션

- Python: `ruff` + `black`, **type hint 필수**.
- 함수 < 40라인 권장, 클래스 < 200라인.
- 모든 외부 호출(LLM·API)은 **retry + timeout 명시**.
- 비밀키는 `.envrc` 에만. 코드·로그·커밋 메시지 어디에도 하드코딩 금지.
- 로깅: `structlog`, 가족 구성원 실명은 로그에 남기지 않음 (ID 로).

---

## 10. Claude Code 에게 — 작업 규칙

### ✅ 자동으로 해도 되는 것
- 새 모듈 생성, 단위 테스트 작성, 리팩토링.
- 표 안의 "대체안" 범위 내 의존성 변경.
- 프론트엔드 UI 개선, CSS 변경.
- 문서·주석 업데이트.

### ⚠️ 반드시 사용자에게 묻는 것
- 외부 API 신규 도입 (비용 발생).
- DB 스키마 변경 (마이그레이션 필요).
- **새 LLM 호출 지점 추가 또는 모델 변경** — Gemini 패밀리 내에서도 항상 후보 + 근거 제시 후 승인 (§3.1).
- 메모리 데이터 삭제·대량 변경.
- 웹 서버를 `127.0.0.1` 외 호스트에 바인딩.

### ⛔ 절대 하지 말 것
- `data/db.sqlite` 또는 `data/attachments/` 를 클라우드/외부로 업로드.
- 가족 실명을 외부 서비스에 식별 가능한 형태로 전송.
- **OpenAI / Anthropic / 기타 비-Gemini LLM 또는 임베딩 호출** (§3.1). 라이브러리 의존성도 추가 금지.
- 첨부파일 화이트리스트 외 확장자 자동 수락 (실행 가능 바이너리·아카이브 등).
- 테스트 없이 Memory Layer 변경.
- 음성/STT/TTS/Wake/마이크 관련 코드 신규 추가 (§1.3 비목표).
- 백그라운드 데몬 / launchd plist 신규 추가 (§1.3 비목표).

### 10.1 로컬 저장 정책
- 모든 메시지·메모·첨부는 사용자의 로컬 디스크에만 저장된다.
- 웹 UI 는 외부 인증·트래킹 없음. CDN 의존 없음 (정적 자원 모두 자체 호스팅).
- 사용자가 메시지를 명시적으로 삭제하면 `archive`(soft) 가 기본. 영구 삭제는 별도 명시적 액션이어야 한다 (현 v0 미구현).

---

## 11. 자주 쓰는 명령

```bash
make setup         # 초기 환경 설정 (uv sync, 디렉토리, .envrc 복사)
make web           # 웹 앱 시작 (http://127.0.0.1:8765)
make dev           # web 의 alias
make consolidate   # 메모리 통합 수동 실행
make test          # pytest 전부
make backup        # data/ 암호화 tar.gz
```

---

## 12. 결정 로그 (Decision Log)

> 큰 결정은 여기에 누적한다. 왜 그렇게 정했는지 미래의 나/Claude 가 알 수 있게.

- _2026-04-30_: 초기 스택 결정 (이 문서 v0.1).
- _2026-04-30_: **언어 정책** — 사용자 입력은 한국어, 에이전트의 사용자 대상 발화도 한국어 기본. 단, **Claude Code 의 개발자 대상 응답은 영어**(토큰 절감, §0.1).
- _2026-04-30_: **저장소** — SQLite + sqlite-vec 채택. 가정/개인 단위 로컬 앱이라 별도 DB 프로세스 불필요, 백업이 단일 파일 복사로 끝남.
- _2026-04-30_: **OS 1순위 = macOS**.
- _2026-04-30_: **LLM = Gemini 3.1 Pro Preview**. 사용자 키 보유 + 한국어 + 멀티모달.
- _2026-04-30_: **LLM 패밀리 = Gemini 전용 (Hard Rule)**. 타 제공자 LLM·임베딩 사용 금지.
- _2026-04-30_: **임베딩 = `gemini-embedding-001` @ 768d, `task_type` 비대칭** (DOCUMENT 쓰기 / QUERY 검색).
- _2026-04-30_: **Consolidator/배치 LLM = `gemini-2.5-flash`**.
- _2026-05-02_: **음성 인터페이스 폐기, 텍스트 챗봇 단일 표면으로 전환** (대규모 피벗). 동기: 일상 사용 효율 — 음성 STT/TTS 의 지연·인식 오류·핸즈프리 제약이 가족용 비서의 일상 사용에 비효율. 텍스트 웹 UI 가 첨부·검토·재읽기에 우월. 영향:
  - 삭제: `apps/channels/{voice,text}/`, `apps/daemon/`, `apps/proactive/`, `apps/settings/consent.py`, `bin/her`, `infra/launchd/`, voice/seed/replay 스크립트, 관련 테스트, 음성 의존성 (sounddevice / numpy / silero-vad / faster-whisper).
  - 추가: `apps/web/` (FastAPI + WS + 3-컬럼 JARVIS 풍 프론트엔드, 단일 사용자 표면).
  - 메모리 v2: episodes → sessions + messages + notes. v1 DB 자동 백업 후 v2 새로 생성. facts.source_episode_id → source_session_id, attachments.episode_id → session_id, vec_episodes → vec_sessions, 신규 vec_messages.
  - AgentCore 시그니처: `episode_id` → `session_id`, `channel` 파라미터 제거. AgentResponse 도 동일.
  - 설정: voice 키 (mic_consent_*, quiet_mode, wake_keyword, daily_proactive_limit, silence_threshold_hours, stt_model, tts, echo_gate_ms) 모두 제거, web_host/web_port 추가, schema_version 8.
  - Phase 0~4 로드맵 → v0~v3 (web chat MVP / 메모리 패널 / Consolidator 자동화 / notes 임베딩).
  - 무효화된 결정: macOS `say` TTS, Gemini TTS, 음성 파이프라인 Path B, Wake Word, Presence orb, launchd 데몬, Proactive 발화 엔진. 향후 음성 재도입은 신규 결정으로 시작.

---

**이 문서는 살아있다. 결정이 바뀌면 즉시 업데이트하라.**
