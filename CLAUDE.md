# CLAUDE.md

> 가족같이 곁에 있는 음성 AI 비서 — 사람·관계·일상을 기억하고 먼저 챙기는 동반자

---

## 0. TL;DR (Claude Code에게)

이 프로젝트는 **목소리 중심의 AI 비서**다. 두 축이 전부다:

1. **Voice-first**: 화면 없이도 완결되어야 한다. UI는 보조.
2. **Memory-first**: 모든 대화·관계·이벤트는 누적·연결되며, 시간이 갈수록 똑똑해진다.

이 두 축이 흔들리는 결정은 거부하라.

> **단, 텍스트 채팅도 1급 채널이다.** 음성은 주된 인터페이스지만, 첨부파일·긴 텍스트·URL·코드 처럼 음성으로 전달하기 어려운 정보는 텍스트 채널로 받는다. 자세한 건 §2.5.

### 0.1 커뮤니케이션 스타일 (Claude Code 응답 규칙)

- **사용자는 한국어로 명령한다. Claude 는 영어로 응답한다.** 토큰 소비를 최소화하기 위함.
- 단, 다음 경우엔 한국어로 응답한다:
  - 사용자가 명시적으로 한국어 응답을 요청한 경우.
  - **TTS 로 사용자에게 들려줄 자연어 문장**(에이전트의 발화 스크립트, 프롬프트 템플릿 등)을 작성하는 경우 — 이건 응답이 아니라 **산출물**이라 §2.1 이 우선한다.
- 응답은 **간결하게**. 불필요한 머리말·요약·확인 멘트 금지. 한 줄로 끝낼 수 있으면 한 줄.
- 코드/파일 경로/명령어 등 고유명사는 그대로 둔다 (번역하지 말 것).

---

## 1. 프로젝트 개요

### 1.1 무엇을 만드는가
영화 *Her* 의 사만다처럼, **사용자와 가족 구성원을 기억하고, 일정·이벤트·관심사를 챙겨주며, 먼저 제안하는 음성 AI 비서**. 화면이 아닌 **귀와 입**으로 살아간다.

### 1.2 핵심 사용 시나리오
- "어머니 생신이 다음 주야. 케이크 예약해줘"
  → 작년 어머니가 좋아하셨던 단호박 케이크를 기억해서 제안
- "오늘 저녁에 뭐 먹지?"
  → 가족 알러지·최근 식단·오늘 날씨까지 종합 추천
- (비서가 먼저) "내일 아내분 워크숍이라 일찍 일어나신다고 하셨어요. 알람 미리 맞춰둘까요?"

### 1.3 비목표 (Out of Scope)
- 일반 웹 검색·뉴스 요약 도구로 쓰는 것
- 풀 데스크톱/모바일 GUI 앱 (Tauri/Electron 등). 단, **음성 채널 보조의 시각적 프레즌스(§6.3 Presence Channel)** 는 본 프로젝트의 정체성에 포함된다 — 영화 *Her* 의 사만다처럼 사용자가 "거기 있는 누군가" 와 대화하는 분위기를 만드는 가벼운 로컬 웹 화면. 풀 GUI 앱은 아니다.
- 클라우드 다중 사용자 SaaS (가정/개인 단위 **로컬 앱**)

---

## 2. 핵심 설계 원칙

### 2.1 Voice-First
- **모든 기능은 음성 단독으로 시작·완료 가능해야 한다** (단, 음성으로 전달 불가능한 입력 — 첨부·긴 텍스트 — 은 §2.5 의 텍스트 채널로 보완).
- **출력 채널이 음성일 때**: 응답은 **TTS로 자연스럽게 들리는 텍스트**여야 한다. 마크다운·번호 매김·긴 리스트 금지.
- **출력 채널이 텍스트일 때**: 마크다운·코드블록·링크 사용 가능. 같은 Agent Core 가 채널별 렌더러로 분기 (§4).
- **응답 지연 SLA (음성)**: 첫 오디오 출력까지 평균 **< 2.0초**. 단순 TTFB 가 아니라 **streaming partial TTS** 로 달성한다 — Pro 가 첫 문장을 토큰 스트림으로 생성하는 즉시 TTS 합성·재생을 시작하고, 나머지 문장은 비동기로 이어붙인다. 초기 1.5초 목표는 §3.1 의 Pro 전용 정책(§12 결정) 으로 인해 비현실적이라 완화됨.
- 텍스트 응답은 별도 SLA 없음.

### 2.2 Memory as Living Tissue
메모리는 단순 로그가 아니라 **사람·시간·맥락의 그래프**다.
- **Person**(사람)이 중심 노드. 그 주변에 Facts·Events·Preferences가 매달림.
- 매일 자정 **메모리 통합(Consolidation)**: 단기 대화 → 장기 사실로 승격.
- 비유: 단기 대화는 "오늘 일기", 통합 후엔 "그 사람을 아는 지식"이 된다.

### 2.3 Local-First & Privacy
- 가족 정보는 외부로 나가지 않는 것이 기본.
- 모든 영구 저장은 **로컬 SQLite + 암호화 파일**.
- LLM 호출 시에는 **익명화된 컨텍스트만** 전송 (실명 → 별칭 매핑).
- **익명화 경계**: Memory Layer 는 항상 실명으로 동작. 익명화는 **Agent Core 의 LLM 어댑터 경계**에서만 수행한다. 매핑 테이블은 메모리 내에서만 유지하고, 응답 수신 후 별칭 → 실명 으로 역매핑한다. 이 경계가 흐려지면 PR 거부.

### 2.4 Proactive but not Pushy
- 비서는 **먼저 말을 걸어야** 한다. 조용히 있으면 가구다.
- 단, 사용자가 "조용 모드"를 선언하면 즉시 침묵 (다음 호출까지).
- Proactive 발화는 **하루 N회 이하** (설정 가능, 기본 3회).

**Proactive 발화 트리거** (Phase 4 구현 시 이 목록에서 시작):
1. **시간 기반**: events 테이블의 임박 일정 (24h / 1h 전).
2. **메모리 패턴**: Consolidator 가 발견한 반복 패턴 (예: "매주 금요일 저녁엔 외식") 의 직전 시점.
3. **외부 이벤트**: 날씨 급변, 도구가 보고하는 새 정보(메일·캘린더 변경 등).
4. **침묵 회피**: 일정 시간 이상 상호작용 없을 때 가벼운 안부 (하한선 설정 필요).

각 트리거는 우선순위·일일 카운터·조용 모드를 통과해야 발화한다. 통과 로직은 Agent Core 가 단일 큐로 관리.

### 2.5 Multi-Channel I/O (음성 + 텍스트)

- 사용자는 **음성 채널** 또는 **텍스트 채널** 중 어느 쪽으로도 비서와 대화할 수 있다. 두 채널은 **같은 Agent Core 와 같은 Memory** 에 연결된다 — 음성으로 한 말과 텍스트로 친 말이 한 대화로 이어진다.
- **채널 선택 기준**:
  - 음성: 일상적 대화, 짧은 질문/명령, 핸즈프리 상황 (운전·요리·이동).
  - 텍스트: **첨부파일** (사진·PDF·문서), 긴 텍스트 (이메일 초안 검토 등), URL/코드/구조화된 데이터, 조용해야 하는 환경.
- **첨부파일은 텍스트 채널에서만** 받는다. 종류: 이미지, PDF, 일반 텍스트/마크다운, 캘린더 ics, 메일 eml. (지원 형식은 Phase 1 까지 확정.)
  - 첨부는 처리 후 관련 사실/이벤트로 메모리에 승격될 수 있다 (출처 = 해당 episode).
  - 원본 첨부는 `data/attachments/<episode_id>/` 에 보관, DB 에는 경로 + 메타만.
- **세션 연속성**: 사용자가 음성으로 시작한 대화를 텍스트로 이어가거나 그 반대도 가능. 동일 `episode_id` 아래 메시지가 채널 태그(`channel: voice|text`) 와 함께 누적된다.
- **응답 채널 결정**: 기본은 *입력 채널 = 출력 채널*. 단, 사용자가 명시적으로 다른 채널을 요청하거나 출력이 음성에 부적합(긴 리스트·코드·링크) 할 땐 텍스트로 폴백 + 음성으로 "텍스트로 보냈어요" 한 줄 안내.
- **Proactive 발화 (§2.4)** 는 사용자가 마지막으로 활성이었던 채널로 전달. 둘 다 비활성이면 음성 우선, 조용 모드면 텍스트로 큐잉.

---

## 3. 기술 스택 (기본 선택)

### 3.1 LLM 정책 — Gemini 전용 (Hard Rule)

- **모든 LLM·임베딩 호출은 Gemini 패밀리 모델만 사용한다.** OpenAI / Anthropic / 기타 LLM 제공자 금지.
- **모델 선택은 태스크별 최적화**: 단일 모델로 통일하지 않는다. 추론·임베딩·요약·실시간 음성 등 각 태스크에 가장 적합한 Gemini 모델을 Claude Code 가 **제안**하고 **사용자 승인 후** 확정한다.
- **승인 워크플로**:
  1. Claude Code 가 새 LLM 호출 지점을 추가하거나 기존 모델을 바꾸기 전, **후보 모델 + 근거**(품질·지연·비용·컨텍스트 길이) 를 제시한다.
  2. 사용자가 승인하면 §3.2 표에 기록 + §12 결정 로그에 추가.
  3. 미승인 상태로 새 모델 ID 를 코드/설정에 박지 않는다.
- 비-LLM 요소(STT 로컬 모델·TTS·Wake Word 등) 는 이 규칙의 대상 아님 — §3.2 표대로 진행.

### 3.2 스택 표

| 영역 | 선택 | 이유 / 대체안 |
|---|---|---|
| 언어/런타임 | Python 3.12 + asyncio | 음성 비동기·생태계 / Node+TS 가능 |
| 백엔드 | FastAPI + WebSocket | 실시간 양방향 스트리밍 |
| LLM (추론·메인) | Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`) | 사용자 확정. 한국어·멀티모달·도구 사용 강함 |
| LLM (배치·요약·Consolidator) | `gemini-2.5-flash` | 야간 배치(매일 03:00) — Pro 대비 ~10-20× 저렴. 추출 품질은 §5.3 의 confidence 임계로 흡수 |
| 임베딩 | `gemini-embedding-001`, **768d**, `task_type` 비대칭 사용 | 한국어·MRL 지원 (768→1536→3072 무재학습 확장). DOCUMENT/QUERY 분리로 회상 품질↑ |
| 음성 파이프 (인입·인출) | **분리 파이프**: STT → `gemini-3.1-pro-preview` → TTS | 추론 품질 우선(§12). Live API 는 미래 검토 — Pro 급 Live 모델 등장 또는 지연이 사용자 인내 한계를 넘을 때 |
| STT (백업·로컬) | faster-whisper (로컬) | 오프라인 / Deepgram Nova-3 대체 |
| TTS (1차) | macOS `say -v "Jian (Premium)"` | 로컬·오프라인·무한 호출. Premium 보이스가 일상 대화 자연스러움 충분. §2.3 Local-First 정합. 첫 오디오 ~200-500ms (Gemini 1-2s 보다 빠름). `tts_voice` 설정으로 변경 가능 (Yuna Premium 등). |
| TTS (선택, 클라우드) | `gemini-2.5-flash-preview-tts` (또는 `-pro-preview-tts`) | 더 표현력 풍부한 보이스가 필요할 때 — 30+ 사전 정의 보이스, 톤 지시 가능. `settings.toml` 의 `tts_provider="gemini"` 로 전환. **단**: 무료 티어 일일 100회 한도 — 가족 단위 사용엔 부족할 수 있음. |
| 구조 메모리 | SQLite + sqlite-vec | 로컬·임베디드·벡터 동시 |
| 패키지 매니저 | uv | 빠름·결정적 |
| 환경 변수 | direnv (`.envrc`) | 비밀키 로컬 격리 (`GEMINI_API_KEY` 만 필수) |
| 백그라운드 실행 | launchd (macOS) | 상시 데몬 |
| Wake Word | VAD + faster-whisper 폴링 (재사용) | Picovoice 키 불필요·온디바이스 한국어 그대로·새 의존성 0. 깨우기 단어는 settings.wake_keyword 에 자연어로 지정. |

> **변경 권한**: 표의 "선택" 컬럼 변경은 **사용자 승인 필요**. "대체안" 범위 내 변경은 자율.

**임베딩 운영 규칙** (sqlite-vec 사용 시 강제):
- 한 인덱스 안에서 **차원·모델 ID 혼합 금지**. 차원 변경은 마이그레이션으로 처리.
- 매 벡터 행에 `model_id`, `dim`, `task_type` 메타 컬럼 함께 저장 (미래 모델 교체 시 모호성 제거).
- 쓰기: `task_type=RETRIEVAL_DOCUMENT`. 검색 쿼리: `task_type=RETRIEVAL_QUERY`. 동일 인덱스 내에서 비대칭 사용.

---

## 4. 아키텍처

```
 [Mic]──STT──▶ Voice Channel ──┐
                               │
 [Keyboard/Files] ─▶ Text Channel ──┐
 (text + attachments)               │
                                    ▼
                              Agent Core ◀──▶ Tool Registry
                                ▲    │        (Calendar, Mail, …)
                                │    ▼
                            Memory Layer
                                ▲
                                │
                           Consolidator   (매일 03:00 배치)
                                    │
                                    ▼
 [Speaker]◀─TTS── Voice Channel ◀──┤
 [Screen/Stdout]◀── Text Channel ◀──┘
```

### 5개 핵심 모듈
1. **Voice Channel** (`apps/channels/voice/`): 마이크/스피커 ↔ STT/TTS 추상화, 인터럽션 처리, Wake/VAD.
2. **Text Channel** (`apps/channels/text/`): CLI/TUI/WebSocket 텍스트 입출력, **첨부파일 수신·검증·저장**, 마크다운 렌더.
3. **Agent Core**: LLM 추론(§3 의 Gemini) + 도구 호출 라우팅 + 대화 상태 + 채널별 출력 렌더러 분기.
4. **Memory Layer**: 사람·이벤트·사실 그래프 + 벡터 검색 + 첨부 메타.
5. **Consolidator**: 단기→장기 메모리 정리, 패턴 발견, 사전 제안 생성.

각 모듈은 명확한 인터페이스로 분리. 모듈 경계를 가로지르는 의존성은 PR에서 거부.

**채널 ↔ Agent Core 메시지 포맷** (공통 envelope):
```
{ episode_id, channel: "voice"|"text", role: "user"|"assistant",
  content: [...parts...],   # text / audio_ref / attachment_ref
  ts }
```
Agent Core 는 채널을 모른다 — envelope 만 본다. 채널은 자기 모달리티만 안다.

---

## 5. Memory System (이 프로젝트의 심장)

### 5.1 3단 계층

| 계층 | 보관 | 형태 | 비유 |
|---|---|---|---|
| **Working** | 현재 대화 | 메시지 배열 | 지금 떠올린 것 |
| **Episodic** | 30일 | 대화 요약 + 임베딩 | 일기 |
| **Semantic** | 영구 | 구조화된 사실 | 그 사람에 대한 앎 |

### 5.2 핵심 엔티티 (SQLite 스키마, 발췌)

```sql
people(
  id, name, relation, birthday,
  preferences_json, created_at, updated_at
)
events(
  id, person_id, type, title, when_at,
  recurrence, source, status
)
facts(
  id, subject_person_id, predicate, object,
  confidence, source_episode_id, valid_from, archived_at
)
episodes(
  id, when_at, summary, transcript_compressed, embedding,
  primary_channel  -- 'voice' | 'text' | 'mixed'
)
preferences(
  person_id, domain, value, last_seen_at
)
attachments(
  id, episode_id, sha256, mime, ext, byte_size,
  path, description, ingested_at
)
```

### 5.3 통합(Consolidation) 규칙
- 매일 **03:00** 실행.
- 직전 24시간 episode → LLM이 fact 후보 추출.
- `confidence ≥ 0.7` 만 semantic으로 승격.
- 기존 fact와 모순 시 → 최신·출처 신뢰도 우선, 이전은 `archived_at` 기록 (삭제 X).
- 결과 로그는 `data/consolidation_log/` 에 보관.

### 5.4 회상(Recall) 전략
질문 들어오면 3단계 병렬 조회:
1. **Structured 검색**: 사람 이름·관계·날짜 키워드 → SQL.
2. **Semantic 검색**: 임베딩 유사도 top-k.
3. **Recency 가중**: 최근 7일 episode는 가중치 ×1.5.

---

## 6. 채널 세부

### 6.1 Voice Channel
**파이프라인 (분리, Path B)**:
```
Mic ─▶ VAD ─▶ STT ─▶ Agent (gemini-3.1-pro-preview, streaming) ─▶ TTS ─▶ Speaker
                                          │
                                          └─▶ 첫 문장 토큰이 도착하는 즉시 TTS 합성·재생 시작
                                              (streaming partial TTS, §2.1)
```

- **Wake Word**: VAD + faster-whisper 폴링 (로컬, 별도 키 없음). 깨우기 단어는 `settings.wake_keyword` 에 자연어 한 줄("자기야", "비서야" 등)로 지정. Whisper 가 한국어를 그대로 인식하므로 별도 학습 불필요.
- **VAD**: silero-vad 로 발화 종료 감지 (Phase 1 기준). Live API 도입 시점에 server-side VAD 로 전환 검토.
- **STT**: 1차 후보는 faster-whisper(로컬). Gemini STT 가 한국어 품질·지연에서 우월하면 Phase 1 진입 전 비교 후 선택 — 사용자 승인 필요(§3.1, LLM 정책과 같은 절차 따름).
- **TTS**: macOS `say -v "Jian (Premium)"` (1차, 로컬). Pro 의 token stream 을 문장 경계에서 분할해 `say` 호출 단위로 흘려보낸다. 24kHz PCM 출력 (AIFF → afconvert 파이프). Premium 보이스 (Jian/Yuna 등) 는 시스템 설정 → 손쉬운 사용 → 보이는 음성 → 보이스 관리 에서 다운로드. `settings.tts_voice` 로 음성 변경. 더 표현력 있는 보이스가 필요하면 `tts_provider="gemini"` 로 전환 (단 일일 한도 있음).
- **인터럽션**: 사용자가 비서 말 끊으면 **300ms 내 TTS 중단** + Pro 호출도 abort.
- **백채널**: 긴 응답 전 "음…" "잠시만요" 같은 신호 — Pro 응답이 첫 토큰까지 1.2s 이상 지연될 때만 발화 (불필요 남발 금지).
- **언어**: 한국어 기본, 사용자가 영어로 말하면 영어로 응답.
- **STT 후처리**: 가족 구성원 이름·고유명사는 메모리에서 끌어와 발음 사전(custom vocab) 으로 주입 (오인식 감소).
- **품질 동등성 보장**: 텍스트 채널과 음성 채널은 **같은 모델·같은 시스템 프롬프트** 를 공유한다. 채널별 차이는 입출력 렌더러(§4) 에서만 발생.

### 6.2 Text Channel
- **표면(Surface)**: Phase 0 = stdin/stdout REPL. Phase 1+ = WebSocket 기반 로컬 웹 UI / TUI 옵션 (확정은 Phase 1 진입 시).
- **첨부 입력**: 파일 경로 드래그·붙여넣기 / 명시적 `/attach <path>` 명령 / 웹 UI 의 첨부 버튼.
- **첨부 검증**: 화이트리스트 확장자 (`.png .jpg .jpeg .pdf .txt .md .ics .eml`), 크기 상한 (기본 25MB), MIME 재확인.
- **첨부 저장**: `data/attachments/<episode_id>/<sha256>.<ext>` (중복 제거). DB `attachments` 테이블에 메타.
- **렌더링**: 응답에 코드·표·링크 사용 가능. 단, 같은 응답이 음성으로도 나갈 수 있음을 가정해 **핵심 정보는 첫 1–2 문장에** 자연어로 요약 후 상세를 마크다운으로.
- **전사 미러링** (선택, 기본 ON): 음성 채널에서 일어난 대화의 전사를 텍스트 채널에도 표시 — 다른 채널로 전환할 때 맥락 손실 방지.

### 6.3 Presence Channel (Phase 3.5+)

**목적**: 음성 대화의 시각적 동반자. 영화 *Her* 의 사만다 화면처럼, 사용자가 "거기 있는 누군가" 와 대화하는 미적 정체성. 풀 GUI 앱이 아니라 **음성을 보조하는 가벼운 로컬 화면** (§1.3).

- **표면(Surface)**: 로컬 FastAPI + WebSocket 서버 (`localhost:8765`), 정적 프론트엔드 (HTML+JS+WebGL/Three.js). 빌드 단계 없음 — `make presence` 로 즉시 실행.
- **시각 레퍼런스 (확정)**: *Her* (Samantha) — **추상 오브** (그라데이션·입자·빛). 의인화된 캐릭터·아바타·HAL 단일 눈 등은 채택하지 않음 (CLAUDE.md §12 2026-05-01).
- **상태 표현**:
  - `idle` — 부드러운 호흡 펄스
  - `listening` — 사용자 음성 진폭에 맞춰 잔물결
  - `thinking` — 느린 와류, 색상 차분
  - `speaking` — 따뜻한 펄스, TTS 청크와 동기화
  - `quiet` — 흐리게, 회색조
  - `wake` — 짧은 플래시 (깨우기 단어 감지 순간)
- **자막**: 사용자 STT 부분 결과 + 에이전트 응답 청크가 영화 자막처럼 페이드 인/아웃. 5초 후 자동 사라짐. 마크다운은 음성 채널 출력과 동일한 톤(평문) — 긴 코드/표는 텍스트 채널로 폴백.
- **이벤트 버스**: 모든 채널·에이전트가 in-process pub-sub (`apps/presence/eventbus.py`) 으로 상태를 publish. Presence 서버가 subscribe → WebSocket 으로 브라우저에 push. 다른 채널의 동작에는 영향 없음 (관찰자 패턴).
- **외부 노출 금지**: 서버는 `127.0.0.1` 만 바인드. LAN/외부 노출은 비목표 (§1.3 클라우드 SaaS 비목표 정신).
- **데이터**: 화면은 로그·DB 를 직접 읽지 않음 — 이벤트 버스 메시지만 소비. 가족 실명은 별칭으로 표시할지 실명으로 표시할지 사용자 선택 (Phase 3.5 v1 기본: **실명 표시 — 본인이 보는 화면이므로 §2.3 익명화 경계 밖**).

### 7.0 어디서 시작하나 (현재 리포 상태)

**현재 리포는 설계 문서(이 파일)만 존재한다. 코드·`pyproject.toml`·`Makefile`·`apps/` 트리 모두 없다.**

빈 리포에서 시작하는 다음 세션은 다음 순서를 지킨다:

1. `pyproject.toml` (uv 기준) + `.envrc` 템플릿 (`GEMINI_API_KEY` 포함).
2. `apps/memory/schema.sql` + `apps/memory/store.py` (CRUD, attachments 테이블 포함).
3. `apps/agent/` 의 Gemini 어댑터 (`gemini-3.1-pro-preview` 호출 + 회상 통합).
4. `apps/channels/text/` 의 REPL (stdin/stdout, `/attach <path>` 명령).
5. `Makefile` 의 `setup` / `dev` / `consolidate` 타겟.
6. `tests/unit/` 의 Memory + Text Channel 테스트.

**음성 코드(STT/TTS/Wake/VAD) 를 먼저 만들면 Phase 0 위반.** 거부할 것.

### 7.1 단계 (반드시 순서대로)
1. **Phase 0** — **Text Channel** 단독 + Agent + Memory. 음성 코드 0줄. (텍스트 채널은 여기서 영구 채널로 자리잡고, 이후 단계에서도 유지된다.)
2. **Phase 1** — Voice Channel 추가: STT/TTS 붙이고 단일턴 음성. 텍스트 채널과 같은 Agent 공유.
3. **Phase 2** — 인터럽션·실시간성. 첨부파일 처리 정식 통합.
4. **Phase 3** — 백그라운드 상시 + Wake Word.
5. **Phase 3.5** — Presence Channel (시각적 동반자, §6.3).
6. **Phase 4** — Proactive 제안 (Consolidator 결과 활용).

> 각 Phase 종료 시 **회귀 테스트 통과 + 사용자 데모** 후 다음 단계.

**Phase 0 종료 조건 (Definition of Done)**:
- [ ] `apps/memory/` 에 people / events / facts / episodes / preferences / attachments 스키마 + CRUD 구현.
- [ ] `apps/agent/` 가 Gemini 3.1 Pro Preview 호출 + §5.4 회상 전략 통합.
- [ ] `apps/channels/text/` REPL 에서 한국어 대화 + 이전 세션 사실 회상 + 기본 첨부파일(`.txt .md .pdf` 최소) 수신·메모리 연결 동작.
- [ ] `scripts/seed_family.py` 로 초기 가족 데이터 입력 가능.
- [ ] Memory Layer 단위 테스트 통과 (`make test`).
- [ ] `scripts/replay.py` 가 최소 1개 fixture 대화를 재생 가능.

### 7.2 테스트
- Memory Layer 모든 연산: **단위 테스트 필수**.
- LLM 호출: 녹음된 대화 `.jsonl` 로 **회귀 테스트** (`scripts/replay.py`).
- 음성 품질: 사람이 듣는 것이 최종 검증.

### 7.3 커밋 규칙
- Conventional Commits: `feat / fix / refactor / docs / test / chore`.
- 영향 영역 명시: `feat(memory): add fact dedup logic`.
- 한 커밋 한 의도.

---

## 8. 디렉토리 구조

```
project/
├── CLAUDE.md
├── pyproject.toml
├── .envrc                 # direnv (비밀키)
├── Makefile
├── apps/
│   ├── channels/
│   │   ├── voice/         # STT / TTS / Wake / VAD (Phase 1+)
│   │   └── text/          # REPL/TUI/WS, 첨부 처리 (Phase 0)
│   ├── agent/             # LLM(Gemini) + 도구 라우팅 + 채널별 렌더러
│   ├── memory/            # 메모리 3계층 전체 + 첨부 메타
│   ├── consolidator/      # 일일 배치
│   └── tools/             # Calendar, Reminder, Mail …
├── data/
│   ├── db.sqlite          # 영구 메모리 (백업 대상)
│   ├── attachments/       # 텍스트 채널로 받은 원본 (episode_id 별)
│   ├── audio_logs/        # 디버그용 음성 (옵션)
│   └── consolidation_log/
├── tests/
│   ├── unit/
│   └── fixtures/          # 회귀 테스트 대화
└── scripts/
    ├── replay.py          # 녹음된 대화 재생
    └── seed_family.py     # 초기 가족 데이터 입력
```

---

## 9. 코딩 컨벤션

- Python: `ruff` + `black`, **type hint 필수**.
- 함수 < 40라인 권장, 클래스 < 200라인.
- 모든 외부 호출(LLM·STT·TTS·API)은 **retry + timeout 명시**.
- 비밀키는 `.envrc` 에만. 코드·로그·커밋 메시지 어디에도 하드코딩 금지.
- 로깅: `structlog`, 가족 구성원 실명은 로그에 남기지 않음 (ID로).

---

## 10. Claude Code에게 — 작업 규칙

### ✅ 자동으로 해도 되는 것
- 새 모듈 생성, 단위 테스트 작성, 리팩토링.
- 표 안의 "대체안" 범위 내 의존성 변경.
- 문서·주석 업데이트.

### ⚠️ 반드시 사용자에게 묻는 것
- 외부 API 신규 도입 (비용 발생).
- DB 스키마 변경 (마이그레이션 필요).
- **새 LLM 호출 지점 추가 또는 모델 변경** — Gemini 패밀리 내에서도 항상 후보 + 근거 제시 후 승인 (§3.1).
- 음성 핵심 컴포넌트(STT/TTS/Wake) 교체.
- 메모리 데이터 삭제·대량 변경.

### ⛔ 절대 하지 말 것
- `data/db.sqlite` 또는 `data/attachments/` 를 클라우드/외부로 업로드.
- 가족 실명을 외부 서비스에 식별 가능한 형태로 전송.
- **OpenAI / Anthropic / 기타 비-Gemini LLM 또는 임베딩 호출** (§3.1). 라이브러리 의존성도 추가 금지.
- **출력 채널이 음성**일 때 마크다운 응답 생성 (TTS 가 못 읽음). 텍스트 채널은 마크다운 OK.
- 첨부파일 화이트리스트 외 확장자 자동 수락 (실행 가능 바이너리·아카이브 등).
- 테스트 없이 Memory Layer 변경.
- 사용자 동의 없이 마이크 상시 녹음.

---

## 11. 자주 쓰는 명령

> ⚠️ 아래 명령은 **Phase 0 셋업이 완료된 이후부터** 동작한다. 리포가 빈 상태(`Makefile` 부재) 라면 §7.0 부터 진행할 것.

```bash
make setup        # 초기 환경 설정 (uv sync, db init)
make first-run    # 첫 실행 점검: 마이크 동의 + 환경변수 확인 (Phase 3+)
make text         # 텍스트 채널 REPL (Phase 0 의 기본 진입점)
make voice        # 음성 채널 (Phase 1+)
make dev          # 두 채널 동시 기동
make consolidate  # 메모리 통합 수동 실행
make replay FILE=tests/fixtures/dialog_001.jsonl
make backup       # data/ 암호화 백업
```

---

## 12. 결정 로그 (Decision Log)

> 큰 결정은 여기에 누적한다. 왜 그렇게 정했는지 미래의 나/Claude가 알 수 있게.

- _2026-04-30_: 초기 스택 결정 (이 문서 v0.1).
- _2026-04-30_: **언어 정책** — 사용자 입력은 한국어, 에이전트의 사용자 대상 발화도 한국어 기본. 사용자가 영어로 말하면 영어로 응답. 단, **Claude Code 의 개발자 대상 응답은 영어**(토큰 절감, §0.1).
- _2026-04-30_: **저장소** — SQLite + sqlite-vec 채택. Postgres+pgvector 가 아닌 이유: 가정/개인 단위 로컬 앱 (§1.3) 이라 별도 DB 프로세스 불필요, 백업이 단일 파일 복사로 끝남.
- _2026-04-30_: **OS 1순위 = macOS** — launchd 우선. Linux/Windows 는 Phase 3 이후 검토 (사용자 환경이 macOS 전제).
- _2026-04-30_: **LLM = Gemini 3.1 Pro Preview** (`gemini-3.1-pro-preview`). 사용자가 Gemini API 키 보유, 한국어 + 멀티모달(첨부파일 처리) 강함. Claude Sonnet 4.5 에서 변경.
- _2026-04-30_: **LLM 패밀리 = Gemini 전용 (Hard Rule)**. OpenAI/Anthropic 등 타 제공자 LLM·임베딩 사용 금지. 임베딩과 배치/요약용 모델은 Gemini 패밀리 내에서 **태스크별 최적 모델을 Claude 가 제안 → 사용자 승인 후 확정** (§3.1). 이로써 이전에 잡혀있던 OpenAI text-embedding-3-small / OpenAI Realtime API 결정은 **무효화**되며 Gemini 후보로 재선정 대기 (TBD).
- _2026-04-30_: **텍스트 채널 1급화** — 음성 단독이 아닌 음성+텍스트 듀얼 채널. 동기: 첨부파일·긴 텍스트·URL 처럼 음성으로 전달 어려운 정보를 받기 위함. Voice-first 원칙은 유지(§2.1) 하되, 텍스트 채널이 Phase 0 의 영구 채널로 자리잡는다(§2.5, §6.2).
- _2026-04-30_: **임베딩 = `gemini-embedding-001` @ 768d, `task_type` 비대칭** (DOCUMENT 쓰기 / QUERY 검색). 이유: 한국어 품질·MRL(무재학습 차원 확장) 지원·비대칭 검색이 §5.4 회상 품질에 직접 기여. 768d 부터 시작 — sqlite-vec 저장 ~3KB/episode, 가족 규모(<100k episodes) 에서 brute-force 충분. 회상 품질 미달 시 1536d 로 확장(마이그레이션).
- _2026-04-30_: **Consolidator/배치 LLM = `gemini-2.5-flash`**. 이유: 야간 배치(§5.3) 는 구조화 추출 — Pro 의 추론 헤드룸이 낭비. 비용/지연 ~10-20× 우위. 추출 노이즈는 confidence ≥ 0.7 임계와 archive-not-delete 정책으로 흡수. 품질 미달 신호는 Phase 4 사전 제안의 부자연스러움 — 그땐 프롬프트부터 의심.
- _2026-04-30_: **음성 파이프라인 = Path B (분리, Pro 직결)**. STT → `gemini-3.1-pro-preview` → TTS. 이유: 사용자 다이얼로그 품질이 제품의 본질이며, 텍스트 채널과 동일한 두뇌를 보장해야 한다 (사용자 의견). Live API(Flash 급) 채택 시 추론 천장이 낮아지는 트레이드 거부. 대가로 §2.1 의 음성 SLA 를 1.5s → ~2.0s 로 완화하고 streaming partial TTS 로 체감 지연 흡수. Live API 는 Pro 급 변형 등장 또는 지연 인내 한계 초과 시 재검토.
- _2026-05-01_: **TTS = `gemini-2.5-flash-preview-tts` + macOS `say -v Yuna` 폴백**. ElevenLabs Multilingual v2 결정을 무효화. 이유: ① §3.1 Gemini 단일 벤더 정합성 (별도 키·계정 불필요), ② AI Studio 무료 티어로 개발 단계 비용 0, ③ 24kHz PCM 스트리밍이 §2.1 streaming partial TTS 와 호환. 품질 미달 시 `gemini-2.5-pro-preview-tts` 승격 — 별도 승인 절차(§3.1) 따름. 오프라인 안전망으로 macOS `say -v Yuna` 항상 유지 — 네트워크/쿼터 장애 시 폴백.
- _2026-05-01_: **TTS 1차 = macOS `say -v "Jian (Premium)"` 로 변경. Gemini TTS 는 선택**. 이전 5/1 결정 일부 수정. 이유: ① **무료 티어 일일 한도 (100 req/day)** — 가족 단위 사용에서 한 번 대화로 금방 소진, ② Gemini TTS preview 가 한국어 짧은 입력을 종종 chat 으로 오해 (400 INVALID_ARGUMENT) 하거나 빈 응답 (`'NoneType'.parts`) 반환 — 폴백 보이스 (Yuna) 가 자주 끼어들어 한 응답 안에서 두 보이스 섞임, ③ macOS Premium 보이스 (`Jian`/`Yuna` Premium) 는 일상 대화 자연스러움 충분, ④ 첫 오디오 200-500ms — Gemini API 1-2s 보다 빠름, §2.1 SLA 더 유리, ⑤ §2.3 Local-First 정합 (오디오 합성도 로컬). 토글: `~/.her/settings.toml` 의 `tts_provider` ("say" 기본 / "gemini" 선택).
- _2026-05-01_: **사용자 설정 = `~/.her/settings.toml` (TOML, stdlib only)**. 이유: 가정 단위 로컬 앱(§1.3) 에 별도 설정 백엔드 불필요. 단순 평면 구조 → tomllib 읽기 + 수기 직렬화 쓰기. 동의 상태(§10), 깨우기 키워드, 조용 모드 기본값을 한 곳에서 관리. her CLI(데몬-eng) 와 모든 채널이 같은 파일을 본다.
- _2026-05-01_: **Wake Word = VAD + faster-whisper 폴링 (Porcupine 결정 무효화)**. 이유: ① 새 API 키(`PICOVOICE_ACCESS_KEY`) 불필요·신규 가입 절차 제거, ② Porcupine 의 한국어 사전학습 부재로 사용자가 별도 .ppn 학습해야 하던 마찰 제거 — Whisper 가 한국어를 그대로 듣기 때문에 깨우기 단어를 `settings.wake_keyword` 에 자연어 한 줄("자기야", "비서야" 등)로 적으면 끝, ③ §2.3 Local-First 정합 (모든 깨우기 판단이 로컬, 외부 호출 없음), ④ Phase 1 부터 이미 깔린 faster-whisper + silero-vad 재사용으로 신규 의존성 0. 트레이드: 깨우기 응답 ~500ms-1s (Porcupine ~300ms) 와 발화당 Whisper 추론 1회 — Apple Silicon 에서 무시 가능. 한국어 자연어 매칭 정확도가 부족하면 fuzzy 매칭/키워드 다중화로 보완.
- _2026-05-01_: **Presence Channel = *Her* (Samantha) 추상 오브 — Phase 3.5 신설**. 사용자 의견: "음성이 메인이지만 화면에 나랑 대화하는 존재가 미래 지향적 영화처럼 그려졌으면". 비목표 §1.3 의 "화면 중심 GUI 앱" 은 **풀 데스크톱 앱** 으로 좁히고, 음성 보조의 시각적 프레즌스는 정체성에 포함. 기술: 로컬 FastAPI + WebSocket + Three.js (CDN, 빌드 단계 무) — `127.0.0.1` 한정. 의인화된 아바타(Joi/JARVIS HUD 등) 는 거부 — *Her* 의 추상 오브가 (a) 가족 누구나 받아들이기 자연스럽고, (b) 캐릭터 디자인·립싱크 등 별개 프로젝트 부담 없고, (c) 프로젝트 이름 `her` 의 정통성과 일치. 가족 실명은 본인 화면에서만 보이므로 §2.3 익명화 경계 밖. 이후 사용자 피드백 ("두뇌 뉴런 섬광")으로 단순 오브에서 **신경망 클러스터 + 시냅스 섬광**(Three.js + UnrealBloomPass + OutputPass) 으로 재디자인.
- _2026-05-01_: **Phase 4 (Proactive + Consolidator) 완료**. ① **Consolidator** (`apps/consolidator/`): 매일 03:00 launchd 배치, 직전 24h episodes → `gemini-2.5-flash` JSON 추출 → confidence ≥ 0.7 만 facts/events 로 승격, 충돌은 archive (CLAUDE.md §5.3 정합). ② **Proactive Engine** (`apps/proactive/`): 데몬 안에서 VoiceChannel 과 동시 실행, 트리거 = `TimeBasedTrigger` (events 24h/1h 임박) + `SilenceTrigger` (`silence_threshold_hours` 무상호작용) + `RecurringPatternTrigger` (Consolidator 가 facts 에 남긴 주간 패턴 신호). 일일 한도 (`settings.daily_proactive_limit` 기본 3) + 쿨다운 + 조용 모드 + dedup_key 6h 모두 통과해야 발화. ③ 채널 통합: `VoiceChannel.say()` / `TextChannel.say()` 추가, 사용자 활동 추적은 `~/.her/activity.json` (atomic write). ④ Presence 버스에 `proactive` state 이벤트 발행 — 오브가 사전 발화 직전·후로 시각 신호 표시.

---

## 13. Daemon (백그라운드 상시 모드)

Phase 3 부터 her 는 macOS launchd 데몬으로 상시 실행 가능하다.

- **첫 실행**: `make first-run` 으로 마이크 동의·환경변수 점검 (`GEMINI_API_KEY` 만 필수).
- **설치**: `make daemon-install` — launchd plist 를 `~/Library/LaunchAgents/` 에 배치, 재부팅 후 자동 시작.
- **제어**: `make daemon-{start,stop,status,logs}`.
- **설정 파일**: `~/.her/settings.toml` — 동의 상태, 깨우기 키워드(`wake_keyword`), 조용 모드 기본값 등 사용자 설정. her CLI 와 데몬 모두 이 파일을 읽는다.
- **깨우기 단어**: `settings.wake_keyword` 에 자연어 한 줄("자기야", "비서야" 등). Wake detector 는 로컬 faster-whisper 로 동작하므로 **별도 API 키 불필요**. `PICOVOICE_ACCESS_KEY` 는 더 이상 참조하지 않는다.
- **로그**: `~/.her/logs/her.log` (구조화 로그) + `her.stdout.log` / `her.stderr.log` (launchd raw 출력).
- **중요**: §10 사용자 동의 없이 마이크 상시 녹음 금지 규칙에 따라, 데몬은 `mic_consent_granted=true` 가 settings 에 기록되어 있지 않으면 시작하지 않고 즉시 종료한다 (exit code 78).

---

**이 문서는 살아있다. 결정이 바뀌면 즉시 업데이트하라.**


