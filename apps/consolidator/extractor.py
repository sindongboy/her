"""Gemini-driven fact / event / note extraction from chat sessions.

Uses gemini-2.5-flash with structured JSON output per CLAUDE.md §5.3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog

from apps.memory.store import Session

log = structlog.get_logger(__name__)

EXTRACTOR_MODEL_ID = "gemini-2.5-flash"
BATCH_SIZE = 10  # sessions per LLM call

_SYSTEM_PROMPT = """\
당신은 가족 AI 비서의 메모리 통합기입니다.
지난 24시간 대화에서 다음을 추출하세요:

1. fact (사실): 가족·사람에 대한 안정적인 진술
   - subject_person_name: 인물 (예: "어머니", "아내"). 없으면 null.
   - predicate: 관계 동사/명사 (예: "좋아한다", "알러지", "직업")
   - object: 구체 대상 (예: "단호박 케이크", "땅콩")
   - confidence: 0..1 (대화 내용 근거의 신뢰도)

2. event (일정): 시간이 있는 이벤트
   - person_name: 관련 인물 이름 (없으면 null)
   - type: 이벤트 유형 (예: "birthday", "appointment", "trip")
   - title: 이벤트 제목
   - when_at: ISO 8601 형식 날짜/시각
   - recurrence: 반복 규칙 (없으면 null)

3. note (메모): 사람에 매이지 않은 자유 메모, 결정 사항, 아이디어, 할일.
   사실로 승격하기 애매하지만 다음에도 떠올리고 싶은 정보.
   - content: 한 문장으로 요약된 메모 본문
   - tags: 관련 키워드 0개 이상 (예: ["work", "decision"])

응답 형식: 엄격한 JSON. 추가 텍스트 금지.
스키마: {"facts": [...], "events": [...], "notes": [...]}

대화:
"""


@dataclass(slots=True, frozen=True)
class ExtractedFact:
    subject_person_name: str | None
    predicate: str
    object: str
    confidence: float


@dataclass(slots=True, frozen=True)
class ExtractedEvent:
    person_name: str | None
    type: str
    title: str
    when_at: str
    recurrence: str | None


@dataclass(slots=True, frozen=True)
class ExtractedNote:
    content: str
    tags: list[str]


@dataclass
class ExtractionResult:
    facts: list[ExtractedFact] = field(default_factory=list)
    events: list[ExtractedEvent] = field(default_factory=list)
    notes: list[ExtractedNote] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _build_sessions_text(sessions_with_text: list[tuple[Session, str]]) -> str:
    """Format sessions into a readable text block for the LLM."""
    lines: list[str] = []
    for session, content in sessions_with_text:
        lines.append(f"[{session.last_active_at}] (session_id={session.id})")
        if session.title:
            lines.append(f"제목: {session.title}")
        lines.append(content if content.strip() else "(내용 없음)")
        lines.append("")
    return "\n".join(lines).strip()


def _parse_extraction_response(
    text: str,
) -> tuple[list[ExtractedFact], list[ExtractedEvent], list[ExtractedNote]]:
    """Parse LLM JSON response into extracted facts, events, and notes."""
    data = json.loads(text)
    facts: list[ExtractedFact] = []
    events: list[ExtractedEvent] = []
    notes: list[ExtractedNote] = []

    for item in data.get("facts", []):
        confidence = float(item.get("confidence", 0.0))
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not predicate or not obj:
            continue
        subject = item.get("subject_person_name")
        facts.append(
            ExtractedFact(
                subject_person_name=str(subject) if subject else None,
                predicate=predicate,
                object=obj,
                confidence=max(0.0, min(1.0, confidence)),
            )
        )

    for item in data.get("events", []):
        etype = str(item.get("type", "")).strip()
        title = str(item.get("title", "")).strip()
        when_at = str(item.get("when_at", "")).strip()
        if not etype or not title or not when_at:
            continue
        pname = item.get("person_name")
        recurrence = item.get("recurrence")
        events.append(
            ExtractedEvent(
                person_name=str(pname) if pname else None,
                type=etype,
                title=title,
                when_at=when_at,
                recurrence=str(recurrence) if recurrence else None,
            )
        )

    for item in data.get("notes", []):
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        raw_tags = item.get("tags") or []
        tags: list[str] = []
        if isinstance(raw_tags, list):
            for t in raw_tags:
                t_str = str(t).strip()
                if t_str:
                    tags.append(t_str)
        notes.append(ExtractedNote(content=content, tags=tags))

    return facts, events, notes


def _call_llm_sync(
    client: object,
    sessions_with_text: list[tuple[Session, str]],
    model_id: str,
) -> str:
    """Call the LLM with structured JSON config when available."""
    sessions_text = _build_sessions_text(sessions_with_text)
    prompt = _SYSTEM_PROMPT + "\n" + sessions_text

    try:
        from google.genai import types  # type: ignore[import-untyped]

        _schema = {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subject_person_name": {"type": ["string", "null"]},
                            "predicate": {"type": "string"},
                            "object": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["predicate", "object", "confidence"],
                    },
                },
                "events": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "person_name": {"type": ["string", "null"]},
                            "type": {"type": "string"},
                            "title": {"type": "string"},
                            "when_at": {"type": "string"},
                            "recurrence": {"type": ["string", "null"]},
                        },
                        "required": ["type", "title", "when_at"],
                    },
                },
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["facts", "events", "notes"],
        }

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_schema,
        )
        genai_client = getattr(client, "_client", client)
        response = genai_client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )
        return response.text or "{}"
    except Exception:
        pass

    messages = [{"role": "user", "content": prompt}]
    result: str = client.generate(messages)  # type: ignore[attr-defined]
    return result


def _extract_batch(
    sessions_with_text: list[tuple[Session, str]],
    *,
    client: object,
    model_id: str,
) -> tuple[
    list[ExtractedFact], list[ExtractedEvent], list[ExtractedNote], list[str]
]:
    """Extract facts/events/notes from one batch with one retry on parse failure."""
    errors: list[str] = []
    for attempt in range(2):
        try:
            raw = _call_llm_sync(client, sessions_with_text, model_id)
            facts, events, notes = _parse_extraction_response(raw)
            return facts, events, notes, errors
        except json.JSONDecodeError as exc:
            if attempt == 0:
                log.warning("extractor.parse_error.retrying", error=str(exc))
                continue
            msg = f"JSON parse error after retry: {exc}"
            log.error("extractor.parse_error.final", error=msg)
            errors.append(msg)
            return [], [], [], errors
        except Exception as exc:
            msg = f"LLM call error (batch of {len(sessions_with_text)}): {exc}"
            log.error("extractor.llm_error", error=msg)
            errors.append(msg)
            return [], [], [], errors
    return [], [], [], errors


async def extract_facts_and_events(
    sessions_with_text: list[tuple[Session, str]],
    *,
    client: object,
    model_id: str = EXTRACTOR_MODEL_ID,
) -> ExtractionResult:
    """Extract structured facts, events, and notes from sessions."""
    import asyncio

    result = ExtractionResult()

    if not sessions_with_text:
        return result

    batches = [
        sessions_with_text[i : i + BATCH_SIZE]
        for i in range(0, len(sessions_with_text), BATCH_SIZE)
    ]
    log.info(
        "extractor.start",
        total_sessions=len(sessions_with_text),
        batches=len(batches),
        model=model_id,
    )

    for batch_idx, batch in enumerate(batches):
        log.debug("extractor.batch", batch_idx=batch_idx, size=len(batch))
        facts, events, notes, errors = await asyncio.to_thread(
            _extract_batch, batch, client=client, model_id=model_id
        )
        result.facts.extend(facts)
        result.events.extend(events)
        result.notes.extend(notes)
        result.errors.extend(errors)

    log.info(
        "extractor.done",
        facts=len(result.facts),
        events=len(result.events),
        notes=len(result.notes),
        errors=len(result.errors),
    )
    return result
