"""Main consolidation orchestrator.

Pulls last 24h sessions, builds a per-session text blob from messages,
extracts facts/events/notes via Gemini Flash, promotes high-confidence
items to long-term semantic memory, and writes a JSON log.

CLAUDE.md §5.3
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

from apps.consolidator.extractor import ExtractionResult, extract_facts_and_events
from apps.consolidator.promoter import promote_event, promote_fact, promote_note
from apps.memory.store import MemoryStore, Session

log = structlog.get_logger(__name__)

DEFAULT_LOG_DIR = Path("data/consolidation_log")
DEFAULT_LOOKBACK_HOURS = 24
_MAX_MESSAGE_CHARS = 4000  # cap per session to keep prompts bounded


@dataclass
class ConsolidationReport:
    ran_at: str
    sessions_processed: int
    facts_extracted: int
    facts_promoted: int
    facts_archived: int
    events_added: int
    notes_added: int
    errors: list[str] = field(default_factory=list)


def _filter_recent_sessions(
    sessions: list[Session],
    cutoff_iso: str,
) -> list[Session]:
    """Return sessions whose last_active_at >= cutoff_iso."""
    results: list[Session] = []
    for s in sessions:
        try:
            ts_str = s.last_active_at.replace("Z", "+00:00")
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                ts = datetime.fromisoformat(ts_str[:19])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            cut_str = cutoff_iso.replace("Z", "+00:00")
            try:
                cut = datetime.fromisoformat(cut_str)
            except ValueError:
                cut = datetime.fromisoformat(cut_str[:19])
            if cut.tzinfo is None:
                cut = cut.replace(tzinfo=timezone.utc)

            if ts >= cut:
                results.append(s)
        except (ValueError, TypeError) as exc:
            log.warning("runner.session_date_parse_error", session_id=s.id, error=str(exc))
    return results


def _build_session_text(store: MemoryStore, session: Session) -> str:
    """Concatenate session summary + recent messages into one text blob."""
    chunks: list[str] = []
    if session.summary:
        chunks.append(f"요약: {session.summary}")
    msgs = store.recall_messages(session.id, limit=40)
    for m in msgs:
        prefix = "사용자" if m.role == "user" else "비서"
        chunks.append(f"{prefix}: {m.content}")
    blob = "\n".join(chunks)
    return blob[:_MAX_MESSAGE_CHARS]


def _write_log(log_dir: Path, report: ConsolidationReport) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = report.ran_at[:10]
    log_file = log_dir / f"{date_str}.json"

    entries: list[dict] = []
    if log_file.exists():
        try:
            existing = json.loads(log_file.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                entries = existing
            else:
                entries = [existing]
        except json.JSONDecodeError:
            entries = []

    entries.append(asdict(report))
    log_file.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("runner.log_written", path=str(log_file))


async def run_consolidation(
    store: MemoryStore,
    *,
    now_iso: str | None = None,
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    log_dir: Path | None = None,
    client: object | None = None,
    dry_run: bool = False,
    confidence_threshold: float = 0.7,
) -> ConsolidationReport:
    ran_at = now_iso or datetime.now(tz=timezone.utc).isoformat()
    effective_log_dir = log_dir or DEFAULT_LOG_DIR
    report = ConsolidationReport(
        ran_at=ran_at,
        sessions_processed=0,
        facts_extracted=0,
        facts_promoted=0,
        facts_archived=0,
        events_added=0,
        notes_added=0,
        errors=[],
    )

    try:
        now_dt = datetime.fromisoformat(ran_at.replace("Z", "+00:00"))
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        cutoff_dt = now_dt - timedelta(hours=lookback_hours)
        cutoff_iso = cutoff_dt.isoformat()
    except (ValueError, TypeError) as exc:
        msg = f"Failed to parse now_iso '{ran_at}': {exc}"
        log.error("runner.time_parse_error", error=msg)
        report.errors.append(msg)
        _write_log(effective_log_dir, report)
        return report

    all_sessions = store.list_recent_sessions(limit=1000, include_archived=False)
    recent_sessions = _filter_recent_sessions(all_sessions, cutoff_iso)

    log.info(
        "runner.sessions_found",
        total=len(all_sessions),
        recent=len(recent_sessions),
        cutoff=cutoff_iso,
    )

    if not recent_sessions:
        log.info("runner.no_sessions")
        _write_log(effective_log_dir, report)
        return report

    report.sessions_processed = len(recent_sessions)

    # Build per-session text blobs.
    sessions_with_text: list[tuple[Session, str]] = []
    for s in recent_sessions:
        sessions_with_text.append((s, _build_session_text(store, s)))

    extraction: ExtractionResult
    if dry_run or client is None:
        log.info("runner.dry_run_skip_llm")
        extraction = ExtractionResult()
    else:
        try:
            extraction = await extract_facts_and_events(
                sessions_with_text, client=client
            )
            report.errors.extend(extraction.errors)
        except Exception as exc:
            msg = f"Extraction failed: {exc}"
            log.error("runner.extraction_error", error=msg)
            report.errors.append(msg)
            _write_log(effective_log_dir, report)
            return report

    report.facts_extracted = len(extraction.facts)

    for extracted_fact in extraction.facts:
        try:
            outcome = promote_fact(
                store,
                extracted_fact,
                confidence_threshold=confidence_threshold,
            )
            if outcome.new_fact_id is not None:
                report.facts_promoted += 1
            report.facts_archived += len(outcome.archived_fact_ids)
        except Exception as exc:
            msg = f"Fact promotion error ({extracted_fact.predicate}): {exc}"
            log.error("runner.fact_promote_error", error=msg)
            report.errors.append(msg)

    for extracted_event in extraction.events:
        try:
            errors_buf: list[str] = []
            event_id = promote_event(
                store,
                extracted_event,
                errors=errors_buf,
            )
            if event_id is not None:
                report.events_added += 1
            report.errors.extend(errors_buf)
        except Exception as exc:
            msg = f"Event add error ({extracted_event.title}): {exc}"
            log.error("runner.event_add_error", error=msg)
            report.errors.append(msg)

    for extracted_note in extraction.notes:
        try:
            note_id = promote_note(store, extracted_note)
            if note_id is not None:
                report.notes_added += 1
        except Exception as exc:
            msg = f"Note add error ({extracted_note.content[:40]}): {exc}"
            log.error("runner.note_add_error", error=msg)
            report.errors.append(msg)

    _write_log(effective_log_dir, report)

    log.info(
        "runner.done",
        sessions=report.sessions_processed,
        facts_extracted=report.facts_extracted,
        facts_promoted=report.facts_promoted,
        facts_archived=report.facts_archived,
        events_added=report.events_added,
        notes_added=report.notes_added,
        errors=len(report.errors),
    )
    return report
