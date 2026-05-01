"""Main consolidation orchestrator.

Pulls last 24h episodes, extracts facts/events via Gemini Flash,
promotes high-confidence items to long-term semantic memory,
archives conflicts, and writes a JSON log.

CLAUDE.md §5.3
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

from apps.consolidator.extractor import ExtractionResult, extract_facts_and_events
from apps.consolidator.promoter import promote_event, promote_fact
from apps.memory.store import Episode, MemoryStore

log = structlog.get_logger(__name__)

DEFAULT_LOG_DIR = Path("data/consolidation_log")
DEFAULT_LOOKBACK_HOURS = 24


@dataclass
class ConsolidationReport:
    ran_at: str  # ISO timestamp
    episodes_processed: int
    facts_extracted: int  # all LLM candidates, pre-filter
    facts_promoted: int  # confidence >= threshold and stored
    facts_archived: int  # old facts moved to archived
    events_added: int
    errors: list[str] = field(default_factory=list)


def _filter_recent_episodes(
    episodes: list[Episode],
    cutoff_iso: str,
) -> list[Episode]:
    """Return episodes whose when_at >= cutoff_iso."""
    results: list[Episode] = []
    for ep in episodes:
        # Normalize both timestamps to naive UTC for comparison
        try:
            ep_ts = ep.when_at.replace("Z", "+00:00")
            try:
                ep_dt = datetime.fromisoformat(ep_ts)
            except ValueError:
                ep_dt = datetime.fromisoformat(ep_ts[:19])
            if ep_dt.tzinfo is None:
                ep_dt = ep_dt.replace(tzinfo=timezone.utc)

            cutoff_ts = cutoff_iso.replace("Z", "+00:00")
            try:
                cutoff_dt = datetime.fromisoformat(cutoff_ts)
            except ValueError:
                cutoff_dt = datetime.fromisoformat(cutoff_ts[:19])
            if cutoff_dt.tzinfo is None:
                cutoff_dt = cutoff_dt.replace(tzinfo=timezone.utc)

            if ep_dt >= cutoff_dt:
                results.append(ep)
        except (ValueError, TypeError) as exc:
            log.warning("runner.episode_date_parse_error", ep_id=ep.id, error=str(exc))
    return results


def _write_log(
    log_dir: Path,
    report: ConsolidationReport,
) -> None:
    """Write consolidation report as JSON under log_dir/<YYYY-MM-DD>.json."""
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = report.ran_at[:10]  # YYYY-MM-DD
    log_file = log_dir / f"{date_str}.json"

    # Append to existing file if it exists (multiple runs same day)
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
    """Run the full consolidation pipeline.

    Args:
        store: Memory store (real names, per §2.3).
        now_iso: Override current time (for testing). Defaults to UTC now.
        lookback_hours: How many hours back to scan. Default 24h.
        log_dir: Where to write JSON logs. Default data/consolidation_log/.
        client: GeminiClient or mock. Required unless dry_run=True.
        dry_run: If True, skip LLM calls (for testing/offline use).
        confidence_threshold: Min confidence for fact promotion.
    """
    ran_at = now_iso or datetime.now(tz=timezone.utc).isoformat()
    effective_log_dir = log_dir or DEFAULT_LOG_DIR
    report = ConsolidationReport(
        ran_at=ran_at,
        episodes_processed=0,
        facts_extracted=0,
        facts_promoted=0,
        facts_archived=0,
        events_added=0,
        errors=[],
    )

    # 1. Compute cutoff and pull recent episodes
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

    # Pull episodes; store is synchronous (SQLite must stay on its creation thread).
    all_episodes = store.list_recent_episodes(limit=1000)
    recent_episodes = _filter_recent_episodes(all_episodes, cutoff_iso)

    log.info(
        "runner.episodes_found",
        total=len(all_episodes),
        recent=len(recent_episodes),
        cutoff=cutoff_iso,
    )

    if not recent_episodes:
        log.info("runner.no_episodes")
        _write_log(effective_log_dir, report)
        return report

    report.episodes_processed = len(recent_episodes)

    # 2. Extract facts and events (skip in dry_run mode)
    extraction: ExtractionResult
    if dry_run or client is None:
        log.info("runner.dry_run_skip_llm")
        extraction = ExtractionResult()
    else:
        try:
            extraction = await extract_facts_and_events(
                recent_episodes, client=client
            )
            report.errors.extend(extraction.errors)
        except Exception as exc:
            msg = f"Extraction failed: {exc}"
            log.error("runner.extraction_error", error=msg)
            report.errors.append(msg)
            _write_log(effective_log_dir, report)
            return report

    report.facts_extracted = len(extraction.facts)

    # 3. Promote facts (synchronous — SQLite must stay on its creation thread)
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

    # 4. Add events (synchronous)
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

    # 5. Write log
    _write_log(effective_log_dir, report)

    log.info(
        "runner.done",
        episodes=report.episodes_processed,
        facts_extracted=report.facts_extracted,
        facts_promoted=report.facts_promoted,
        facts_archived=report.facts_archived,
        events_added=report.events_added,
        errors=len(report.errors),
    )
    return report
