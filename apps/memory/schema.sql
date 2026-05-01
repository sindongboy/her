-- Schema for the `her` memory store. See CLAUDE.md §5.2.
-- The vec0 virtual table requires sqlite-vec to be loaded *before* this script runs.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- People: the central node of the memory graph.
CREATE TABLE IF NOT EXISTS people (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL,
    relation          TEXT,
    birthday          TEXT,                         -- 'YYYY-MM-DD' or 'MM-DD'
    preferences_json  TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_people_relation ON people(relation);

-- Episodes: one conversational session.
CREATE TABLE IF NOT EXISTS episodes (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    when_at                TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    summary                TEXT,
    transcript_compressed  BLOB,
    primary_channel        TEXT NOT NULL CHECK (primary_channel IN ('voice','text','mixed'))
);
CREATE INDEX IF NOT EXISTS idx_episodes_when_at ON episodes(when_at);

-- Events: scheduled or recurring items, optionally linked to a person.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER REFERENCES people(id) ON DELETE SET NULL,
    type        TEXT NOT NULL,
    title       TEXT NOT NULL,
    when_at     TEXT NOT NULL,
    recurrence  TEXT,                               -- iCal RRULE; NULL = one-shot
    source      TEXT,
    status      TEXT NOT NULL DEFAULT 'pending'     -- 'pending' | 'done' | 'cancelled'
);
CREATE INDEX IF NOT EXISTS idx_events_when_at ON events(when_at);
CREATE INDEX IF NOT EXISTS idx_events_person  ON events(person_id);

-- Facts: structured assertions with provenance. Conflicting facts are archived, not deleted.
CREATE TABLE IF NOT EXISTS facts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_person_id   INTEGER REFERENCES people(id) ON DELETE CASCADE,
    predicate           TEXT NOT NULL,
    object              TEXT NOT NULL,
    confidence          REAL NOT NULL,
    source_episode_id   INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
    valid_from          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_person_id);
CREATE INDEX IF NOT EXISTS idx_facts_active  ON facts(subject_person_id) WHERE archived_at IS NULL;

-- Preferences: rapid-access 'domain → value' map per person.
CREATE TABLE IF NOT EXISTS preferences (
    person_id     INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    domain        TEXT NOT NULL,
    value         TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (person_id, domain, value)
);

-- Attachments: text-channel-only inbound files (CLAUDE.md §2.5, §6.2).
CREATE TABLE IF NOT EXISTS attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id   INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    sha256       TEXT NOT NULL,
    mime         TEXT,
    ext          TEXT,
    byte_size    INTEGER,
    path         TEXT NOT NULL,
    description  TEXT,
    ingested_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(episode_id, sha256)
);
CREATE INDEX IF NOT EXISTS idx_attachments_sha ON attachments(sha256);

-- Episode embeddings via sqlite-vec.
-- Operational rules (CLAUDE.md §3.2): dim and model_id are fixed per row;
-- writes use task_type=RETRIEVAL_DOCUMENT, queries use RETRIEVAL_QUERY.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(
    episode_id INTEGER PRIMARY KEY,
    embedding  FLOAT[768]
);

CREATE TABLE IF NOT EXISTS episode_embedding_meta (
    episode_id  INTEGER PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    task_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
