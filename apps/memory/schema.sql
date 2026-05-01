-- Schema v2 for the `her` memory store. See CLAUDE.md §5.
-- vec0 virtual tables require sqlite-vec to be loaded *before* this script runs.
-- MemoryStore handles version detection + backup before applying this script.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- People: the central node of the memory graph.
CREATE TABLE IF NOT EXISTS people (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL,
    relation          TEXT,
    birthday          TEXT,
    preferences_json  TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_people_relation ON people(relation);

-- Sessions: one chat session in the web UI. Replaces v1 "episodes".
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_active_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    title           TEXT,
    summary         TEXT,
    archived_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON sessions(last_active_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(archived_at)
    WHERE archived_at IS NULL;

-- Messages: one chat turn. Per-session ordered by ts.
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content     TEXT NOT NULL,
    ts          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session_ts ON messages(session_id, ts);

-- Events: scheduled or recurring items, optionally linked to a person.
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER REFERENCES people(id) ON DELETE SET NULL,
    type        TEXT NOT NULL,
    title       TEXT NOT NULL,
    when_at     TEXT NOT NULL,
    recurrence  TEXT,
    source      TEXT,
    status      TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_events_when_at ON events(when_at);
CREATE INDEX IF NOT EXISTS idx_events_person  ON events(person_id);

-- Facts: structured assertions with provenance. Conflicts archive, never delete.
CREATE TABLE IF NOT EXISTS facts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_person_id   INTEGER REFERENCES people(id) ON DELETE CASCADE,
    predicate           TEXT NOT NULL,
    object              TEXT NOT NULL,
    confidence          REAL NOT NULL,
    source_session_id   INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    valid_from          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_person_id);
CREATE INDEX IF NOT EXISTS idx_facts_active  ON facts(subject_person_id) WHERE archived_at IS NULL;

-- Preferences: rapid-access 'domain → value' map per person. person_id may be
-- NULL to express the user's own preferences.
CREATE TABLE IF NOT EXISTS preferences (
    person_id     INTEGER REFERENCES people(id) ON DELETE CASCADE,
    domain        TEXT NOT NULL,
    value         TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_preferences_unique
    ON preferences(COALESCE(person_id, -1), domain, value);

-- Notes: free-form memos surfaced via consolidation or user input.
CREATE TABLE IF NOT EXISTS notes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    content            TEXT NOT NULL,
    tags               TEXT NOT NULL DEFAULT '[]',  -- JSON array
    source_session_id  INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    created_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    archived_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_notes_active ON notes(archived_at)
    WHERE archived_at IS NULL;

-- Attachments: text-channel-only inbound files.
CREATE TABLE IF NOT EXISTS attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sha256       TEXT NOT NULL,
    mime         TEXT,
    ext          TEXT,
    byte_size    INTEGER,
    path         TEXT NOT NULL,
    description  TEXT,
    ingested_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, sha256)
);
CREATE INDEX IF NOT EXISTS idx_attachments_sha ON attachments(sha256);

-- Session embeddings via sqlite-vec.
-- Operational rules (CLAUDE.md §3.2): dim and model_id are fixed per row;
-- writes use task_type=RETRIEVAL_DOCUMENT, queries use RETRIEVAL_QUERY.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_sessions USING vec0(
    session_id INTEGER PRIMARY KEY,
    embedding  FLOAT[768]
);

CREATE TABLE IF NOT EXISTS session_embedding_meta (
    session_id  INTEGER PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    task_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Per-message embeddings: turn-level semantic search.
CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
    message_id INTEGER PRIMARY KEY,
    embedding  FLOAT[768]
);

CREATE TABLE IF NOT EXISTS message_embedding_meta (
    message_id  INTEGER PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
    model_id    TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    task_type   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
