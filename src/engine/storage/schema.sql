PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS contacts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  wa_id           TEXT NOT NULL UNIQUE,
  display_name    TEXT,
  is_group        INTEGER NOT NULL DEFAULT 0,
  whitelisted     INTEGER NOT NULL DEFAULT 0,
  notes           TEXT,
  first_seen_at   INTEGER NOT NULL,
  last_seen_at    INTEGER NOT NULL,
  live_cutoff_ts  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_contacts_whitelisted ON contacts(whitelisted);

CREATE TABLE IF NOT EXISTS conversation_bursts (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  contact_id     INTEGER NOT NULL REFERENCES contacts(id),
  start_ts       INTEGER NOT NULL,
  end_ts         INTEGER NOT NULL,
  message_count  INTEGER NOT NULL DEFAULT 0,
  filter_kept    INTEGER,
  processed_at   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_bursts_contact_end ON conversation_bursts(contact_id, end_ts);
CREATE INDEX IF NOT EXISTS idx_bursts_unprocessed ON conversation_bursts(processed_at) WHERE processed_at IS NULL;

CREATE TABLE IF NOT EXISTS raw_messages (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  wa_msg_id     TEXT NOT NULL UNIQUE,
  contact_id    INTEGER NOT NULL REFERENCES contacts(id),
  sender_wa_id  TEXT NOT NULL,
  direction     TEXT NOT NULL CHECK (direction IN ('in','out')),
  body          TEXT NOT NULL DEFAULT '',
  ts            INTEGER NOT NULL,
  media_type    TEXT,
  media_pointer TEXT,
  burst_id      INTEGER REFERENCES conversation_bursts(id),
  filter_kept   INTEGER,
  processed_at  INTEGER,
  ingested_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_contact_ts ON raw_messages(contact_id, ts);
-- idx_raw_burst, idx_raw_unburst created in migrate() after burst_id is guaranteed.

CREATE TABLE IF NOT EXISTS facts (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  source_msg_id     INTEGER REFERENCES raw_messages(id),
  source_burst_id   INTEGER REFERENCES conversation_bursts(id),
  subject_wa_id     TEXT NOT NULL,
  category          TEXT NOT NULL,
  content           TEXT NOT NULL,
  confidence        REAL NOT NULL DEFAULT 1.0,
  extracted_at      INTEGER NOT NULL,
  -- Unix seconds when an event/commitment is anchored. Null for non-event
  -- categories or when the burst didn't determine an unambiguous date.
  -- Retrieval surfaces this separately so future plans can rank above stale
  -- history.
  event_ts          INTEGER,
  superseded_by_id  INTEGER REFERENCES facts(id),
  deleted_at        INTEGER
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_wa_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
-- idx_facts_burst, idx_facts_active created in migrate() after the columns
-- they reference are guaranteed to exist on the table.

CREATE TABLE IF NOT EXISTS processing_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  msg_id       INTEGER REFERENCES raw_messages(id),
  burst_id     INTEGER REFERENCES conversation_bursts(id),
  stage        TEXT NOT NULL,
  model        TEXT NOT NULL,
  tokens_in    INTEGER,
  tokens_out   INTEGER,
  cost_usd     REAL,
  ts           INTEGER NOT NULL
);

-- Many supporting turns / bursts per fact. The legacy facts.source_burst_id
-- and facts.source_msg_id are still populated for backward-compat reads, but
-- this table is the source of truth for "what evidence backs this fact".
CREATE TABLE IF NOT EXISTS fact_sources (
  fact_id          INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  source_burst_id  INTEGER REFERENCES conversation_bursts(id),
  source_msg_id    INTEGER REFERENCES raw_messages(id),
  attached_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fact_sources_fact ON fact_sources(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_sources_burst ON fact_sources(source_burst_id);
CREATE INDEX IF NOT EXISTS idx_fact_sources_msg ON fact_sources(source_msg_id);

-- Rolled-up prose summaries per (subject, category). Refreshed by the
-- pipeline whenever facts change for that group. Read at chat time via
-- composeMemoryBlock — agents read prose better than triples.
CREATE TABLE IF NOT EXISTS cluster_summaries (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  subject_wa_id     TEXT NOT NULL,
  category          TEXT NOT NULL,
  summary           TEXT NOT NULL,
  fact_count        INTEGER NOT NULL,
  fact_ids_json     TEXT NOT NULL,
  last_refreshed_at INTEGER NOT NULL,
  UNIQUE(subject_wa_id, category)
);

CREATE INDEX IF NOT EXISTS idx_cluster_subject ON cluster_summaries(subject_wa_id);
