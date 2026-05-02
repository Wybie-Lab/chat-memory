PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS contacts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  wa_id         TEXT NOT NULL UNIQUE,
  display_name  TEXT,
  is_group      INTEGER NOT NULL DEFAULT 0,
  whitelisted   INTEGER NOT NULL DEFAULT 0,
  notes         TEXT,
  first_seen_at INTEGER NOT NULL,
  last_seen_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_contacts_whitelisted ON contacts(whitelisted);

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
  filter_kept   INTEGER,
  processed_at  INTEGER,
  ingested_at   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_contact_ts ON raw_messages(contact_id, ts);
CREATE INDEX IF NOT EXISTS idx_raw_unprocessed ON raw_messages(filter_kept) WHERE filter_kept IS NULL;

CREATE TABLE IF NOT EXISTS facts (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  source_msg_id   INTEGER NOT NULL REFERENCES raw_messages(id),
  subject_wa_id   TEXT NOT NULL,
  category        TEXT NOT NULL,
  content         TEXT NOT NULL,
  confidence      REAL NOT NULL DEFAULT 1.0,
  extracted_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_wa_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);

CREATE TABLE IF NOT EXISTS processing_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  msg_id       INTEGER REFERENCES raw_messages(id),
  stage        TEXT NOT NULL,
  model        TEXT NOT NULL,
  tokens_in    INTEGER,
  tokens_out   INTEGER,
  cost_usd     REAL,
  ts           INTEGER NOT NULL
);
