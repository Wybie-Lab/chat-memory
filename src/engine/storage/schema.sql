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

-- Source-backed knowledge graph projection. Facts remain the canonical truth;
-- these rows are rebuildable from active facts and improve connected retrieval.
CREATE TABLE IF NOT EXISTS entities (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type     TEXT NOT NULL,
  canonical_key   TEXT NOT NULL UNIQUE,
  display_name    TEXT NOT NULL,
  aliases_json    TEXT NOT NULL DEFAULT '[]',
  confidence      REAL NOT NULL DEFAULT 1.0,
  created_at      INTEGER NOT NULL,
  updated_at      INTEGER NOT NULL,
  merged_into_id  INTEGER REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_display ON entities(display_name);

CREATE TABLE IF NOT EXISTS fact_entity_mentions (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  fact_id       INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  entity_id     INTEGER NOT NULL REFERENCES entities(id),
  role          TEXT NOT NULL,
  mention_text  TEXT,
  confidence    REAL NOT NULL DEFAULT 1.0,
  created_at    INTEGER NOT NULL,
  UNIQUE(fact_id, entity_id, role)
);

CREATE INDEX IF NOT EXISTS idx_mentions_fact ON fact_entity_mentions(fact_id);
CREATE INDEX IF NOT EXISTS idx_mentions_entity ON fact_entity_mentions(entity_id);

CREATE TABLE IF NOT EXISTS knowledge_edges (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  source_entity_id       INTEGER NOT NULL REFERENCES entities(id),
  predicate              TEXT NOT NULL,
  target_entity_id       INTEGER NOT NULL REFERENCES entities(id),
  confidence             REAL NOT NULL DEFAULT 1.0,
  source_fact_id          INTEGER REFERENCES facts(id),
  source_burst_id         INTEGER REFERENCES conversation_bursts(id),
  extracted_at            INTEGER NOT NULL,
  event_ts                INTEGER,
  valid_from_ts           INTEGER,
  valid_to_ts             INTEGER,
  status                  TEXT NOT NULL DEFAULT 'active',
  superseded_by_edge_id   INTEGER REFERENCES knowledge_edges(id),
  deleted_at              INTEGER,
  qualifiers_json         TEXT NOT NULL DEFAULT '{}',
  UNIQUE(source_entity_id, predicate, target_entity_id)
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges(source_entity_id, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges(target_entity_id, predicate);
CREATE INDEX IF NOT EXISTS idx_edges_fact ON knowledge_edges(source_fact_id);
CREATE INDEX IF NOT EXISTS idx_edges_active_source
  ON knowledge_edges(source_entity_id)
  WHERE status = 'active' AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_edges_active_target
  ON knowledge_edges(target_entity_id)
  WHERE status = 'active' AND deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS edge_sources (
  edge_id      INTEGER NOT NULL REFERENCES knowledge_edges(id) ON DELETE CASCADE,
  fact_id      INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  burst_id     INTEGER REFERENCES conversation_bursts(id),
  attached_at  INTEGER NOT NULL,
  PRIMARY KEY(edge_id, fact_id)
);

CREATE INDEX IF NOT EXISTS idx_edge_sources_fact ON edge_sources(fact_id);

-- Curator agent runs. The agent reads existing facts/entities and proposes
-- mutations (UPDATE/DELETE/MERGE) to be applied later. Nothing in agent_runs
-- or agent_actions touches the facts table directly — application is a
-- separate gated step (see applyRun / agent_actions.status flow).
CREATE TABLE IF NOT EXISTS agent_runs (
  id                  INTEGER PRIMARY KEY AUTOINCREMENT,
  trigger             TEXT NOT NULL,         -- 'manual' | 'entity_signal' | 'scheduled'
  scope_type          TEXT NOT NULL,         -- 'subject' | 'entity'
  scope_ref           TEXT NOT NULL,         -- subject_wa_id or entity_id (as text)
  trigger_fact_id     INTEGER REFERENCES facts(id),
  status              TEXT NOT NULL DEFAULT 'planned',
                                             -- 'planned' | 'running' | 'proposed'
                                             -- | 'applied' | 'rejected' | 'failed'
  budget_ops          INTEGER NOT NULL,
  budget_llm_calls    INTEGER NOT NULL,
  llm_calls_used      INTEGER NOT NULL DEFAULT 0,
  reasoning           TEXT,                  -- agent's final summary
  error               TEXT,                  -- on 'failed'
  started_at          INTEGER NOT NULL,
  completed_at        INTEGER,
  applied_at          INTEGER,
  approved_by         TEXT                   -- 'auto' | 'user:<name>' | NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs(status);
CREATE INDEX IF NOT EXISTS idx_agent_runs_scope ON agent_runs(scope_type, scope_ref);

-- Proposed mutations from a curator run. status='proposed' until either
-- applied (creates / supersedes / soft-deletes facts in the apply path) or
-- rejected by a reviewer. citing_fact_ids_json is required (≥1) so every
-- proposal is grounded in existing memory.
CREATE TABLE IF NOT EXISTS agent_actions (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id               INTEGER NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
  seq                  INTEGER NOT NULL,    -- order within the run
  op                   TEXT NOT NULL,       -- 'update' | 'delete' | 'merge'
  target_fact_id       INTEGER REFERENCES facts(id),
  new_content          TEXT,
  new_category         TEXT,
  merge_fact_ids_json  TEXT,                -- for 'merge': source fact ids being collapsed
  citing_fact_ids_json TEXT NOT NULL,       -- ≥1 required, validated at insert time
  reason               TEXT NOT NULL,
  confidence           REAL NOT NULL,
  status               TEXT NOT NULL DEFAULT 'proposed',
                                            -- 'proposed' | 'applied' | 'rejected' | 'skipped'
  applied_fact_id      INTEGER REFERENCES facts(id),
  rejected_reason      TEXT,
  created_at           INTEGER NOT NULL,
  applied_at           INTEGER
);

CREATE INDEX IF NOT EXISTS idx_agent_actions_run ON agent_actions(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_actions_status ON agent_actions(status);
