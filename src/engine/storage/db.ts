import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import fs from 'fs';
import path from 'path';

export type DB = Database.Database;

export const BURST_GAP_SECONDS = 30 * 60;

export interface RawMessageInput {
  wa_msg_id: string;
  contact_id: number;
  sender_wa_id: string;
  direction: 'in' | 'out';
  body: string;
  ts: number;
  media_type?: string | null;
  media_pointer?: string | null;
}

export interface FactInput {
  source_msg_id?: number | null;
  source_burst_id: number;
  subject: string;
  category: string;
  content: string;
  confidence: number;
  /** Unix seconds. Set for event/commitment when burst pinned an unambiguous date. Null otherwise. */
  event_ts?: number | null;
}

export interface ProcessingLogInput {
  msg_id?: number | null;
  burst_id?: number | null;
  stage: 'filter' | 'extract' | 'embed' | 'graph_extract';
  model: string;
  tokens_in: number;
  tokens_out: number;
}

export interface BurstRow {
  id: number;
  contact_id: number;
  start_ts: number;
  end_ts: number;
  message_count: number;
  filter_kept: number | null;
  processed_at: number | null;
}

export interface UnprocessedBurst {
  id: number;
  contact_id: number;
  start_ts: number;
  end_ts: number;
  message_count: number;
  chat_wa_id: string;
  chat_display_name: string | null;
  is_group: boolean;
  chat_notes: string | null;
}

export interface BurstMessage {
  id: number;
  sender_wa_id: string;
  direction: 'in' | 'out';
  body: string;
  ts: number;
}

export function openDb(dbPath: string): DB {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);

  sqliteVec.load(db);

  const schemaPath = path.join(__dirname, 'schema.sql');
  const schema = fs.readFileSync(schemaPath, 'utf-8');
  db.exec(schema);

  migrate(db);

  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(
      fact_id INTEGER PRIMARY KEY,
      embedding FLOAT[1024]
    );
  `);

  return db;
}

function migrate(db: DB): void {
  // ALTER TABLE first to guarantee all migrated-in columns exist, THEN
  // create indexes that reference them. Indexes use IF NOT EXISTS so they're
  // a no-op for fresh DBs where the column was created by the CREATE TABLE
  // in schema.sql.
  if (!hasColumn(db, 'raw_messages', 'burst_id')) {
    db.exec('ALTER TABLE raw_messages ADD COLUMN burst_id INTEGER REFERENCES conversation_bursts(id)');
  }
  if (!hasColumn(db, 'facts', 'source_burst_id')) {
    db.exec('ALTER TABLE facts ADD COLUMN source_burst_id INTEGER REFERENCES conversation_bursts(id)');
  }
  if (!hasColumn(db, 'facts', 'superseded_by_id')) {
    db.exec('ALTER TABLE facts ADD COLUMN superseded_by_id INTEGER REFERENCES facts(id)');
  }
  if (!hasColumn(db, 'facts', 'deleted_at')) {
    db.exec('ALTER TABLE facts ADD COLUMN deleted_at INTEGER');
  }
  if (!hasColumn(db, 'processing_log', 'burst_id')) {
    db.exec('ALTER TABLE processing_log ADD COLUMN burst_id INTEGER REFERENCES conversation_bursts(id)');
  }
  if (!hasColumn(db, 'contacts', 'live_cutoff_ts')) {
    db.exec('ALTER TABLE contacts ADD COLUMN live_cutoff_ts INTEGER');
  }
  if (!hasColumn(db, 'facts', 'event_ts')) {
    db.exec('ALTER TABLE facts ADD COLUMN event_ts INTEGER');
  }

  db.exec('CREATE INDEX IF NOT EXISTS idx_raw_burst ON raw_messages(burst_id)');
  db.exec('CREATE INDEX IF NOT EXISTS idx_raw_unburst ON raw_messages(contact_id, ts) WHERE burst_id IS NULL');
  db.exec('CREATE INDEX IF NOT EXISTS idx_facts_burst ON facts(source_burst_id)');
  db.exec(`CREATE INDEX IF NOT EXISTS idx_facts_active
           ON facts(subject_wa_id)
           WHERE superseded_by_id IS NULL AND deleted_at IS NULL`);
  db.exec('CREATE INDEX IF NOT EXISTS idx_facts_event_ts ON facts(event_ts) WHERE event_ts IS NOT NULL');

  backfillFactSources(db);
}

/**
 * One-shot backfill: copy every existing fact's primary source into the new
 * fact_sources table so multi-source reads return correct citations on
 * pre-multi-source rows. Idempotent — only inserts when fact_sources is empty
 * relative to the row count of facts that have a source.
 */
function backfillFactSources(db: DB): void {
  const existing = db.prepare('SELECT COUNT(*) AS n FROM fact_sources').get() as { n: number };
  if (existing.n > 0) return;

  const factsWithSource = db
    .prepare(
      `SELECT COUNT(*) AS n FROM facts
       WHERE source_burst_id IS NOT NULL OR source_msg_id IS NOT NULL`
    )
    .get() as { n: number };
  if (factsWithSource.n === 0) return;

  db.exec(`
    INSERT INTO fact_sources (fact_id, source_burst_id, source_msg_id, attached_at)
    SELECT id, source_burst_id, source_msg_id, extracted_at
    FROM facts
    WHERE source_burst_id IS NOT NULL OR source_msg_id IS NOT NULL
  `);
}

function hasColumn(db: DB, table: string, column: string): boolean {
  const rows = db.prepare(`PRAGMA table_info(${table})`).all() as Array<{ name: string }>;
  return rows.some((r) => r.name === column);
}

export function upsertContact(
  db: DB,
  args: { wa_id: string; display_name?: string | null; is_group: boolean; ts: number }
): number {
  const existing = db
    .prepare('SELECT id FROM contacts WHERE wa_id = ?')
    .get(args.wa_id) as { id: number } | undefined;

  if (existing) {
    db.prepare(
      `UPDATE contacts
         SET display_name = COALESCE(?, display_name),
             last_seen_at = ?
       WHERE id = ?`
    ).run(args.display_name ?? null, args.ts, existing.id);
    return existing.id;
  }

  const result = db
    .prepare(
      `INSERT INTO contacts (wa_id, display_name, is_group, first_seen_at, last_seen_at)
       VALUES (?, ?, ?, ?, ?)`
    )
    .run(args.wa_id, args.display_name ?? null, args.is_group ? 1 : 0, args.ts, args.ts);

  return Number(result.lastInsertRowid);
}

export function insertRawMessage(db: DB, msg: RawMessageInput): number | null {
  const now = Math.floor(Date.now() / 1000);
  try {
    const result = db
      .prepare(
        `INSERT INTO raw_messages
           (wa_msg_id, contact_id, sender_wa_id, direction, body, ts,
            media_type, media_pointer, ingested_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(
        msg.wa_msg_id,
        msg.contact_id,
        msg.sender_wa_id,
        msg.direction,
        msg.body,
        msg.ts,
        msg.media_type ?? null,
        msg.media_pointer ?? null,
        now
      );
    return Number(result.lastInsertRowid);
  } catch (err: unknown) {
    if (err instanceof Error && err.message.includes('UNIQUE')) {
      return null;
    }
    throw err;
  }
}

/**
 * Assign a freshly-inserted message to a burst. If a burst exists for this
 * contact whose end_ts is within BURST_GAP_SECONDS of the message ts, extend
 * it. Otherwise open a new burst. Late messages that arrive after a burst was
 * already processed start a new burst (we do not retroactively reopen).
 */
export function assignMessageToBurst(db: DB, msgId: number): number {
  const msg = db
    .prepare(
      `SELECT id, contact_id, ts, body, media_type FROM raw_messages WHERE id = ?`
    )
    .get(msgId) as
    | { id: number; contact_id: number; ts: number; body: string; media_type: string | null }
    | undefined;

  if (!msg) throw new Error(`assignMessageToBurst: msg ${msgId} not found`);

  const candidate = db
    .prepare(
      `SELECT id, end_ts, message_count, processed_at
       FROM conversation_bursts
       WHERE contact_id = ?
       ORDER BY end_ts DESC
       LIMIT 1`
    )
    .get(msg.contact_id) as
    | { id: number; end_ts: number; message_count: number; processed_at: number | null }
    | undefined;

  let burstId: number;
  if (
    candidate &&
    candidate.processed_at === null &&
    Math.abs(msg.ts - candidate.end_ts) <= BURST_GAP_SECONDS
  ) {
    burstId = candidate.id;
    db.prepare(
      `UPDATE conversation_bursts
       SET end_ts = MAX(end_ts, ?),
           start_ts = MIN(start_ts, ?),
           message_count = message_count + 1
       WHERE id = ?`
    ).run(msg.ts, msg.ts, burstId);
  } else {
    const result = db
      .prepare(
        `INSERT INTO conversation_bursts (contact_id, start_ts, end_ts, message_count)
         VALUES (?, ?, ?, 1)`
      )
      .run(msg.contact_id, msg.ts, msg.ts);
    burstId = Number(result.lastInsertRowid);
  }

  db.prepare('UPDATE raw_messages SET burst_id = ? WHERE id = ?').run(burstId, msgId);
  return burstId;
}

/**
 * Rebuild all bursts from scratch using raw_messages. Used after backfill or
 * when the gap parameter changes. Idempotent: clears existing bursts first.
 * Only includes messages with non-empty body and no media (the same filter
 * the pipeline uses).
 */
export function rebuildAllBursts(db: DB): { bursts: number; messages: number } {
  const tx = db.transaction(() => {
    db.exec('UPDATE raw_messages SET burst_id = NULL');
    db.exec('DELETE FROM conversation_bursts');

    const messages = db
      .prepare(
        `SELECT id, contact_id, ts FROM raw_messages
         WHERE body != '' AND media_type IS NULL
         ORDER BY contact_id, ts, id`
      )
      .all() as Array<{ id: number; contact_id: number; ts: number }>;

    const insertBurst = db.prepare(
      `INSERT INTO conversation_bursts (contact_id, start_ts, end_ts, message_count)
       VALUES (?, ?, ?, ?)`
    );
    const updateBurst = db.prepare(
      `UPDATE conversation_bursts
       SET end_ts = ?, message_count = message_count + 1
       WHERE id = ?`
    );
    const setMsgBurst = db.prepare('UPDATE raw_messages SET burst_id = ? WHERE id = ?');

    let burstId: number | null = null;
    let lastContactId: number | null = null;
    let lastTs = 0;
    let burstCount = 0;
    let assigned = 0;

    for (const m of messages) {
      if (
        burstId === null ||
        m.contact_id !== lastContactId ||
        m.ts - lastTs > BURST_GAP_SECONDS
      ) {
        const r = insertBurst.run(m.contact_id, m.ts, m.ts, 1);
        burstId = Number(r.lastInsertRowid);
        burstCount++;
      } else {
        updateBurst.run(m.ts, burstId);
      }
      setMsgBurst.run(burstId, m.id);
      lastContactId = m.contact_id;
      lastTs = m.ts;
      assigned++;
    }

    return { bursts: burstCount, messages: assigned };
  });

  return tx();
}

/**
 * Bursts whose end_ts is at least BURST_GAP_SECONDS in the past (so we know no
 * more messages can join) and that haven't been processed yet. Pass
 * `includeUnsettled: true` to bypass the cutoff — useful for the eval
 * harness where every transcript is finalized at ingest time.
 */
export function listUnprocessedBursts(
  db: DB,
  limit: number,
  opts: { includeUnsettled?: boolean } = {}
): UnprocessedBurst[] {
  const cutoff = opts.includeUnsettled
    ? Math.floor(Date.now() / 1000) + 86400 // far future = include all
    : Math.floor(Date.now() / 1000) - BURST_GAP_SECONDS;
  const rows = db
    .prepare(
      `SELECT
         b.id, b.contact_id, b.start_ts, b.end_ts, b.message_count,
         c.wa_id AS chat_wa_id, c.display_name AS chat_display_name,
         c.is_group, c.notes AS chat_notes
       FROM conversation_bursts b
       JOIN contacts c ON c.id = b.contact_id
       WHERE b.processed_at IS NULL
         AND b.end_ts <= ?
         AND b.message_count > 0
       ORDER BY b.end_ts ASC
       LIMIT ?`
    )
    .all(cutoff, limit) as Array<{
    id: number;
    contact_id: number;
    start_ts: number;
    end_ts: number;
    message_count: number;
    chat_wa_id: string;
    chat_display_name: string | null;
    is_group: number;
    chat_notes: string | null;
  }>;

  return rows.map((r) => ({ ...r, is_group: r.is_group === 1 }));
}

export interface BurstQueueStats {
  total: number;
  processed: number;
}

export function getBurstQueueStats(db: DB): BurstQueueStats {
  const row = db
    .prepare(
      `SELECT
         COUNT(*) AS total,
         SUM(CASE WHEN processed_at IS NOT NULL THEN 1 ELSE 0 END) AS processed
       FROM conversation_bursts`
    )
    .get() as { total: number; processed: number | null };
  return { total: row.total, processed: row.processed ?? 0 };
}

export function getBurstMessages(db: DB, burstId: number): BurstMessage[] {
  return db
    .prepare(
      `SELECT id, sender_wa_id, direction, body, ts
       FROM raw_messages
       WHERE burst_id = ?
         AND body != ''
         AND media_type IS NULL
       ORDER BY ts ASC, id ASC`
    )
    .all(burstId) as BurstMessage[];
}

export function markBurstFiltered(db: DB, burstId: number, kept: boolean): void {
  db.prepare(`UPDATE conversation_bursts SET filter_kept = ? WHERE id = ?`).run(
    kept ? 1 : 0,
    burstId
  );
  db.prepare(`UPDATE raw_messages SET filter_kept = ? WHERE burst_id = ?`).run(
    kept ? 1 : 0,
    burstId
  );
}

export function markBurstProcessed(db: DB, burstId: number): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(`UPDATE conversation_bursts SET processed_at = ? WHERE id = ?`).run(now, burstId);
  db.prepare(`UPDATE raw_messages SET processed_at = ? WHERE burst_id = ?`).run(now, burstId);
}

export function insertFact(db: DB, fact: FactInput): number {
  const now = Math.floor(Date.now() / 1000);
  const result = db
    .prepare(
      `INSERT INTO facts
         (source_msg_id, source_burst_id, subject_wa_id, category, content, confidence, extracted_at, event_ts)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      fact.source_msg_id ?? null,
      fact.source_burst_id,
      fact.subject,
      fact.category,
      fact.content,
      fact.confidence,
      now,
      fact.event_ts ?? null
    );
  const factId = Number(result.lastInsertRowid);

  if (fact.source_burst_id !== null || (fact.source_msg_id ?? null) !== null) {
    db.prepare(
      `INSERT INTO fact_sources (fact_id, source_burst_id, source_msg_id, attached_at)
       VALUES (?, ?, ?, ?)`
    ).run(factId, fact.source_burst_id ?? null, fact.source_msg_id ?? null, now);
  }

  return factId;
}

/**
 * Attach an additional supporting source to an existing fact. Used when the
 * consolidator confirms an existing fact via a new burst — instead of inserting
 * a duplicate, we record that another piece of evidence backs it. Importance
 * (used by retrieval scoring) is the count of distinct sources.
 */
export function attachFactSource(
  db: DB,
  args: { fact_id: number; source_burst_id?: number | null; source_msg_id?: number | null }
): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO fact_sources (fact_id, source_burst_id, source_msg_id, attached_at)
     VALUES (?, ?, ?, ?)`
  ).run(args.fact_id, args.source_burst_id ?? null, args.source_msg_id ?? null, now);
}

export interface FactSourceRow {
  fact_id: number;
  source_burst_id: number | null;
  source_msg_id: number | null;
  attached_at: number;
}

export function getFactSources(db: DB, factId: number): FactSourceRow[] {
  return db
    .prepare(
      `SELECT fact_id, source_burst_id, source_msg_id, attached_at
       FROM fact_sources
       WHERE fact_id = ?
       ORDER BY attached_at ASC`
    )
    .all(factId) as FactSourceRow[];
}

export function countFactSources(db: DB, factId: number): number {
  const row = db
    .prepare('SELECT COUNT(*) AS n FROM fact_sources WHERE fact_id = ?')
    .get(factId) as { n: number };
  return row.n;
}

export function insertEmbedding(db: DB, factId: number, vector: number[]): void {
  const buf = Buffer.from(new Float32Array(vector).buffer);
  db.prepare('INSERT INTO fact_embeddings(fact_id, embedding) VALUES (?, ?)').run(
    BigInt(factId),
    buf
  );
}

function deleteEmbedding(db: DB, factId: number): void {
  db.prepare('DELETE FROM fact_embeddings WHERE fact_id = ?').run(BigInt(factId));
}

export interface ActiveFactRow {
  id: number;
  subject_wa_id: string;
  category: string;
  content: string;
  confidence: number;
  extracted_at: number;
  event_ts: number | null;
}

/**
 * Active facts for a single (subject, category) cluster, oldest first so the
 * summarizer reads them in the order they were learned. Used by the cluster
 * summary refresh and by retrieval as a typed-view fallback.
 */
export function activeFactsForCluster(
  db: DB,
  subjectWaId: string,
  category: string
): ActiveFactRow[] {
  return db
    .prepare(
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at, event_ts
       FROM facts
       WHERE subject_wa_id = ?
         AND category = ?
         AND superseded_by_id IS NULL
         AND deleted_at IS NULL
       ORDER BY extracted_at ASC, id ASC`
    )
    .all(subjectWaId, category) as ActiveFactRow[];
}

/**
 * Active facts about a subject, most recent first. Used as the "existing"
 * input to the consolidation step. Capped at `limit` to keep the LLM payload
 * bounded; for subjects with very large memory this should later be replaced
 * with vector-similarity selection against the new candidate.
 */
export function existingFactsForSubject(
  db: DB,
  subjectWaId: string,
  limit = 50
): ActiveFactRow[] {
  return db
    .prepare(
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at, event_ts
       FROM facts
       WHERE subject_wa_id = ?
         AND superseded_by_id IS NULL
         AND deleted_at IS NULL
       ORDER BY id DESC
       LIMIT ?`
    )
    .all(subjectWaId, limit) as ActiveFactRow[];
}

/**
 * Mark `oldId` as superseded by `newId` (the newly-inserted fact) and remove
 * its embedding so vector searches naturally skip it.
 */
export function markFactSuperseded(db: DB, oldId: number, newId: number): void {
  db.prepare('UPDATE facts SET superseded_by_id = ? WHERE id = ?').run(newId, oldId);
  deleteEmbedding(db, oldId);
}

/**
 * Soft-delete a fact (no replacement) and remove its embedding.
 */
export function markFactDeleted(db: DB, factId: number): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare('UPDATE facts SET deleted_at = ? WHERE id = ?').run(now, factId);
  deleteEmbedding(db, factId);
}

export interface FactRow {
  id: number;
  subject_wa_id: string;
  category: string;
  content: string;
  confidence: number;
  extracted_at: number;
  event_ts: number | null;
  source_msg_id: number | null;
  source_burst_id: number | null;
  source_body: string | null;
  source_ts: number | null;
  source_direction: 'in' | 'out' | null;
}

export interface FactSearchResult extends FactRow {
  distance: number;
}

export interface FactListFilters {
  subject?: string;
  category?: string;
  contains?: string;
}

export function listFacts(
  db: DB,
  filters: FactListFilters = {},
  limit = 200
): FactRow[] {
  const where: string[] = ['f.superseded_by_id IS NULL', 'f.deleted_at IS NULL'];
  const params: unknown[] = [];

  if (filters.subject) {
    where.push('f.subject_wa_id LIKE ?');
    params.push(`%${filters.subject}%`);
  }
  if (filters.category) {
    where.push('f.category = ?');
    params.push(filters.category);
  }
  if (filters.contains) {
    where.push('f.content LIKE ?');
    params.push(`%${filters.contains}%`);
  }

  const whereClause = `WHERE ${where.join(' AND ')}`;
  const sql = `
    SELECT
      f.id, f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at, f.event_ts,
      f.source_msg_id, f.source_burst_id,
      COALESCE(rm.body, fbm.body)             AS source_body,
      COALESCE(rm.ts,   b.start_ts)           AS source_ts,
      COALESCE(rm.direction, fbm.direction)   AS source_direction
    FROM facts f
    LEFT JOIN raw_messages rm ON rm.id = f.source_msg_id
    LEFT JOIN conversation_bursts b ON b.id = f.source_burst_id
    LEFT JOIN raw_messages fbm ON fbm.id = (
      SELECT id FROM raw_messages
      WHERE burst_id = f.source_burst_id AND body != ''
      ORDER BY ts ASC, id ASC LIMIT 1
    )
    ${whereClause}
    ORDER BY f.id DESC
    LIMIT ?
  `;
  params.push(limit);
  return db.prepare(sql).all(...params) as FactRow[];
}

export function searchFactsByVector(
  db: DB,
  vector: number[],
  k: number
): FactSearchResult[] {
  const buf = Buffer.from(new Float32Array(vector).buffer);
  return db
    .prepare(
      `SELECT
         fe.fact_id AS id,
         fe.distance AS distance,
         f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at, f.event_ts,
         f.source_msg_id, f.source_burst_id,
         COALESCE(rm.body, fbm.body)             AS source_body,
         COALESCE(rm.ts,   b.start_ts)           AS source_ts,
         COALESCE(rm.direction, fbm.direction)   AS source_direction
       FROM fact_embeddings fe
       JOIN facts f ON f.id = fe.fact_id
       LEFT JOIN raw_messages rm ON rm.id = f.source_msg_id
       LEFT JOIN conversation_bursts b ON b.id = f.source_burst_id
       LEFT JOIN raw_messages fbm ON fbm.id = (
         SELECT id FROM raw_messages
         WHERE burst_id = f.source_burst_id AND body != ''
         ORDER BY ts ASC, id ASC LIMIT 1
       )
       WHERE fe.embedding MATCH ?
         AND k = ?
         AND f.superseded_by_id IS NULL
         AND f.deleted_at IS NULL
       ORDER BY fe.distance`
    )
    .all(buf, k) as FactSearchResult[];
}

/**
 * Vector search restricted to one subject. Used by the consolidator to pick
 * the most-relevant existing facts to compare against a new candidate, when
 * the subject has too many active facts to send them all to the LLM.
 *
 * The vec0 MATCH pre-filter returns up to `overFetch` global hits ordered by
 * distance; the JOIN then narrows to the requested subject. If the subject's
 * facts don't appear in the over-fetched window the result is short or empty
 * — the caller should fall back to recency-based selection.
 */
export function searchFactsForSubjectByVector(
  db: DB,
  vector: number[],
  subjectWaId: string,
  k: number,
  overFetch = 200
): ActiveFactRow[] {
  const buf = Buffer.from(new Float32Array(vector).buffer);
  return db
    .prepare(
      `SELECT
         fe.fact_id AS id,
         f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at, f.event_ts
       FROM fact_embeddings fe
       JOIN facts f ON f.id = fe.fact_id
       WHERE fe.embedding MATCH ?
         AND k = ?
         AND f.subject_wa_id = ?
         AND f.superseded_by_id IS NULL
         AND f.deleted_at IS NULL
       ORDER BY fe.distance
       LIMIT ?`
    )
    .all(buf, overFetch, subjectWaId, k) as ActiveFactRow[];
}

export function countActiveFactsForSubject(db: DB, subjectWaId: string): number {
  const row = db
    .prepare(
      `SELECT COUNT(*) AS n
       FROM facts
       WHERE subject_wa_id = ?
         AND superseded_by_id IS NULL
         AND deleted_at IS NULL`
    )
    .get(subjectWaId) as { n: number };
  return row.n;
}

export function listCategories(db: DB): string[] {
  const rows = db
    .prepare('SELECT DISTINCT category FROM facts ORDER BY category')
    .all() as Array<{ category: string }>;
  return rows.map((r) => r.category);
}

export function setLiveCutoff(db: DB, contactId: number, ts: number): void {
  db.prepare('UPDATE contacts SET live_cutoff_ts = ? WHERE id = ?').run(ts, contactId);
}

export function getLiveCutoff(db: DB, contactId: number): number | null {
  const row = db
    .prepare('SELECT live_cutoff_ts FROM contacts WHERE id = ?')
    .get(contactId) as { live_cutoff_ts: number | null } | undefined;
  return row?.live_cutoff_ts ?? null;
}

export function logProcessing(db: DB, args: ProcessingLogInput): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO processing_log (msg_id, burst_id, stage, model, tokens_in, tokens_out, ts)
     VALUES (?, ?, ?, ?, ?, ?, ?)`
  ).run(
    args.msg_id ?? null,
    args.burst_id ?? null,
    args.stage,
    args.model,
    args.tokens_in,
    args.tokens_out,
    now
  );
}

export interface ClusterSummaryRow {
  id: number;
  subject_wa_id: string;
  category: string;
  summary: string;
  fact_count: number;
  fact_ids: number[];
  last_refreshed_at: number;
}

export function upsertClusterSummary(
  db: DB,
  args: {
    subject_wa_id: string;
    category: string;
    summary: string;
    fact_ids: number[];
  }
): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO cluster_summaries
       (subject_wa_id, category, summary, fact_count, fact_ids_json, last_refreshed_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(subject_wa_id, category) DO UPDATE SET
       summary = excluded.summary,
       fact_count = excluded.fact_count,
       fact_ids_json = excluded.fact_ids_json,
       last_refreshed_at = excluded.last_refreshed_at`
  ).run(
    args.subject_wa_id,
    args.category,
    args.summary,
    args.fact_ids.length,
    JSON.stringify(args.fact_ids),
    now
  );
}

export function deleteClusterSummary(
  db: DB,
  subjectWaId: string,
  category: string
): void {
  db.prepare(
    'DELETE FROM cluster_summaries WHERE subject_wa_id = ? AND category = ?'
  ).run(subjectWaId, category);
}

function rowToCluster(r: {
  id: number;
  subject_wa_id: string;
  category: string;
  summary: string;
  fact_count: number;
  fact_ids_json: string;
  last_refreshed_at: number;
}): ClusterSummaryRow {
  return {
    id: r.id,
    subject_wa_id: r.subject_wa_id,
    category: r.category,
    summary: r.summary,
    fact_count: r.fact_count,
    fact_ids: JSON.parse(r.fact_ids_json) as number[],
    last_refreshed_at: r.last_refreshed_at,
  };
}

export function getClusterSummary(
  db: DB,
  subjectWaId: string,
  category: string
): ClusterSummaryRow | null {
  const row = db
    .prepare(
      `SELECT id, subject_wa_id, category, summary, fact_count, fact_ids_json, last_refreshed_at
       FROM cluster_summaries
       WHERE subject_wa_id = ? AND category = ?`
    )
    .get(subjectWaId, category) as
    | {
        id: number;
        subject_wa_id: string;
        category: string;
        summary: string;
        fact_count: number;
        fact_ids_json: string;
        last_refreshed_at: number;
      }
    | undefined;
  return row ? rowToCluster(row) : null;
}

export function listClusterSummariesForSubject(
  db: DB,
  subjectWaId: string
): ClusterSummaryRow[] {
  const rows = db
    .prepare(
      `SELECT id, subject_wa_id, category, summary, fact_count, fact_ids_json, last_refreshed_at
       FROM cluster_summaries
       WHERE subject_wa_id = ?
       ORDER BY category`
    )
    .all(subjectWaId) as Array<{
    id: number;
    subject_wa_id: string;
    category: string;
    summary: string;
    fact_count: number;
    fact_ids_json: string;
    last_refreshed_at: number;
  }>;
  return rows.map(rowToCluster);
}

export function listClusterSummariesForSubjects(
  db: DB,
  subjectWaIds: string[]
): ClusterSummaryRow[] {
  if (subjectWaIds.length === 0) return [];
  const placeholders = subjectWaIds.map(() => '?').join(',');
  const rows = db
    .prepare(
      `SELECT id, subject_wa_id, category, summary, fact_count, fact_ids_json, last_refreshed_at
       FROM cluster_summaries
       WHERE subject_wa_id IN (${placeholders})
       ORDER BY subject_wa_id, category`
    )
    .all(...subjectWaIds) as Array<{
    id: number;
    subject_wa_id: string;
    category: string;
    summary: string;
    fact_count: number;
    fact_ids_json: string;
    last_refreshed_at: number;
  }>;
  return rows.map(rowToCluster);
}

/**
 * All active preference-category facts, optionally narrowed to one subject.
 * Used by composeMemoryBlock as the always-on section: every chat turn includes
 * preferences regardless of similarity to the question.
 */
export function activePreferences(db: DB, subjectWaId?: string): ActiveFactRow[] {
  const where = ['superseded_by_id IS NULL', 'deleted_at IS NULL', "category = 'preference'"];
  const params: unknown[] = [];
  if (subjectWaId) {
    where.push('subject_wa_id = ?');
    params.push(subjectWaId);
  }
  return db
    .prepare(
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at, event_ts
       FROM facts
       WHERE ${where.join(' AND ')}
       ORDER BY confidence DESC, extracted_at DESC`
    )
    .all(...params) as ActiveFactRow[];
}

/**
 * Recent + upcoming episode-shaped facts (event + commitment categories).
 * The temporal window applies to event_ts when set (so future events with no
 * event_ts fall back to extracted_at); the past-side window is `days` days.
 *
 * Returned ordering: future events first, soonest at the top, then past events
 * most-recent-first. This shape lets retrieval split into <upcoming_events>
 * and <recent_past> sections without re-sorting in JS.
 */
export function recentEpisodes(db: DB, days: number): ActiveFactRow[] {
  const now = Math.floor(Date.now() / 1000);
  const pastCutoff = now - days * 86400;
  return db
    .prepare(
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at, event_ts
       FROM facts
       WHERE superseded_by_id IS NULL
         AND deleted_at IS NULL
         AND category IN ('event', 'commitment')
         AND COALESCE(event_ts, extracted_at) >= ?
       ORDER BY
         -- 0 = future, 1 = past so future bubbles up.
         CASE WHEN COALESCE(event_ts, extracted_at) >= ? THEN 0 ELSE 1 END,
         -- Within future, soonest first; within past, most-recent first.
         CASE WHEN COALESCE(event_ts, extracted_at) >= ?
              THEN COALESCE(event_ts, extracted_at)
              ELSE -COALESCE(event_ts, extracted_at)
         END`
    )
    .all(pastCutoff, now, now) as ActiveFactRow[];
}

export interface SubjectInfo {
  subject_wa_id: string;
  display_name: string | null;
  active_fact_count: number;
  /** Most-recent extracted_at across active facts; null if no active facts. */
  last_updated_ts: number | null;
}

/**
 * All subjects that have at least one active fact, with their contact display
 * name (if a matching contact exists). Used for entity matching, the
 * subject-list sidebar, and any "what people do we know about" report.
 */
export function allActiveSubjects(db: DB): SubjectInfo[] {
  return db
    .prepare(
      `SELECT
         f.subject_wa_id                   AS subject_wa_id,
         c.display_name                    AS display_name,
         COUNT(*)                          AS active_fact_count,
         MAX(f.extracted_at)               AS last_updated_ts
       FROM facts f
       LEFT JOIN contacts c ON c.wa_id = f.subject_wa_id
       WHERE f.superseded_by_id IS NULL AND f.deleted_at IS NULL
       GROUP BY f.subject_wa_id, c.display_name
       ORDER BY active_fact_count DESC`
    )
    .all() as SubjectInfo[];
}

/** Single subject lookup with the same shape as allActiveSubjects entries. */
export function getSubjectInfo(db: DB, subjectWaId: string): SubjectInfo | null {
  const row = db
    .prepare(
      `SELECT
         f.subject_wa_id                   AS subject_wa_id,
         c.display_name                    AS display_name,
         COUNT(*)                          AS active_fact_count,
         MAX(f.extracted_at)               AS last_updated_ts
       FROM facts f
       LEFT JOIN contacts c ON c.wa_id = f.subject_wa_id
       WHERE f.subject_wa_id = ?
         AND f.superseded_by_id IS NULL
         AND f.deleted_at IS NULL
       GROUP BY f.subject_wa_id, c.display_name`
    )
    .get(subjectWaId) as SubjectInfo | undefined;
  return row && row.active_fact_count > 0 ? row : null;
}

export interface SupersededFactRow {
  id: number;
  subject_wa_id: string;
  category: string;
  content: string;
  confidence: number;
  extracted_at: number;
  event_ts: number | null;
  superseded_by_id: number | null;
  deleted_at: number | null;
  /** When superseded, the replacement fact's content (null when soft-deleted). */
  replacement_content: string | null;
  replacement_id: number | null;
}

/**
 * Historical facts for a subject — the "previously believed" trail. Includes
 * both superseded facts (have a replacement) and soft-deleted facts (no
 * replacement). Most recent first so the UI shows the most recent change at
 * the top.
 */
export function supersededFactsForSubject(
  db: DB,
  subjectWaId: string,
  limit = 200
): SupersededFactRow[] {
  return db
    .prepare(
      `SELECT
         f.id, f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at, f.event_ts,
         f.superseded_by_id, f.deleted_at,
         repl.content AS replacement_content,
         repl.id      AS replacement_id
       FROM facts f
       LEFT JOIN facts repl ON repl.id = f.superseded_by_id
       WHERE f.subject_wa_id = ?
         AND (f.superseded_by_id IS NOT NULL OR f.deleted_at IS NOT NULL)
       ORDER BY f.extracted_at DESC
       LIMIT ?`
    )
    .all(subjectWaId, limit) as SupersededFactRow[];
}

/**
 * Active facts for a subject WITH the same source-detail JOIN as listFacts —
 * used by the subject detail endpoint so the UI can render source bodies and
 * timestamps next to each fact, the same way the global Facts view does.
 */
export function factsAboutSubjectWithSource(db: DB, subjectWaId: string): FactRow[] {
  return db
    .prepare(
      `SELECT
         f.id, f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at, f.event_ts,
         f.source_msg_id, f.source_burst_id,
         COALESCE(rm.body, fbm.body)             AS source_body,
         COALESCE(rm.ts,   b.start_ts)           AS source_ts,
         COALESCE(rm.direction, fbm.direction)   AS source_direction
       FROM facts f
       LEFT JOIN raw_messages rm ON rm.id = f.source_msg_id
       LEFT JOIN conversation_bursts b ON b.id = f.source_burst_id
       LEFT JOIN raw_messages fbm ON fbm.id = (
         SELECT id FROM raw_messages
         WHERE burst_id = f.source_burst_id AND body != ''
         ORDER BY ts ASC, id ASC LIMIT 1
       )
       WHERE f.subject_wa_id = ?
         AND f.superseded_by_id IS NULL
         AND f.deleted_at IS NULL
       ORDER BY f.category, f.confidence DESC, f.extracted_at DESC`
    )
    .all(subjectWaId) as FactRow[];
}

/**
 * Active facts about a specific subject. Caps at `limit` (default 100) to keep
 * retrieval-time payloads bounded; the hybrid scorer filters and trims further.
 */
export function factsAboutSubject(
  db: DB,
  subjectWaId: string,
  limit = 100
): ActiveFactRow[] {
  return db
    .prepare(
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at, event_ts
       FROM facts
       WHERE subject_wa_id = ?
         AND superseded_by_id IS NULL
         AND deleted_at IS NULL
       ORDER BY confidence DESC, extracted_at DESC
       LIMIT ?`
    )
    .all(subjectWaId, limit) as ActiveFactRow[];
}

export interface GraphEntityInput {
  entity_type: string;
  canonical_key: string;
  display_name: string;
  aliases?: string[];
  confidence: number;
}

export interface GraphEntityRow {
  id: number;
  entity_type: string;
  canonical_key: string;
  display_name: string;
  aliases: string[];
  confidence: number;
  created_at: number;
  updated_at: number;
  merged_into_id: number | null;
}

export interface FactEntityMentionInput {
  fact_id: number;
  entity_id: number;
  role: string;
  mention_text?: string | null;
  confidence: number;
}

export interface FactEntityMentionRow {
  id: number;
  fact_id: number;
  entity_id: number;
  role: string;
  mention_text: string | null;
  confidence: number;
  created_at: number;
}

export interface KnowledgeEdgeInput {
  source_entity_id: number;
  predicate: string;
  target_entity_id: number;
  confidence: number;
  source_fact_id?: number | null;
  source_burst_id?: number | null;
  event_ts?: number | null;
  valid_from_ts?: number | null;
  valid_to_ts?: number | null;
  qualifiers?: Record<string, unknown>;
}

export interface KnowledgeEdgeRow {
  id: number;
  source_entity_id: number;
  source_display_name: string;
  source_entity_type: string;
  predicate: string;
  target_entity_id: number;
  target_display_name: string;
  target_entity_type: string;
  confidence: number;
  source_fact_id: number | null;
  source_burst_id: number | null;
  extracted_at: number;
  event_ts: number | null;
  valid_from_ts: number | null;
  valid_to_ts: number | null;
  status: string;
  superseded_by_edge_id: number | null;
  deleted_at: number | null;
  qualifiers: Record<string, unknown>;
}

export interface GraphFactRow extends ActiveFactRow {
  source_burst_id: number | null;
}

export interface GraphBuildStats {
  entities: number;
  mentions: number;
  edges: number;
}

export function clearGraphTables(db: DB): void {
  db.transaction(() => {
    db.exec('DELETE FROM edge_sources;');
    db.exec('DELETE FROM knowledge_edges;');
    db.exec('DELETE FROM fact_entity_mentions;');
    db.exec('DELETE FROM entities;');
  })();
}

export function listActiveFactsForGraph(
  db: DB,
  opts: { subject?: string; limit?: number } = {}
): GraphFactRow[] {
  const where = ['superseded_by_id IS NULL', 'deleted_at IS NULL'];
  const params: unknown[] = [];
  if (opts.subject) {
    where.push('subject_wa_id = ?');
    params.push(opts.subject);
  }
  const limit = opts.limit ?? 1000;
  params.push(limit);
  return db
    .prepare(
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at, event_ts, source_burst_id
       FROM facts
       WHERE ${where.join(' AND ')}
       ORDER BY id ASC
       LIMIT ?`
    )
    .all(...params) as GraphFactRow[];
}

export function upsertEntity(db: DB, input: GraphEntityInput): number {
  const now = Math.floor(Date.now() / 1000);
  const aliases = JSON.stringify(input.aliases ?? []);
  db.prepare(
    `INSERT INTO entities
       (entity_type, canonical_key, display_name, aliases_json, confidence, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(canonical_key) DO UPDATE SET
       display_name = CASE
         WHEN length(excluded.display_name) > length(entities.display_name)
         THEN excluded.display_name
         ELSE entities.display_name
       END,
       aliases_json = excluded.aliases_json,
       confidence = MAX(entities.confidence, excluded.confidence),
       updated_at = excluded.updated_at`
  ).run(
    input.entity_type,
    input.canonical_key,
    input.display_name,
    aliases,
    input.confidence,
    now,
    now
  );

  const row = db
    .prepare('SELECT id FROM entities WHERE canonical_key = ?')
    .get(input.canonical_key) as { id: number };
  return row.id;
}

export function insertFactEntityMention(db: DB, input: FactEntityMentionInput): number {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO fact_entity_mentions
       (fact_id, entity_id, role, mention_text, confidence, created_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(fact_id, entity_id, role) DO UPDATE SET
       mention_text = COALESCE(excluded.mention_text, fact_entity_mentions.mention_text),
       confidence = MAX(fact_entity_mentions.confidence, excluded.confidence)`
  ).run(
    input.fact_id,
    input.entity_id,
    input.role,
    input.mention_text ?? null,
    input.confidence,
    now
  );

  const row = db
    .prepare(
      `SELECT id FROM fact_entity_mentions
       WHERE fact_id = ? AND entity_id = ? AND role = ?`
    )
    .get(input.fact_id, input.entity_id, input.role) as { id: number };
  return row.id;
}

export function upsertKnowledgeEdge(db: DB, input: KnowledgeEdgeInput): number {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO knowledge_edges
       (source_entity_id, predicate, target_entity_id, confidence, source_fact_id, source_burst_id,
        extracted_at, event_ts, valid_from_ts, valid_to_ts, qualifiers_json)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(source_entity_id, predicate, target_entity_id) DO UPDATE SET
       confidence = MAX(knowledge_edges.confidence, excluded.confidence),
       source_fact_id = COALESCE(knowledge_edges.source_fact_id, excluded.source_fact_id),
       source_burst_id = COALESCE(knowledge_edges.source_burst_id, excluded.source_burst_id),
       event_ts = COALESCE(knowledge_edges.event_ts, excluded.event_ts),
       valid_from_ts = COALESCE(knowledge_edges.valid_from_ts, excluded.valid_from_ts),
       valid_to_ts = COALESCE(knowledge_edges.valid_to_ts, excluded.valid_to_ts),
       status = 'active',
       deleted_at = NULL`
  ).run(
    input.source_entity_id,
    input.predicate,
    input.target_entity_id,
    input.confidence,
    input.source_fact_id ?? null,
    input.source_burst_id ?? null,
    now,
    input.event_ts ?? null,
    input.valid_from_ts ?? null,
    input.valid_to_ts ?? null,
    JSON.stringify(input.qualifiers ?? {})
  );

  const row = db
    .prepare(
      `SELECT id FROM knowledge_edges
       WHERE source_entity_id = ? AND predicate = ? AND target_entity_id = ?`
    )
    .get(input.source_entity_id, input.predicate, input.target_entity_id) as { id: number };

  if (input.source_fact_id !== null && input.source_fact_id !== undefined) {
    attachEdgeSource(db, {
      edge_id: row.id,
      fact_id: input.source_fact_id,
      burst_id: input.source_burst_id ?? null,
    });
  }

  return row.id;
}

export function attachEdgeSource(
  db: DB,
  input: { edge_id: number; fact_id: number; burst_id?: number | null }
): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO edge_sources (edge_id, fact_id, burst_id, attached_at)
     VALUES (?, ?, ?, ?)
     ON CONFLICT(edge_id, fact_id) DO UPDATE SET
       burst_id = COALESCE(excluded.burst_id, edge_sources.burst_id),
       attached_at = edge_sources.attached_at`
  ).run(input.edge_id, input.fact_id, input.burst_id ?? null, now);
}

export function deactivateGraphForFact(db: DB, factId: number, status: 'deleted' | 'superseded'): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare('DELETE FROM fact_entity_mentions WHERE fact_id = ?').run(factId);
  db.prepare('DELETE FROM edge_sources WHERE fact_id = ?').run(factId);
  db.prepare(
    `UPDATE knowledge_edges
     SET status = ?, deleted_at = ?
     WHERE source_fact_id = ?
       AND NOT EXISTS (
         SELECT 1 FROM edge_sources es WHERE es.edge_id = knowledge_edges.id
       )`
  ).run(status, now, factId);
}

export function graphCounts(db: DB): GraphBuildStats {
  const entities = db.prepare('SELECT COUNT(*) AS n FROM entities').get() as { n: number };
  const mentions = db
    .prepare('SELECT COUNT(*) AS n FROM fact_entity_mentions')
    .get() as { n: number };
  const edges = db
    .prepare("SELECT COUNT(*) AS n FROM knowledge_edges WHERE status = 'active' AND deleted_at IS NULL")
    .get() as { n: number };
  return { entities: entities.n, mentions: mentions.n, edges: edges.n };
}

function rowToGraphEntity(r: {
  id: number;
  entity_type: string;
  canonical_key: string;
  display_name: string;
  aliases_json: string;
  confidence: number;
  created_at: number;
  updated_at: number;
  merged_into_id: number | null;
}): GraphEntityRow {
  return {
    id: r.id,
    entity_type: r.entity_type,
    canonical_key: r.canonical_key,
    display_name: r.display_name,
    aliases: JSON.parse(r.aliases_json) as string[],
    confidence: r.confidence,
    created_at: r.created_at,
    updated_at: r.updated_at,
    merged_into_id: r.merged_into_id,
  };
}

export function searchEntities(db: DB, query: string, limit = 20): GraphEntityRow[] {
  const rows = db
    .prepare(
      `SELECT id, entity_type, canonical_key, display_name, aliases_json, confidence,
              created_at, updated_at, merged_into_id
       FROM entities
       WHERE merged_into_id IS NULL
         AND (display_name LIKE ? OR canonical_key LIKE ? OR aliases_json LIKE ?)
       ORDER BY confidence DESC, length(display_name) ASC
       LIMIT ?`
    )
    .all(`%${query}%`, `%${query}%`, `%${query}%`, limit) as Array<Parameters<typeof rowToGraphEntity>[0]>;
  return rows.map(rowToGraphEntity);
}

function rowToKnowledgeEdge(r: Omit<KnowledgeEdgeRow, 'qualifiers'> & { qualifiers_json: string }): KnowledgeEdgeRow {
  return {
    ...r,
    qualifiers: JSON.parse(r.qualifiers_json) as Record<string, unknown>,
  };
}

export function graphNeighborhood(
  db: DB,
  entityId: number,
  opts: { limit?: number } = {}
): KnowledgeEdgeRow[] {
  const limit = opts.limit ?? 100;
  const rows = db
    .prepare(
      `SELECT
         e.id,
         e.source_entity_id,
         s.display_name AS source_display_name,
         s.entity_type AS source_entity_type,
         e.predicate,
         e.target_entity_id,
         t.display_name AS target_display_name,
         t.entity_type AS target_entity_type,
         e.confidence,
         e.source_fact_id,
         e.source_burst_id,
         e.extracted_at,
         e.event_ts,
         e.valid_from_ts,
         e.valid_to_ts,
         e.status,
         e.superseded_by_edge_id,
         e.deleted_at,
         e.qualifiers_json
       FROM knowledge_edges e
       JOIN entities s ON s.id = e.source_entity_id
       JOIN entities t ON t.id = e.target_entity_id
       WHERE (e.source_entity_id = ? OR e.target_entity_id = ?)
         AND e.status = 'active'
         AND e.deleted_at IS NULL
       ORDER BY e.confidence DESC, e.id DESC
       LIMIT ?`
    )
    .all(entityId, entityId, limit) as Array<Omit<KnowledgeEdgeRow, 'qualifiers'> & { qualifiers_json: string }>;
  return rows.map(rowToKnowledgeEdge);
}

export interface GraphEntityWithStatsRow extends GraphEntityRow {
  outgoing_edge_count: number;
  incoming_edge_count: number;
}

export function listGraphEntitiesWithStats(db: DB, limit = 500): GraphEntityWithStatsRow[] {
  const rows = db
    .prepare(
      `SELECT
         e.id, e.entity_type, e.canonical_key, e.display_name, e.aliases_json,
         e.confidence, e.created_at, e.updated_at, e.merged_into_id,
         COUNT(DISTINCT out_e.id) AS outgoing_edge_count,
         COUNT(DISTINCT in_e.id) AS incoming_edge_count
       FROM entities e
       LEFT JOIN knowledge_edges out_e
         ON out_e.source_entity_id = e.id
        AND out_e.status = 'active'
        AND out_e.deleted_at IS NULL
       LEFT JOIN knowledge_edges in_e
         ON in_e.target_entity_id = e.id
        AND in_e.status = 'active'
        AND in_e.deleted_at IS NULL
       WHERE e.merged_into_id IS NULL
       GROUP BY e.id
       ORDER BY (COUNT(DISTINCT out_e.id) + COUNT(DISTINCT in_e.id)) DESC,
                e.confidence DESC,
                e.display_name ASC
       LIMIT ?`
    )
    .all(limit) as Array<Parameters<typeof rowToGraphEntity>[0] & {
    outgoing_edge_count: number;
    incoming_edge_count: number;
  }>;
  return rows.map((r) => ({
    ...rowToGraphEntity(r),
    outgoing_edge_count: r.outgoing_edge_count,
    incoming_edge_count: r.incoming_edge_count,
  }));
}

export function listKnowledgeEdges(db: DB, limit = 1000): KnowledgeEdgeRow[] {
  const rows = db
    .prepare(
      `SELECT
         e.id,
         e.source_entity_id,
         s.display_name AS source_display_name,
         s.entity_type AS source_entity_type,
         e.predicate,
         e.target_entity_id,
         t.display_name AS target_display_name,
         t.entity_type AS target_entity_type,
         e.confidence,
         e.source_fact_id,
         e.source_burst_id,
         e.extracted_at,
         e.event_ts,
         e.valid_from_ts,
         e.valid_to_ts,
         e.status,
         e.superseded_by_edge_id,
         e.deleted_at,
         e.qualifiers_json
       FROM knowledge_edges e
       JOIN entities s ON s.id = e.source_entity_id
       JOIN entities t ON t.id = e.target_entity_id
       WHERE e.status = 'active'
         AND e.deleted_at IS NULL
       ORDER BY e.confidence DESC, e.id DESC
       LIMIT ?`
    )
    .all(limit) as Array<Omit<KnowledgeEdgeRow, 'qualifiers'> & { qualifiers_json: string }>;
  return rows.map(rowToKnowledgeEdge);
}

export function graphForFact(db: DB, factId: number): {
  mentions: Array<FactEntityMentionRow & { entity: GraphEntityRow }>;
  edges: KnowledgeEdgeRow[];
} {
  const mentionRows = db
    .prepare(
      `SELECT
         m.id, m.fact_id, m.entity_id, m.role, m.mention_text, m.confidence, m.created_at,
         e.entity_type, e.canonical_key, e.display_name, e.aliases_json,
         e.confidence AS entity_confidence, e.created_at AS entity_created_at,
         e.updated_at, e.merged_into_id
       FROM fact_entity_mentions m
       JOIN entities e ON e.id = m.entity_id
       WHERE m.fact_id = ?
       ORDER BY m.role, e.display_name`
    )
    .all(factId) as Array<{
    id: number;
    fact_id: number;
    entity_id: number;
    role: string;
    mention_text: string | null;
    confidence: number;
    created_at: number;
    entity_type: string;
    canonical_key: string;
    display_name: string;
    aliases_json: string;
    entity_confidence: number;
    entity_created_at: number;
    updated_at: number;
    merged_into_id: number | null;
  }>;

  const edges = db
    .prepare(
      `SELECT
         e.id,
         e.source_entity_id,
         s.display_name AS source_display_name,
         s.entity_type AS source_entity_type,
         e.predicate,
         e.target_entity_id,
         t.display_name AS target_display_name,
         t.entity_type AS target_entity_type,
         e.confidence,
         e.source_fact_id,
         e.source_burst_id,
         e.extracted_at,
         e.event_ts,
         e.valid_from_ts,
         e.valid_to_ts,
         e.status,
         e.superseded_by_edge_id,
         e.deleted_at,
         e.qualifiers_json
       FROM knowledge_edges e
       JOIN entities s ON s.id = e.source_entity_id
       JOIN entities t ON t.id = e.target_entity_id
       JOIN edge_sources es ON es.edge_id = e.id
       WHERE es.fact_id = ?
       ORDER BY e.predicate, s.display_name, t.display_name`
    )
    .all(factId) as Array<Omit<KnowledgeEdgeRow, 'qualifiers'> & { qualifiers_json: string }>;

  return {
    mentions: mentionRows.map((m) => ({
      id: m.id,
      fact_id: m.fact_id,
      entity_id: m.entity_id,
      role: m.role,
      mention_text: m.mention_text,
      confidence: m.confidence,
      created_at: m.created_at,
      entity: rowToGraphEntity({
        id: m.entity_id,
        entity_type: m.entity_type,
        canonical_key: m.canonical_key,
        display_name: m.display_name,
        aliases_json: m.aliases_json,
        confidence: m.entity_confidence,
        created_at: m.entity_created_at,
        updated_at: m.updated_at,
        merged_into_id: m.merged_into_id,
      }),
    })),
    edges: edges.map(rowToKnowledgeEdge),
  };
}
