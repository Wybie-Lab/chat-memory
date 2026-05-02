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
}

export interface ProcessingLogInput {
  msg_id?: number | null;
  burst_id?: number | null;
  stage: 'filter' | 'extract' | 'embed';
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

  db.exec('CREATE INDEX IF NOT EXISTS idx_raw_burst ON raw_messages(burst_id)');
  db.exec('CREATE INDEX IF NOT EXISTS idx_raw_unburst ON raw_messages(contact_id, ts) WHERE burst_id IS NULL');
  db.exec('CREATE INDEX IF NOT EXISTS idx_facts_burst ON facts(source_burst_id)');
  db.exec(`CREATE INDEX IF NOT EXISTS idx_facts_active
           ON facts(subject_wa_id)
           WHERE superseded_by_id IS NULL AND deleted_at IS NULL`);
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
 * more messages can join) and that haven't been processed yet.
 */
export function listUnprocessedBursts(db: DB, limit: number): UnprocessedBurst[] {
  const cutoff = Math.floor(Date.now() / 1000) - BURST_GAP_SECONDS;
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
         (source_msg_id, source_burst_id, subject_wa_id, category, content, confidence, extracted_at)
       VALUES (?, ?, ?, ?, ?, ?, ?)`
    )
    .run(
      fact.source_msg_id ?? null,
      fact.source_burst_id,
      fact.subject,
      fact.category,
      fact.content,
      fact.confidence,
      now
    );
  return Number(result.lastInsertRowid);
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
      `SELECT id, subject_wa_id, category, content, confidence, extracted_at
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
      f.id, f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at,
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
         f.subject_wa_id, f.category, f.content, f.confidence, f.extracted_at,
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
