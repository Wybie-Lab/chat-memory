import Database from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import fs from 'fs';
import path from 'path';

export type DB = Database.Database;

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
  source_msg_id: number;
  subject: string;
  category: string;
  content: string;
  confidence: number;
}

export interface ProcessingLogInput {
  msg_id: number;
  stage: 'filter' | 'extract' | 'embed';
  model: string;
  tokens_in: number;
  tokens_out: number;
}

export interface UnprocessedMessage {
  id: number;
  contact_id: number;
  sender_wa_id: string;
  direction: 'in' | 'out';
  body: string;
  ts: number;
  chat_wa_id: string;
  chat_display_name: string | null;
  is_group: boolean;
  chat_notes: string | null;
}

export function openDb(dbPath: string): DB {
  fs.mkdirSync(path.dirname(dbPath), { recursive: true });
  const db = new Database(dbPath);

  sqliteVec.load(db);

  const schemaPath = path.join(__dirname, 'schema.sql');
  const schema = fs.readFileSync(schemaPath, 'utf-8');
  db.exec(schema);

  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS fact_embeddings USING vec0(
      fact_id INTEGER PRIMARY KEY,
      embedding FLOAT[1024]
    );
  `);

  return db;
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

export function getRecentContext(
  db: DB,
  contactId: number,
  beforeMsgId: number,
  limit: number
): Array<{ direction: 'in' | 'out'; body: string; ts: number }> {
  const rows = db
    .prepare(
      `SELECT direction, body, ts
       FROM raw_messages
       WHERE contact_id = ?
         AND id < ?
         AND body != ''
         AND media_type IS NULL
       ORDER BY ts DESC, id DESC
       LIMIT ?`
    )
    .all(contactId, beforeMsgId, limit) as Array<{
    direction: 'in' | 'out';
    body: string;
    ts: number;
  }>;
  return rows.reverse();
}

export function listUnprocessedMessages(db: DB, limit: number): UnprocessedMessage[] {
  const rows = db
    .prepare(
      `SELECT
         rm.id, rm.contact_id, rm.sender_wa_id, rm.direction, rm.body, rm.ts,
         c.wa_id AS chat_wa_id, c.display_name AS chat_display_name,
         c.is_group, c.notes AS chat_notes
       FROM raw_messages rm
       JOIN contacts c ON c.id = rm.contact_id
       WHERE rm.filter_kept IS NULL
         AND rm.body != ''
         AND rm.media_type IS NULL
       ORDER BY rm.ts ASC
       LIMIT ?`
    )
    .all(limit) as Array<{
    id: number;
    contact_id: number;
    sender_wa_id: string;
    direction: 'in' | 'out';
    body: string;
    ts: number;
    chat_wa_id: string;
    chat_display_name: string | null;
    is_group: number;
    chat_notes: string | null;
  }>;

  return rows.map((r) => ({
    ...r,
    is_group: r.is_group === 1,
  }));
}

export function markFiltered(
  db: DB,
  msgId: number,
  kept: boolean
): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `UPDATE raw_messages SET filter_kept = ?, processed_at = ? WHERE id = ?`
  ).run(kept ? 1 : 0, now, msgId);
}

export function insertFact(db: DB, fact: FactInput): number {
  const now = Math.floor(Date.now() / 1000);
  const result = db
    .prepare(
      `INSERT INTO facts (source_msg_id, subject_wa_id, category, content, confidence, extracted_at)
       VALUES (?, ?, ?, ?, ?, ?)`
    )
    .run(
      fact.source_msg_id,
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

export function logProcessing(db: DB, args: ProcessingLogInput): void {
  const now = Math.floor(Date.now() / 1000);
  db.prepare(
    `INSERT INTO processing_log (msg_id, stage, model, tokens_in, tokens_out, ts)
     VALUES (?, ?, ?, ?, ?, ?)`
  ).run(args.msg_id, args.stage, args.model, args.tokens_in, args.tokens_out, now);
}
