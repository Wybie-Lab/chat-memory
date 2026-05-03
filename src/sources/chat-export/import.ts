import crypto from 'crypto';
import {
  upsertContact,
  insertRawMessage,
  rebuildAllBursts,
  setLiveCutoff,
  type DB,
} from '../../engine';
import { parseExport, type ParseStats } from './parse';

export interface ImportChatArgs {
  /** Raw text content of the WhatsApp _chat.txt export. */
  content: string;
  /** Chat JID (e.g. "393331234567@c.us"). Must already be in the whitelist for live ingest to allow it; the importer doesn't enforce that here — caller does. */
  waId: string;
  /** Sender display name as it appears in the export for messages from the user. */
  me: string;
  /** Optional override for the contact's display name in the DB. */
  displayName?: string | null;
}

export interface ImportChatStats {
  parse: ParseStats;
  emitted: number;
  inserted: number;
  duplicate: number;
  bursts: number;
  burst_messages: number;
  first_ts: number | null;
  last_ts: number | null;
  live_cutoff_ts: number | null;
}

function syntheticWaMsgId(waId: string, ts: number, sender: string, body: string): string {
  const h = crypto.createHash('sha1');
  h.update(`${waId}|${ts}|${sender}|${body}`);
  return `import:${h.digest('hex').slice(0, 24)}`;
}

/**
 * Parse a WhatsApp export, insert all messages under the given waId, rebuild
 * bursts across the whole DB, and set the contact's live_cutoff_ts to the
 * last imported message's timestamp. Idempotent on re-run for identical text
 * messages (synthetic wa_msg_id is deterministic per chat+ts+sender+body);
 * media-omitted rows from the same second collapse to one row, which is
 * fine because they're filtered at burst-build time.
 */
export function importChat(db: DB, args: ImportChatArgs): ImportChatStats {
  const parsed = parseExport(args.content, args.me);
  const messages = parsed.messages;

  if (messages.length === 0) {
    return {
      parse: parsed.stats,
      emitted: 0,
      inserted: 0,
      duplicate: 0,
      bursts: 0,
      burst_messages: 0,
      first_ts: null,
      last_ts: null,
      live_cutoff_ts: null,
    };
  }

  const firstTs = messages[0].ts;
  const lastTs = messages[messages.length - 1].ts;

  const contactId = upsertContact(db, {
    wa_id: args.waId,
    display_name: args.displayName ?? null,
    is_group: args.waId.endsWith('@g.us'),
    ts: firstTs,
  });

  let inserted = 0;
  let duplicate = 0;
  const tx = db.transaction(() => {
    for (const m of messages) {
      const wa_msg_id = syntheticWaMsgId(args.waId, m.ts, m.sender, m.body);
      const ok = insertRawMessage(db, {
        wa_msg_id,
        contact_id: contactId,
        sender_wa_id: args.waId,
        direction: m.direction,
        body: m.body,
        ts: m.ts,
        media_type: m.media_type,
        media_pointer: null,
      });
      if (ok === null) duplicate++;
      else inserted++;
    }
  });
  tx();

  const rb = rebuildAllBursts(db);
  setLiveCutoff(db, contactId, lastTs);

  return {
    parse: parsed.stats,
    emitted: messages.length,
    inserted,
    duplicate,
    bursts: rb.bursts,
    burst_messages: rb.messages,
    first_ts: firstTs,
    last_ts: lastTs,
    live_cutoff_ts: lastTs,
  };
}
