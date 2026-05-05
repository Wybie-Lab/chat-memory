import crypto from 'crypto';
import {
  upsertContact,
  insertRawMessage,
  rebuildAllBursts,
  type DB,
} from '../../engine';
import type { ParsedSample } from './parse';

export interface IngestResult {
  contactId: number;
  waId: string;
  inserted: number;
  duplicate: number;
  bursts: number;
  burstMessages: number;
  firstTs: number;
  lastTs: number;
}

/**
 * Build a deterministic synthetic waId for a LOCOMO sample. Using @c.us
 * because the engine treats anything not ending in @g.us as a 1:1 chat.
 */
export function locomoWaId(sample: ParsedSample): string {
  return `locomo-${sample.sampleId}-${slug(sample.speakerB)}@c.us`;
}

function slug(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'x';
}

function syntheticWaMsgId(waId: string, ts: number, sender: string, body: string): string {
  const h = crypto.createHash('sha1');
  h.update(`${waId}|${ts}|${sender}|${body}`);
  return `locomo:${h.digest('hex').slice(0, 24)}`;
}

/**
 * Ingest one LOCOMO sample as raw_messages and rebuild bursts. Speaker A is
 * mapped to "out" (me) and speaker B to "in" (the contact). Within a session,
 * turns receive consecutive 1-second offsets so they sort and stay inside a
 * single burst (gap < 30 min).
 *
 * Returns the synthetic waId for the contact so the caller can reference it
 * later if needed.
 */
export function ingestSample(db: DB, sample: ParsedSample): IngestResult {
  const waId = locomoWaId(sample);
  const firstTs = sample.sessions[0].startTs;

  const contactId = upsertContact(db, {
    wa_id: waId,
    display_name: sample.speakerB,
    is_group: false,
    ts: firstTs,
  });

  let inserted = 0;
  let duplicate = 0;
  let lastTs = firstTs;

  const tx = db.transaction(() => {
    for (const session of sample.sessions) {
      for (let i = 0; i < session.turns.length; i++) {
        const turn = session.turns[i];
        const ts = session.startTs + i;
        if (ts > lastTs) lastTs = ts;
        const direction: 'in' | 'out' = turn.speaker === sample.speakerA ? 'out' : 'in';
        const body = (turn.text ?? '').trim();
        if (body === '') continue;
        const wa_msg_id = syntheticWaMsgId(waId, ts, turn.speaker, body);
        const msgId = insertRawMessage(db, {
          wa_msg_id,
          contact_id: contactId,
          sender_wa_id: waId,
          direction,
          body,
          ts,
          media_type: null,
          media_pointer: null,
        });
        if (msgId === null) duplicate++;
        else inserted++;
      }
    }
  });
  tx();

  const rb = rebuildAllBursts(db);

  return {
    contactId,
    waId,
    inserted,
    duplicate,
    bursts: rb.bursts,
    burstMessages: rb.messages,
    firstTs,
    lastTs,
  };
}
