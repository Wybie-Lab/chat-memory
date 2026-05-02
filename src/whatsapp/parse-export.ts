/**
 * Parser for WhatsApp's "Export Chat" .txt format.
 *
 * Tested format: iOS, English UI, Italian content, DD/MM/YY 24h date format.
 * Each message starts with `[DD/MM/YY, HH:MM:SS] sender name: body`.
 * Continuation lines (multi-line bodies) have NO date prefix.
 * Some lines start with U+200E (LEFT-TO-RIGHT MARK), used by WhatsApp to
 * mark system-generated content; we strip it.
 *
 * Other locales / Android variants use different date formats and media
 * markers — extend MEDIA_PATTERNS / HEADER_RE if needed.
 */

export interface ParsedMessage {
  ts: number;            // unix seconds, interpreted in the host's local TZ
  sender: string;        // exact display name from the export
  direction: 'in' | 'out';
  body: string;          // empty when media
  media_type: string | null;
}

const LRM = /‎/g;

const HEADER_RE =
  /^\[(\d{1,2})\/(\d{1,2})\/(\d{2,4}),\s*(\d{1,2}):(\d{2}):(\d{2})\]\s+([^:]+?):\s?(.*)$/;

const MEDIA_PATTERNS: Array<{ re: RegExp; type: string }> = [
  { re: /^image omitted$/i,             type: 'image' },
  { re: /^video omitted$/i,             type: 'video' },
  { re: /^audio omitted$/i,             type: 'audio' },
  { re: /^voice message omitted$/i,     type: 'audio' },
  { re: /^GIF omitted$/i,               type: 'gif' },
  { re: /^sticker omitted$/i,           type: 'sticker' },
  { re: /^document omitted$/i,          type: 'document' },
  { re: /^contact card omitted$/i,      type: 'contact' },
  { re: /^Location: https?:\/\/\S+$/i,  type: 'location' },
];

const SYSTEM_BODY_PATTERNS = [
  /^Messages and calls are end-to-end encrypted/i,
  /^This message was deleted\.?$/i,
  /^You deleted this message\.?$/i,
  /^.* changed (the )?(group )?subject (from|to)/i,
  /^.* (added|removed|left|joined)/i,
];

interface RawHeader {
  ts: number;
  sender: string;
  body: string;
}

function parseHeader(line: string): RawHeader | null {
  const stripped = line.replace(LRM, '');
  const m = HEADER_RE.exec(stripped);
  if (!m) return null;
  const [, dd, MM, yy, HH, mm, ss, sender, body] = m;
  const year = yy.length === 2 ? 2000 + Number(yy) : Number(yy);
  // Local-TZ interpretation: assumes the import is run on a machine whose TZ
  // matches the export's source phone. Italy → CET/CEST handled by the host.
  const date = new Date(
    year,
    Number(MM) - 1,
    Number(dd),
    Number(HH),
    Number(mm),
    Number(ss)
  );
  return {
    ts: Math.floor(date.getTime() / 1000),
    sender: sender.trim(),
    body: body,
  };
}

function classifyBody(body: string): { body: string; media_type: string | null; skip: boolean } {
  const trimmed = body.replace(LRM, '').trim();

  for (const p of SYSTEM_BODY_PATTERNS) {
    if (p.test(trimmed)) return { body: '', media_type: null, skip: true };
  }
  for (const p of MEDIA_PATTERNS) {
    if (p.re.test(trimmed)) return { body: '', media_type: p.type, skip: false };
  }
  return { body: trimmed, media_type: null, skip: false };
}

export interface ParseStats {
  total: number;
  text: number;
  media: number;
  skipped_system: number;
  skipped_empty: number;
}

export function parseExport(
  text: string,
  meName: string
): { messages: ParsedMessage[]; stats: ParseStats } {
  const lines = text.split(/\r?\n/);
  const out: ParsedMessage[] = [];
  const stats: ParseStats = {
    total: 0,
    text: 0,
    media: 0,
    skipped_system: 0,
    skipped_empty: 0,
  };

  // Working buffer for the message currently being assembled. Continuation
  // lines append to its body until we hit the next header line.
  let cur:
    | (RawHeader & { bodyLines: string[] })
    | null = null;

  const flush = () => {
    if (!cur) return;
    const fullBody = cur.bodyLines.join('\n');
    const classified = classifyBody(fullBody);

    if (classified.skip) {
      stats.skipped_system++;
      cur = null;
      return;
    }
    if (!classified.media_type && classified.body === '') {
      stats.skipped_empty++;
      cur = null;
      return;
    }

    out.push({
      ts: cur.ts,
      sender: cur.sender,
      direction: cur.sender === meName ? 'out' : 'in',
      body: classified.body,
      media_type: classified.media_type,
    });
    if (classified.media_type) stats.media++;
    else stats.text++;
    cur = null;
  };

  for (const line of lines) {
    const header = parseHeader(line);
    if (header) {
      flush();
      cur = { ...header, bodyLines: [header.body] };
      stats.total++;
    } else if (cur) {
      // continuation line of the current message
      cur.bodyLines.push(line.replace(LRM, ''));
    }
    // lines without a current message and without a header are dropped silently
    // (export usually doesn't have these, but it's safe to ignore).
  }
  flush();

  return { messages: out, stats };
}
