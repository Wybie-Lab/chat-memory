/**
 * Parser for LOCOMO benchmark conversations.
 *
 * LOCOMO format (snap-research/locomo, data/locomo10.json) is a list of
 * samples. Each sample has:
 *   - sample_id: string
 *   - conversation: { speaker_a, speaker_b, session_<i>, session_<i>_date_time }
 *       Each session_<i> is an array of { speaker, dia_id, text } turns.
 *       session_<i>_date_time is a human string like "1:56 pm on 8 May, 2023".
 *   - qa: array of { question, answer, evidence: ["D1:3", ...], category: 1..5 }
 *
 * We map the LOCOMO sample onto the engine's data model by treating
 * speaker_a as `me` and speaker_b as the contact. Within a session we
 * spread turns over consecutive 1-second offsets so ordering is preserved
 * but every turn in one session falls inside a single burst (gap < 30 min).
 * Sessions days/weeks apart land in separate bursts naturally.
 */

export interface LocomoTurn {
  speaker: string;
  dia_id: string;
  text: string;
}

export interface LocomoSample {
  sample_id: string;
  conversation: Record<string, unknown> & {
    speaker_a: string;
    speaker_b: string;
  };
  qa: Array<{
    question: string;
    answer: string | number;
    evidence?: string[];
    category: number;
    adversarial_answer?: string;
  }>;
}

export interface ParsedSession {
  index: number;
  startTs: number;
  dateTimeRaw: string;
  turns: LocomoTurn[];
}

export interface ParsedSample {
  sampleId: string;
  speakerA: string;
  speakerB: string;
  sessions: ParsedSession[];
}

const MONTHS: Record<string, number> = {
  january: 0, february: 1, march: 2, april: 3, may: 4, june: 5,
  july: 6, august: 7, september: 8, october: 9, november: 10, december: 11,
};

const DATE_RE = /^(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})$/i;

/**
 * Parse a LOCOMO date_time string like "1:56 pm on 8 May, 2023" into a
 * Unix-second timestamp interpreted as UTC. (LOCOMO doesn't disclose a
 * timezone; UTC is the only consistent choice for reproducibility.)
 */
export function parseSessionDateTime(s: string): number {
  const m = DATE_RE.exec(s.trim());
  if (!m) throw new Error(`unrecognized session date_time: "${s}"`);
  const [, hhStr, mmStr, ampm, ddStr, monthName, yyyyStr] = m;
  let hh = Number(hhStr);
  const mm = Number(mmStr);
  const dd = Number(ddStr);
  const year = Number(yyyyStr);
  const monthIdx = MONTHS[monthName.toLowerCase()];
  if (monthIdx === undefined) throw new Error(`unknown month "${monthName}" in "${s}"`);
  const isPm = ampm.toLowerCase() === 'pm';
  if (hh === 12) hh = isPm ? 12 : 0;
  else if (isPm) hh += 12;
  return Math.floor(Date.UTC(year, monthIdx, dd, hh, mm, 0) / 1000);
}

export function parseSample(raw: LocomoSample): ParsedSample {
  const conv = raw.conversation;
  const sessions: ParsedSession[] = [];
  for (let i = 1; i < 100; i++) {
    const turnsKey = `session_${i}`;
    const dateKey = `session_${i}_date_time`;
    const turns = conv[turnsKey] as LocomoTurn[] | undefined;
    if (!turns) continue;
    const dateRaw = conv[dateKey] as string | undefined;
    if (!dateRaw) throw new Error(`${raw.sample_id}: ${turnsKey} has no ${dateKey}`);
    sessions.push({
      index: i,
      startTs: parseSessionDateTime(dateRaw),
      dateTimeRaw: dateRaw,
      turns,
    });
  }
  if (sessions.length === 0) throw new Error(`${raw.sample_id}: no sessions found`);
  return {
    sampleId: raw.sample_id,
    speakerA: conv.speaker_a,
    speakerB: conv.speaker_b,
    sessions,
  };
}
