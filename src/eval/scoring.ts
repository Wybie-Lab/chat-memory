import type { ExpectedFact, FactRecallResult, FactPrecisionResult } from './types';
import { listFacts, type DB } from '../memory/db';
import type { LLMProvider } from '../llm/provider';

function normalize(s: string): string {
  return s.toLowerCase().trim();
}

function factMatches(
  expected: ExpectedFact,
  actual: { subject_wa_id: string; category: string; content: string }
): boolean {
  if (normalize(actual.subject_wa_id) !== normalize(expected.subject)) return false;
  if (expected.category && actual.category !== expected.category) return false;
  const content = normalize(actual.content);
  return expected.content_contains.every((needle) => content.includes(normalize(needle)));
}

export function scoreFactRecall(db: DB, expected: ExpectedFact[]): FactRecallResult {
  const extracted = listFacts(db, {}, 1000);
  let matched = 0;
  const missing: ExpectedFact[] = [];
  for (const e of expected) {
    if (extracted.some((a) => factMatches(e, a))) matched++;
    else missing.push(e);
  }
  return { expected: expected.length, matched, missing };
}

/**
 * Loose precision: count extracted facts that have NO subject overlap with
 * any expected fact. Real conversations contain valid extras (e.g. "she said
 * thanks" → preference for politeness) that the test won't list, so we do
 * not penalize them — only facts about subjects we never expected anything
 * about. Tighter precision needs hand-labeled "should not extract" lists.
 */
export function scoreFactPrecision(db: DB, expected: ExpectedFact[]): FactPrecisionResult {
  const extracted = listFacts(db, {}, 1000);
  const expectedSubjects = new Set(expected.map((e) => normalize(e.subject)));
  const unanchored = extracted.filter(
    (a) => !expectedSubjects.has(normalize(a.subject_wa_id))
  ).length;
  return { total_extracted: extracted.length, unanchored };
}

const RETRIEVAL_JUDGE_SYSTEM = `You are scoring how useful a memory block is for answering a user's question, on a 0-10 scale.

10 = the memory block contains the exact info needed; the assistant can answer fully and accurately.
7 = the memory block has the relevant subject(s) and most needed info; minor gaps.
4 = the memory block touches the topic but is missing key facts; assistant could only partially answer.
1 = the memory block is unrelated; the assistant cannot answer from it.
0 = empty memory block.

Be strict. Output JSON only.`;

const RETRIEVAL_JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    score: { type: 'number' },
    reasoning: { type: 'string' },
  },
  required: ['score', 'reasoning'],
  additionalProperties: false,
} as const;

/**
 * LLM-as-judge for retrieval relevance. Re-uses the chat() provider hook with
 * a hand-crafted prompt — the structured JSON parsing is done inline since
 * provider.chat returns prose. For evals we accept the cost of a Sonnet call
 * per query.
 */
export async function judgeRetrievalRelevance(
  provider: LLMProvider,
  question: string,
  memoryBlock: string
): Promise<{ score: number; reasoning: string }> {
  const prompt = [
    `Question: ${question}`,
    '',
    'Memory block:',
    memoryBlock,
    '',
    'Score how well this memory block enables answering the question.',
  ].join('\n');

  // We only have provider.chat as a free-form text path; for the judge we
  // post-parse a JSON answer rather than wiring a new SDK call shape. The
  // judge prompt asks for JSON; the parser extracts the first {...} blob.
  const r = await provider.chat({
    question: prompt,
    memoryBlock:
      `<memory>\n<judging_instructions>\n${RETRIEVAL_JUDGE_SYSTEM}\n` +
      `\nReturn JSON: {"score": number 0-10, "reasoning": "..."}\n` +
      `</judging_instructions>\n</memory>`,
  });

  const json = extractJsonObject(r.answer);
  if (!json) return { score: 0, reasoning: `judge returned non-JSON: ${r.answer.slice(0, 100)}` };
  const score = typeof json.score === 'number' ? Math.max(0, Math.min(10, json.score)) : 0;
  const reasoning = typeof json.reasoning === 'string' ? json.reasoning : '';
  return { score, reasoning };
}

function extractJsonObject(s: string): Record<string, unknown> | null {
  const start = s.indexOf('{');
  const end = s.lastIndexOf('}');
  if (start === -1 || end === -1 || end < start) return null;
  try {
    return JSON.parse(s.slice(start, end + 1)) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export function scoreAnswerSubstrings(
  answer: string,
  expected: string[] | undefined
): { hits: number; total: number } {
  if (!expected || expected.length === 0) return { hits: 0, total: 0 };
  const hay = answer.toLowerCase();
  let hits = 0;
  for (const needle of expected) {
    if (hay.includes(needle.toLowerCase())) hits++;
  }
  return { hits, total: expected.length };
}
