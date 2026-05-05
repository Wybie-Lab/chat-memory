/**
 * Lightweight QA scoring for the LOCOMO thin slice.
 *
 * Two metrics, both bag-of-words on normalized text:
 *   - exactMatch: 1 if normalized strings are identical, else 0
 *   - tokenF1:    standard SQuAD-style token overlap F1
 *
 * This is intentionally simpler than LOCOMO's published eval (which uses
 * Rouge/BLEU + an LLM judge). For the thin slice we want a sanity score
 * that's free, fast, and deterministic.
 */

export interface QaScore {
  exactMatch: number;
  tokenF1: number;
  predNormalized: string;
  goldNormalized: string;
}

const ARTICLES_RE = /\b(a|an|the)\b/g;
const PUNCT_RE = /[\p{P}\p{S}]+/gu;
const WS_RE = /\s+/g;
// "[fact:42]" / "[fact:42, fact:7]" / "[fact:42,7]" — chat-style citations
// that aren't part of the answer. Strip before normalizing so they don't
// bleed into token-F1.
const CITATION_RE = /\[fact:[\d,\s]+\]/gi;

function normalize(s: string): string {
  return s
    .replace(CITATION_RE, ' ')
    .toLowerCase()
    .replace(PUNCT_RE, ' ')
    .replace(ARTICLES_RE, ' ')
    .replace(WS_RE, ' ')
    .trim();
}

function tokenize(s: string): string[] {
  return normalize(s).split(' ').filter((t) => t.length > 0);
}

function f1(predTokens: string[], goldTokens: string[]): number {
  if (predTokens.length === 0 && goldTokens.length === 0) return 1;
  if (predTokens.length === 0 || goldTokens.length === 0) return 0;
  const goldCounts = new Map<string, number>();
  for (const t of goldTokens) goldCounts.set(t, (goldCounts.get(t) ?? 0) + 1);
  let common = 0;
  for (const t of predTokens) {
    const left = goldCounts.get(t) ?? 0;
    if (left > 0) {
      common++;
      goldCounts.set(t, left - 1);
    }
  }
  if (common === 0) return 0;
  const precision = common / predTokens.length;
  const recall = common / goldTokens.length;
  return (2 * precision * recall) / (precision + recall);
}

export function scoreAnswer(pred: string, gold: string | number): QaScore {
  const goldStr = String(gold);
  const predNorm = normalize(pred);
  const goldNorm = normalize(goldStr);
  const predTokens = tokenize(pred);
  const goldTokens = tokenize(goldStr);
  return {
    exactMatch: predNorm === goldNorm ? 1 : 0,
    tokenF1: f1(predTokens, goldTokens),
    predNormalized: predNorm,
    goldNormalized: goldNorm,
  };
}
