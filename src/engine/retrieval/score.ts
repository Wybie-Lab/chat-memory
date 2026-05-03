import {
  searchFactsByVector,
  countFactSources,
  allActiveSubjects,
  type DB,
  type FactSearchResult,
  type SubjectInfo,
} from '../storage/db';

export interface ScoreWeights {
  semantic: number;
  recency: number;
  confidence: number;
  entity: number;
  importance: number;
}

export const DEFAULT_WEIGHTS: ScoreWeights = {
  semantic: 1.0,
  recency: 0.3,
  confidence: 0.5,
  entity: 1.5,
  importance: 0.3,
};

const RECENCY_HALFLIFE_DAYS = 365;

export interface ScoredFact extends FactSearchResult {
  score: number;
  components: {
    similarity: number;
    recency: number;
    confidence: number;
    entity_match: number;
    importance: number;
  };
  matched_subject_ids: string[];
}

export interface RetrievalContext {
  /** All distinct subject ids present in active facts, with display_name. */
  subjects: SubjectInfo[];
  /** Subject ids whose display_name (or wa_id prefix) matched a token in the query. */
  matchedSubjectIds: Set<string>;
}

/**
 * Build a one-shot retrieval context the scorer can read against. Re-build
 * once per query — the active subject list changes only as new bursts land,
 * not faster than the chat turn rate.
 */
export function buildRetrievalContext(db: DB, query: string): RetrievalContext {
  const subjects = allActiveSubjects(db);
  const matchedSubjectIds = matchSubjectsInQuery(query, subjects);
  return { subjects, matchedSubjectIds };
}

/**
 * Substring match the query against each subject's display_name and wa_id
 * prefix (the part before "@c.us"). Case-insensitive. Single-letter or
 * 2-letter names are skipped to avoid spurious matches against noise tokens
 * like "is", "in", "at".
 */
export function matchSubjectsInQuery(query: string, subjects: SubjectInfo[]): Set<string> {
  const q = query.toLowerCase();
  const matched = new Set<string>();
  for (const s of subjects) {
    const candidates: string[] = [];
    if (s.display_name) {
      for (const tok of s.display_name.toLowerCase().split(/\s+/)) {
        if (tok.length >= 3) candidates.push(tok);
      }
    }
    const waPrefix = s.subject_wa_id.split('@')[0];
    if (waPrefix.length >= 3 && /[a-z]/i.test(waPrefix)) {
      candidates.push(waPrefix.toLowerCase());
    }
    for (const c of candidates) {
      if (q.includes(c)) {
        matched.add(s.subject_wa_id);
        break;
      }
    }
  }
  return matched;
}

/**
 * Convert sqlite-vec L2 distance to a 0–1 similarity score. Monotonic, so
 * ranking is preserved; the absolute scale is mostly cosmetic for blending.
 */
function similarityFromDistance(distance: number): number {
  return 1 / (1 + Math.max(0, distance));
}

/**
 * Exponential decay over days since extraction. Half-life RECENCY_HALFLIFE_DAYS,
 * so a fact 1 year old scores ~0.5, a brand-new fact scores ~1.0.
 */
function recencyScore(extractedAt: number): number {
  const ageDays = Math.max(0, (Date.now() / 1000 - extractedAt) / 86400);
  return Math.exp(-(Math.LN2 * ageDays) / RECENCY_HALFLIFE_DAYS);
}

/**
 * Importance proxy: how many distinct supporting sources back this fact.
 * log-scaled so the first 2-3 sources are most informative; diminishing
 * returns past 10. Returns 0–1.
 */
function importanceScore(sourceCount: number): number {
  return Math.min(1, Math.log(1 + sourceCount) / Math.log(10));
}

export interface ScoreOptions {
  weights?: Partial<ScoreWeights>;
}

/**
 * Score and rank a list of vector-search candidates against the query
 * context. The candidates come from sqlite-vec semantic search; this
 * function blends in non-semantic signals and returns them ranked.
 */
export function scoreCandidates(
  db: DB,
  candidates: FactSearchResult[],
  ctx: RetrievalContext,
  opts: ScoreOptions = {}
): ScoredFact[] {
  const w: ScoreWeights = { ...DEFAULT_WEIGHTS, ...(opts.weights ?? {}) };

  const scored: ScoredFact[] = candidates.map((c) => {
    const similarity = similarityFromDistance(c.distance);
    const recency = recencyScore(c.extracted_at);
    const confidence = c.confidence;
    const entity_match = ctx.matchedSubjectIds.has(c.subject_wa_id) ? 1 : 0;
    const importance = importanceScore(countFactSources(db, c.id));

    const score =
      w.semantic * similarity +
      w.recency * recency +
      w.confidence * confidence +
      w.entity * entity_match +
      w.importance * importance;

    return {
      ...c,
      score,
      components: { similarity, recency, confidence, entity_match, importance },
      matched_subject_ids: entity_match ? [c.subject_wa_id] : [],
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored;
}

/**
 * One-call retrieve: query embedding → vector search → hybrid scoring.
 * Used by composeMemoryBlock as the "top semantic+entity matches" section.
 */
export function hybridRetrieve(
  db: DB,
  queryEmbedding: number[],
  ctx: RetrievalContext,
  opts: { k?: number; rerankTopK?: number } & ScoreOptions = {}
): ScoredFact[] {
  const k = opts.k ?? 30;
  const candidates = searchFactsByVector(db, queryEmbedding, k);
  const scored = scoreCandidates(db, candidates, ctx, opts);
  return opts.rerankTopK ? scored.slice(0, opts.rerankTopK) : scored;
}
