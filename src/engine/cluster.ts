/**
 * Cluster summary refresh — shared between the burst pipeline and the
 * curator's apply path. Both write paths can mutate the active fact set for
 * a (subject, category) cluster, and both need the same rolled-up summary
 * to stay in sync.
 *
 * Below CLUSTER_SUMMARY_MIN_FACTS active facts → delete the summary; the
 * memory block falls back to listing raw facts for that subject.
 */

import {
  activeFactsForCluster,
  deleteClusterSummary,
  logProcessing,
  upsertClusterSummary,
  type ActiveFactRow,
  type DB,
} from './storage/db';
import type { ExtractedFact, LLMProvider } from '../llm/provider';

export const CLUSTER_SUMMARY_MIN_FACTS = 3;

export type ClusterRefreshResult = 'refreshed' | 'deleted' | 'noop';

export interface RefreshClusterOptions {
  /** Optional burst id to attribute the summarize LLM call to in processing_log. */
  burstId?: number | null;
  /** Optional log callback (for pipeline; no-op for apply path). */
  log?: (line: string) => void;
}

export async function refreshClusterSummary(
  db: DB,
  provider: LLMProvider,
  subject: string,
  category: string,
  opts: RefreshClusterOptions = {}
): Promise<ClusterRefreshResult> {
  const log = opts.log ?? (() => {});
  const burstId = opts.burstId ?? null;
  const facts = activeFactsForCluster(db, subject, category);
  if (facts.length < CLUSTER_SUMMARY_MIN_FACTS) {
    deleteClusterSummary(db, subject, category);
    return 'deleted';
  }

  const r = await provider.summarizeCluster({
    subject,
    category: category as ExtractedFact['category'],
    facts: facts.map((f) => ({
      id: f.id,
      content: f.content,
      confidence: f.confidence,
      age_days: ageDays(f),
    })),
  });
  upsertClusterSummary(db, {
    subject_wa_id: subject,
    category,
    summary: r.summary,
    fact_ids: facts.map((f) => f.id),
  });
  logProcessing(db, {
    burst_id: burstId,
    stage: 'extract',
    model: r.usage.model + ' [summarize]',
    tokens_in: r.usage.tokens_in,
    tokens_out: r.usage.tokens_out,
  });
  log(
    `  CLUSTER ${subject}/${category} (${facts.length} facts) → ${r.summary.length}c`
  );
  return 'refreshed';
}

function ageDays(fact: ActiveFactRow): number {
  const now = Math.floor(Date.now() / 1000);
  return Math.max(0, (now - fact.extracted_at) / 86400);
}
