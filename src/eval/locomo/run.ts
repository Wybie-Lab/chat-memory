import {
  composeMemoryBlock,
  mapWithConcurrency,
  processUntilDrained,
  retrieveAgentic,
  type DB,
} from '../../engine';
import type { LLMProvider } from '../../llm/provider';
import type { ParsedSample } from './parse';
import { ingestSample, type IngestResult } from './ingest';
import { scoreAnswer, type QaScore } from './score';

export interface RunOptions {
  /** Skip ingestion (assume DB already populated). */
  skipIngest?: boolean;
  /** Skip pipeline processing (assume DB already processed). */
  skipProcess?: boolean;
  /** Limit number of QA pairs to evaluate (default: all). */
  limit?: number;
  /** Skip questions whose category is in this set (LOCOMO category numbers 1..5). */
  skipCategories?: Set<number>;
  /** Only run questions in this category set (when set, others are skipped). */
  onlyCategories?: Set<number>;
  /** Max in-flight QAs (default: 5). Each QA = 1 embed + 1 chat call. */
  qaConcurrency?: number;
  /**
   * Use the retrieval agent (`retrieveAgentic`) instead of the one-shot
   * compose+chat path. The agent drives an ai-sdk loop with read tools over
   * the memory graph; more expensive but can multi-hop.
   */
  agentic?: boolean;
  /** Hard cap on agent tool steps when `agentic=true`. Default 8. */
  agentMaxSteps?: number;
  /** Per-event log line. */
  log?: (line: string) => void;
}

export interface QaResult {
  index: number;
  category: number;
  question: string;
  goldAnswer: string;
  predicted: string;
  score: QaScore;
  memoryChars: number;
  citations: number;
}

export interface RunResult {
  sampleId: string;
  ingest: IngestResult | null;
  qa: QaResult[];
  aggregates: {
    n: number;
    meanExactMatch: number;
    meanTokenF1: number;
    byCategory: Record<number, { n: number; meanExactMatch: number; meanTokenF1: number }>;
  };
}

const DEFAULT_LOG = (line: string) => console.log(line);

export async function runEvalForSample(
  db: DB,
  provider: LLMProvider,
  sample: ParsedSample,
  rawQa: Array<{ question: string; answer: string | number; category: number }>,
  opts: RunOptions = {}
): Promise<RunResult> {
  const log = opts.log ?? DEFAULT_LOG;

  let ingest: IngestResult | null = null;
  if (!opts.skipIngest) {
    log(`[ingest] sample=${sample.sampleId} sessions=${sample.sessions.length}`);
    ingest = ingestSample(db, sample);
    log(
      `[ingest] inserted=${ingest.inserted} duplicate=${ingest.duplicate} bursts=${ingest.bursts}`
    );
  }

  if (!opts.skipProcess) {
    log(`[process] running pipeline (selfLabel=${sample.speakerA})...`);
    const stats = await processUntilDrained(db, provider, {
      batchSize: 5,
      selfLabel: sample.speakerA,
      log: (line) => log(`  ${line}`),
    });
    log(
      `[process] bursts kept=${stats.bursts_kept}/dropped=${stats.bursts_dropped} ` +
        `facts +${stats.facts_added} dropped=${stats.facts_dropped} guarded=${stats.facts_guarded} errors=${stats.errors}`
    );
  }

  const qaToRun = (opts.limit ? rawQa.slice(0, opts.limit) : rawQa).filter(
    (q) => {
      if (opts.skipCategories?.has(q.category)) return false;
      if (opts.onlyCategories && !opts.onlyCategories.has(q.category)) return false;
      return true;
    }
  );

  const concurrency = opts.qaConcurrency ?? 5;
  let done = 0;
  const results = await mapWithConcurrency(qaToRun, concurrency, async (q, i) => {
    let predicted: string;
    let memoryChars: number;
    let citationsCount: number;
    let agentSteps: number | null = null;

    if (opts.agentic) {
      const agent = await retrieveAgentic(db, provider, q.question, {
        maxSteps: opts.agentMaxSteps,
      });
      predicted = agent.answer;
      memoryChars = agent.usage.tokens_in + agent.usage.tokens_out;
      citationsCount = agent.citations.length;
      agentSteps = agent.steps;
    } else {
      const composed = await composeMemoryBlock(db, provider, q.question);
      const chat = await provider.chat({
        question: q.question,
        memoryBlock: composed.block,
        style: 'factoid',
      });
      predicted = chat.answer;
      memoryChars = composed.block.length;
      citationsCount = composed.citations.length;
    }
    const score = scoreAnswer(predicted, q.answer);
    const result: QaResult = {
      index: i,
      category: q.category,
      question: q.question,
      goldAnswer: String(q.answer),
      predicted,
      score,
      memoryChars,
      citations: citationsCount,
    };
    done++;
    const stepsTag = agentSteps !== null ? ` steps=${agentSteps}` : '';
    log(
      `[qa ${done}/${qaToRun.length}]${stepsTag} cat=${q.category} EM=${score.exactMatch} F1=${score.tokenF1.toFixed(2)} | Q: ${truncate(q.question, 80)} | gold: ${truncate(String(q.answer), 60)} | pred: ${truncate(predicted, 80)}`
    );
    return result;
  });

  return {
    sampleId: sample.sampleId,
    ingest,
    qa: results,
    aggregates: aggregate(results),
  };
}

function aggregate(qa: QaResult[]): RunResult['aggregates'] {
  const n = qa.length;
  if (n === 0) {
    return { n: 0, meanExactMatch: 0, meanTokenF1: 0, byCategory: {} };
  }
  const sumEm = qa.reduce((a, r) => a + r.score.exactMatch, 0);
  const sumF1 = qa.reduce((a, r) => a + r.score.tokenF1, 0);

  const byCategory: Record<number, { n: number; meanExactMatch: number; meanTokenF1: number }> = {};
  const buckets = new Map<number, QaResult[]>();
  for (const r of qa) {
    const list = buckets.get(r.category) ?? [];
    list.push(r);
    buckets.set(r.category, list);
  }
  for (const [cat, list] of buckets) {
    byCategory[cat] = {
      n: list.length,
      meanExactMatch: list.reduce((a, r) => a + r.score.exactMatch, 0) / list.length,
      meanTokenF1: list.reduce((a, r) => a + r.score.tokenF1, 0) / list.length,
    };
  }
  return {
    n,
    meanExactMatch: sumEm / n,
    meanTokenF1: sumF1 / n,
    byCategory,
  };
}

function truncate(s: string, n: number): string {
  const oneLine = s.replace(/\s+/g, ' ');
  return oneLine.length <= n ? oneLine : oneLine.slice(0, n - 1) + '…';
}
