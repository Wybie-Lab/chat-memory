import fs from 'fs';
import path from 'path';
import {
  openDb,
  upsertContact,
  insertRawMessage,
  assignMessageToBurst,
  rebuildAllBursts,
  type DB,
} from '../memory/db';
import { processUntilDrained } from '../pipeline/process';
import type { LLMProvider } from '../llm/provider';
import { composeMemoryBlock } from '../retrieval/memory-block';
import {
  scoreFactRecall,
  scoreFactPrecision,
  judgeRetrievalRelevance,
  scoreAnswerSubstrings,
} from './scoring';
import type { TestCase, TestTurn, RunResult, QueryResult } from './types';

export interface RunnerOptions {
  /** Where to write per-test sqlite databases. Cleaned up after each test. */
  workdir: string;
  log?: (line: string) => void;
}

/**
 * Run one test case end-to-end: ingest the synthetic transcript into a fresh
 * sqlite DB, drain the processing pipeline, score recall/precision, then run
 * each query through composeMemoryBlock + chat and score the result.
 */
export async function runTest(
  test: TestCase,
  provider: LLMProvider,
  opts: RunnerOptions
): Promise<RunResult> {
  const log = opts.log ?? (() => {});
  const dbPath = path.join(opts.workdir, `${test.id}.db`);
  if (fs.existsSync(dbPath)) fs.unlinkSync(dbPath);
  fs.mkdirSync(opts.workdir, { recursive: true });

  const db = openDb(dbPath);
  const errors: string[] = [];
  const t0 = Date.now();
  let ingested = 0;

  try {
    ingested = ingestTranscript(db, test);
    rebuildAllBursts(db);
  } catch (err) {
    errors.push(`ingest: ${err instanceof Error ? err.message : String(err)}`);
    db.close();
    return emptyRunResult(test, ingested, errors, t0);
  }

  const ingestionMs = Date.now() - t0;

  let bursts_processed = 0;
  try {
    const stats = await processUntilDrained(db, provider, {
      log: (line) => log(`[${test.id}] ${line}`),
      includeUnsettledBursts: true,
    });
    bursts_processed = stats.bursts_kept + stats.bursts_dropped;
  } catch (err) {
    errors.push(`process: ${err instanceof Error ? err.message : String(err)}`);
  }

  const fact_recall = scoreFactRecall(db, test.expected_facts ?? []);
  const fact_precision = scoreFactPrecision(db, test.expected_facts ?? []);

  const queries: QueryResult[] = [];
  for (const q of test.queries ?? []) {
    try {
      const composed = await composeMemoryBlock(db, provider, q.q);
      const chatRes = await provider.chat({ question: q.q, memoryBlock: composed.block });

      const subs = scoreAnswerSubstrings(chatRes.answer, q.expected_answer_substrings);
      const matchedSubjectsOk = q.expected_matched_subjects
        ? q.expected_matched_subjects.every((s) =>
            composed.matched_subjects.some((m) => m.toLowerCase().includes(s.toLowerCase()))
          )
        : true;

      const queryResult: QueryResult = {
        q: q.q,
        matched_subjects: composed.matched_subjects,
        answer: chatRes.answer,
        answer_substring_hits: subs.hits,
        answer_substring_total: subs.total,
        matched_subjects_ok: matchedSubjectsOk,
        citations_count: composed.citations.length,
      };

      if (q.judge_retrieval !== false) {
        const judge = await judgeRetrievalRelevance(provider, q.q, composed.block);
        queryResult.retrieval_relevance_score = judge.score;
        queryResult.retrieval_judge_reasoning = judge.reasoning;
      }

      queries.push(queryResult);
    } catch (err) {
      errors.push(
        `query "${q.q.slice(0, 40)}…": ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  db.close();
  fs.unlinkSync(dbPath);

  return {
    test_id: test.id,
    description: test.description,
    ingested_turns: ingested,
    bursts_processed,
    fact_recall,
    fact_precision,
    queries,
    ingestion_ms: ingestionMs,
    total_ms: Date.now() - t0,
    errors,
  };
}

export async function runAll(
  tests: TestCase[],
  provider: LLMProvider,
  opts: RunnerOptions
): Promise<RunResult[]> {
  const results: RunResult[] = [];
  for (const t of tests) {
    (opts.log ?? (() => {}))(`▶ ${t.id} — ${t.description ?? ''}`);
    results.push(await runTest(t, provider, opts));
  }
  return results;
}

function ingestTranscript(db: DB, test: TestCase): number {
  const contactId = upsertContact(db, {
    wa_id: test.contact_wa_id,
    display_name: test.contact_name,
    is_group: false,
    ts: Math.floor(new Date(test.transcript[0]?.ts ?? Date.now()).getTime() / 1000),
  });

  // Mark the contact as whitelisted so the import-style flow accepts it. The
  // eval pipeline doesn't run the live whitelist check, but this matches the
  // production schema's expectations.
  db.prepare('UPDATE contacts SET whitelisted = 1 WHERE id = ?').run(contactId);

  let count = 0;
  for (const t of test.transcript) {
    const ts = Math.floor(new Date(t.ts).getTime() / 1000);
    const inserted = insertRawMessage(db, {
      wa_msg_id: `eval_${test.id}_${count}`,
      contact_id: contactId,
      sender_wa_id: t.direction === 'out' ? 'me' : test.contact_wa_id,
      direction: t.direction,
      body: t.body,
      ts,
    });
    if (inserted !== null) {
      assignMessageToBurst(db, inserted);
      count++;
    }
  }
  return count;
}

function emptyRunResult(
  test: TestCase,
  ingested: number,
  errors: string[],
  t0: number
): RunResult {
  return {
    test_id: test.id,
    description: test.description,
    ingested_turns: ingested,
    bursts_processed: 0,
    fact_recall: { expected: test.expected_facts?.length ?? 0, matched: 0, missing: [] },
    fact_precision: { total_extracted: 0, unanchored: 0 },
    queries: [],
    ingestion_ms: 0,
    total_ms: Date.now() - t0,
    errors,
  };
}
