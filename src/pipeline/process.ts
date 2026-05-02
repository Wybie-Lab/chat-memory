import {
  insertFact,
  insertEmbedding,
  markBurstFiltered,
  markBurstProcessed,
  markFactSuperseded,
  markFactDeleted,
  existingFactsForSubject,
  logProcessing,
  listUnprocessedBursts,
  getBurstMessages,
  type DB,
  type UnprocessedBurst,
  type ActiveFactRow,
} from '../memory/db';
import type {
  LLMProvider,
  BurstInput,
  ConsolidateInput,
  ConsolidationOp,
  ExtractedFact,
} from '../llm/provider';
import { guardFacts } from './extract-guard';

export interface ProcessOptions {
  batchSize?: number;
  log?: (line: string) => void;
}

export interface ProcessStats {
  bursts_scanned: number;
  bursts_kept: number;
  bursts_dropped: number;
  facts_added: number;
  facts_updated: number;
  facts_deleted: number;
  facts_dropped: number;
  facts_guarded: number;
  errors: number;
}

function emptyStats(scanned = 0): ProcessStats {
  return {
    bursts_scanned: scanned,
    bursts_kept: 0,
    bursts_dropped: 0,
    facts_added: 0,
    facts_updated: 0,
    facts_deleted: 0,
    facts_dropped: 0,
    facts_guarded: 0,
    errors: 0,
  };
}

interface BurstResult {
  added: number;
  updated: number;
  deleted: number;
  dropped: number;
  guarded: number;
}

export async function processBatch(
  db: DB,
  provider: LLMProvider,
  opts: ProcessOptions = {}
): Promise<ProcessStats> {
  const limit = opts.batchSize ?? 5;
  const log = opts.log ?? (() => {});
  const bursts = listUnprocessedBursts(db, limit);

  const stats = emptyStats(bursts.length);

  for (const burst of bursts) {
    try {
      const result = await processOne(db, provider, burst, log);
      if (result === null) {
        stats.bursts_dropped++;
      } else {
        stats.bursts_kept++;
        stats.facts_added += result.added;
        stats.facts_updated += result.updated;
        stats.facts_deleted += result.deleted;
        stats.facts_dropped += result.dropped;
        stats.facts_guarded += result.guarded;
      }
    } catch (err) {
      stats.errors++;
      log(`  burst ${burst.id} ERROR: ${err instanceof Error ? err.message : err}`);
    }
  }

  return stats;
}

function normalizeSubject(s: string): string {
  return s.trim().toLowerCase();
}

async function processOne(
  db: DB,
  provider: LLMProvider,
  burst: UnprocessedBurst,
  log: (line: string) => void
): Promise<BurstResult | null> {
  const messages = getBurstMessages(db, burst.id);
  if (messages.length === 0) {
    markBurstFiltered(db, burst.id, false);
    markBurstProcessed(db, burst.id);
    log(`  burst ${burst.id} empty after body/media filter — skipped`);
    return null;
  }

  const burstInput: BurstInput = {
    contactName: burst.chat_display_name,
    contactNotes: burst.chat_notes,
    isGroup: burst.is_group,
    startTs: burst.start_ts,
    endTs: burst.end_ts,
    lines: messages.map((m) => ({ direction: m.direction, body: m.body, ts: m.ts })),
  };

  const filterResult = await provider.filterBurst(burstInput);
  logProcessing(db, {
    burst_id: burst.id,
    stage: 'filter',
    model: filterResult.usage.model,
    tokens_in: filterResult.usage.tokens_in,
    tokens_out: filterResult.usage.tokens_out,
  });

  if (!filterResult.keep) {
    markBurstFiltered(db, burst.id, false);
    markBurstProcessed(db, burst.id);
    log(`  burst ${burst.id} (${messages.length} msgs) drop: ${filterResult.reason}`);
    return null;
  }

  const extractResult = await provider.extractBurst(burstInput);
  logProcessing(db, {
    burst_id: burst.id,
    stage: 'extract',
    model: extractResult.usage.model,
    tokens_in: extractResult.usage.tokens_in,
    tokens_out: extractResult.usage.tokens_out,
  });

  const guardResults = guardFacts(extractResult.facts);
  const kept = guardResults.filter((g) => !g.drop).map((g) => g.fact);
  const guarded = guardResults.filter((g) => g.drop);

  for (const d of guarded) {
    log(`  burst ${burst.id} GUARD drop: "${d.fact.content}" — ${d.reason}`);
  }

  if (kept.length === 0) {
    markBurstFiltered(db, burst.id, true);
    markBurstProcessed(db, burst.id);
    if (extractResult.facts.length === 0) {
      log(`  burst ${burst.id} kept but extracted 0 facts`);
    } else {
      log(`  burst ${burst.id} all ${extractResult.facts.length} fact(s) dropped by guard`);
    }
    return { added: 0, updated: 0, deleted: 0, dropped: 0, guarded: guarded.length };
  }

  const groups = new Map<string, ExtractedFact[]>();
  for (const f of kept) {
    const key = normalizeSubject(f.subject);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(f);
  }

  type ResolvedOp =
    | { kind: 'add'; subject: string; fact: ExtractedFact }
    | { kind: 'update'; subject: string; fact: ExtractedFact; oldId: number }
    | { kind: 'delete'; oldId: number; reason: string }
    | { kind: 'drop'; fact: ExtractedFact; reason: string };

  const ops: ResolvedOp[] = [];

  for (const [subjectKey, candidates] of groups) {
    const existing = existingFactsForSubject(db, subjectKey);
    const subjectStored = subjectKey;

    if (existing.length === 0) {
      for (const c of candidates) ops.push({ kind: 'add', subject: subjectStored, fact: c });
      continue;
    }

    const consolidateInput: ConsolidateInput = {
      subject: subjectStored,
      existing: existing.map((e) => ({
        id: e.id,
        content: e.content,
        category: e.category,
        confidence: e.confidence,
        age_days: ageDays(e),
      })),
      candidates,
    };

    const consolidation = await provider.consolidate(consolidateInput);
    logProcessing(db, {
      burst_id: burst.id,
      stage: 'extract',
      model: consolidation.usage.model + ' [consolidate]',
      tokens_in: consolidation.usage.tokens_in,
      tokens_out: consolidation.usage.tokens_out,
    });

    const consumedCandidates = new Set<number>();
    for (const op of consolidation.ops) {
      const resolved = resolveOp(op, candidates, subjectStored, consumedCandidates, log, burst.id);
      if (resolved) ops.push(resolved);
    }
    for (let i = 0; i < candidates.length; i++) {
      if (!consumedCandidates.has(i)) {
        log(
          `  burst ${burst.id} CONSOLIDATE skipped candidate ${i} ("${candidates[i].content}") — defaulting to ADD`
        );
        ops.push({ kind: 'add', subject: subjectStored, fact: candidates[i] });
      }
    }
  }

  const toEmbed = ops.filter(
    (o): o is Extract<ResolvedOp, { kind: 'add' | 'update' }> =>
      o.kind === 'add' || o.kind === 'update'
  );
  const embedded = new Map<ResolvedOp, { vector: number[]; usage: { model: string; tokens_in: number; tokens_out: number } }>();
  for (const o of toEmbed) {
    const r = await provider.embed(o.fact.content);
    embedded.set(o, { vector: r.vector, usage: r.usage });
  }

  const result: BurstResult = {
    added: 0,
    updated: 0,
    deleted: 0,
    dropped: 0,
    guarded: guarded.length,
  };

  const writeTx = db.transaction(() => {
    for (const o of ops) {
      if (o.kind === 'add') {
        const e = embedded.get(o)!;
        const factId = insertFact(db, {
          source_burst_id: burst.id,
          source_msg_id: null,
          subject: o.subject,
          category: o.fact.category,
          content: o.fact.content,
          confidence: o.fact.confidence,
        });
        insertEmbedding(db, factId, e.vector);
        logProcessing(db, {
          burst_id: burst.id,
          stage: 'embed',
          model: e.usage.model,
          tokens_in: e.usage.tokens_in,
          tokens_out: e.usage.tokens_out,
        });
        result.added++;
        log(`  burst ${burst.id} ADD [${o.fact.category}] about ${o.subject}: ${o.fact.content}`);
      } else if (o.kind === 'update') {
        const e = embedded.get(o)!;
        const newId = insertFact(db, {
          source_burst_id: burst.id,
          source_msg_id: null,
          subject: o.subject,
          category: o.fact.category,
          content: o.fact.content,
          confidence: o.fact.confidence,
        });
        insertEmbedding(db, newId, e.vector);
        markFactSuperseded(db, o.oldId, newId);
        logProcessing(db, {
          burst_id: burst.id,
          stage: 'embed',
          model: e.usage.model,
          tokens_in: e.usage.tokens_in,
          tokens_out: e.usage.tokens_out,
        });
        result.updated++;
        log(`  burst ${burst.id} UPDATE fact ${o.oldId} → ${newId} about ${o.subject}: ${o.fact.content}`);
      } else if (o.kind === 'delete') {
        markFactDeleted(db, o.oldId);
        result.deleted++;
        log(`  burst ${burst.id} DELETE fact ${o.oldId} — ${o.reason}`);
      } else {
        result.dropped++;
        log(`  burst ${burst.id} DROP candidate "${o.fact.content}" — ${o.reason}`);
      }
    }
    markBurstFiltered(db, burst.id, true);
    markBurstProcessed(db, burst.id);
  });
  writeTx();

  return result;
}

function ageDays(fact: ActiveFactRow): number {
  const now = Math.floor(Date.now() / 1000);
  return Math.max(0, (now - fact.extracted_at) / 86400);
}

function resolveOp(
  op: ConsolidationOp,
  candidates: ExtractedFact[],
  subject: string,
  consumedCandidates: Set<number>,
  log: (line: string) => void,
  burstId: number
):
  | { kind: 'add'; subject: string; fact: ExtractedFact }
  | { kind: 'update'; subject: string; fact: ExtractedFact; oldId: number }
  | { kind: 'delete'; oldId: number; reason: string }
  | { kind: 'drop'; fact: ExtractedFact; reason: string }
  | null {
  if (op.op === 'ADD') {
    if (consumedCandidates.has(op.candidate_index)) {
      log(`  burst ${burstId} CONSOLIDATE error: candidate ${op.candidate_index} consumed twice`);
      return null;
    }
    consumedCandidates.add(op.candidate_index);
    return {
      kind: 'add',
      subject,
      fact: {
        subject,
        category: op.category,
        content: op.content,
        confidence: op.confidence,
      },
    };
  }
  if (op.op === 'UPDATE') {
    if (consumedCandidates.has(op.candidate_index)) {
      log(`  burst ${burstId} CONSOLIDATE error: candidate ${op.candidate_index} consumed twice`);
      return null;
    }
    consumedCandidates.add(op.candidate_index);
    return {
      kind: 'update',
      subject,
      oldId: op.old_fact_id,
      fact: {
        subject,
        category: op.category,
        content: op.content,
        confidence: op.confidence,
      },
    };
  }
  if (op.op === 'DELETE') {
    return { kind: 'delete', oldId: op.old_fact_id, reason: op.reason };
  }
  // DROP
  if (consumedCandidates.has(op.candidate_index)) {
    log(`  burst ${burstId} CONSOLIDATE error: candidate ${op.candidate_index} consumed twice`);
    return null;
  }
  consumedCandidates.add(op.candidate_index);
  return { kind: 'drop', fact: candidates[op.candidate_index], reason: op.reason };
}

export async function processUntilDrained(
  db: DB,
  provider: LLMProvider,
  opts: ProcessOptions = {}
): Promise<ProcessStats> {
  const total = emptyStats();

  while (true) {
    const stats = await processBatch(db, provider, opts);
    total.bursts_scanned += stats.bursts_scanned;
    total.bursts_kept += stats.bursts_kept;
    total.bursts_dropped += stats.bursts_dropped;
    total.facts_added += stats.facts_added;
    total.facts_updated += stats.facts_updated;
    total.facts_deleted += stats.facts_deleted;
    total.facts_dropped += stats.facts_dropped;
    total.facts_guarded += stats.facts_guarded;
    total.errors += stats.errors;

    if (stats.bursts_scanned === 0) break;
  }

  return total;
}
