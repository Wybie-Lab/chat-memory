import {
  insertFact,
  insertFactConnection,
  insertEmbedding,
  markBurstFiltered,
  markBurstProcessed,
  existingFactsForSubject,
  countActiveFactsForSubject,
  searchFactsForSubjectByVector,
  logProcessing,
  listUnprocessedBursts,
  getBurstMessages,
  listMemoryThreads,
  createMemoryThread,
  addFactToThread,
  type ConnectionPredicate,
  type DB,
  type UnprocessedBurst,
  type ActiveFactRow,
} from './storage/db';
import { writeExtractedGraph, type GraphFactContext } from './graph';
import { refreshClusterSummary } from './cluster';
import { planTriggeredRunsForFact } from './agent/curator';
import { mapWithConcurrency } from './concurrency';
import type {
  LLMProvider,
  BurstInput,
  ConsolidateInput,
  ConsolidationOp,
  ExtractedFact,
} from '../llm/provider';
import { guardFacts } from './guard';

export interface ProcessOptions {
  batchSize?: number;
  log?: (line: string) => void;
  /** For eval: ignore the BURST_GAP_SECONDS settling cutoff. */
  includeUnsettledBursts?: boolean;
  /**
   * For eval: override the literal subject string used for the user's own
   * messages (default 'me'). When set, first-person facts will be
   * attributed to this name in the DB and the chat layer can answer
   * third-person questions using that name directly.
   */
  selfLabel?: string;
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
  clusters_refreshed: number;
  clusters_deleted: number;
  errors: number;
}

function ageDays(fact: { extracted_at: number }): number {
  const now = Math.floor(Date.now() / 1000);
  return Math.max(0, (now - fact.extracted_at) / 86400);
}

// Above this many active facts for a subject, the consolidator switches from
// "send all existing facts to the LLM" to "embed each candidate first, then
// pick top-N existing by vector similarity". Keeps consolidate prompts bounded
// for power-chats with hundreds of facts about one person.
const CONSOLIDATION_FULL_LIST_THRESHOLD = 30;
const CONSOLIDATION_SIMILAR_PER_CANDIDATE = 12;
const GRAPH_ENABLED = process.env.ENABLE_GRAPH === '1';

// Max in-flight LLM/embed calls per parallel stage in processOne. Bounded so
// we don't fan out to N requests at once and trip provider rate limits.
const LLM_CONCURRENCY = 5;

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
    clusters_refreshed: 0,
    clusters_deleted: 0,
    errors: 0,
  };
}

interface BurstResult {
  added: number;
  updated: number;
  deleted: number;
  dropped: number;
  guarded: number;
  clusters_refreshed: number;
  clusters_deleted: number;
}

export async function processBatch(
  db: DB,
  provider: LLMProvider,
  opts: ProcessOptions = {}
): Promise<ProcessStats> {
  const limit = opts.batchSize ?? 5;
  const log = opts.log ?? (() => {});
  const bursts = listUnprocessedBursts(db, limit, {
    includeUnsettled: opts.includeUnsettledBursts,
  });

  const stats = emptyStats(bursts.length);

  for (const burst of bursts) {
    try {
      const result = await processOne(db, provider, burst, log, opts.selfLabel);
      if (result === null) {
        stats.bursts_dropped++;
      } else {
        stats.bursts_kept++;
        stats.facts_added += result.added;
        stats.facts_updated += result.updated;
        stats.facts_deleted += result.deleted;
        stats.facts_dropped += result.dropped;
        stats.facts_guarded += result.guarded;
        stats.clusters_refreshed += result.clusters_refreshed;
        stats.clusters_deleted += result.clusters_deleted;
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
  log: (line: string) => void,
  selfLabel?: string
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
    selfLabel,
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
    return {
      added: 0,
      updated: 0,
      deleted: 0,
      dropped: 0,
      guarded: guarded.length,
      clusters_refreshed: 0,
      clusters_deleted: 0,
    };
  }

  const groups = new Map<string, ExtractedFact[]>();
  for (const f of kept) {
    const key = normalizeSubject(f.subject);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(f);
  }

  type ResolvedOp =
    | { kind: 'add'; subject: string; fact: ExtractedFact }
    | {
        kind: 'connect';
        subject: string;
        fact: ExtractedFact;
        oldId: number;
        predicate: ConnectionPredicate;
        reason: string;
      }
    | { kind: 'drop'; fact: ExtractedFact; reason: string };

  const ops: ResolvedOp[] = [];

  // Track the (subject, category) of every existing fact we touch, so a refresh
  // pass after the writeTx can rebuild affected cluster summaries — including
  // when an UPDATE moves a fact between categories.
  const oldFactCluster = new Map<number, { subject: string; category: string }>();

  // Embeddings produced during consolidation (large-subject path). Reused in
  // the writeTx for ADD/UPDATE so we don't pay to embed the same content twice.
  const precomputedEmbeddings = new Map<
    ExtractedFact,
    { vector: number[]; usage: { model: string; tokens_in: number; tokens_out: number } }
  >();

  // Each subject group is independent at the LLM-prompt level — consolidate
  // only sees one subject's existing+candidate facts. Run them with bounded
  // concurrency, then merge results into shared state in input order so the
  // ops stream and oldFactCluster map are deterministic across runs.
  const subjectGroups = [...groups.entries()];
  const groupResults = await mapWithConcurrency(
    subjectGroups,
    LLM_CONCURRENCY,
    async ([subjectKey, candidates]) => {
      const subjectStored = subjectKey;
      const localOps: ResolvedOp[] = [];
      const localOldFactCluster: Array<[number, { subject: string; category: string }]> = [];

      const activeCount = countActiveFactsForSubject(db, subjectKey);

      if (activeCount === 0) {
        for (const c of candidates) localOps.push({ kind: 'add', subject: subjectStored, fact: c });
        return { ops: localOps, oldFactClusterEntries: localOldFactCluster };
      }

      let existing: ActiveFactRow[];
      if (activeCount <= CONSOLIDATION_FULL_LIST_THRESHOLD) {
        existing = existingFactsForSubject(db, subjectKey);
      } else {
        existing = await selectExistingByVectorSim(
          db,
          provider,
          subjectKey,
          candidates,
          precomputedEmbeddings,
          burst.id,
          log
        );
        // Vector search may return zero rows for very off-topic candidates;
        // fall back to recency so the LLM has *something* to compare against.
        if (existing.length === 0) {
          existing = existingFactsForSubject(db, subjectKey);
        }
      }

      for (const e of existing) {
        localOldFactCluster.push([e.id, { subject: e.subject_wa_id, category: e.category }]);
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
        if (resolved) localOps.push(resolved);
      }
      for (let i = 0; i < candidates.length; i++) {
        if (!consumedCandidates.has(i)) {
          log(
            `  burst ${burst.id} CONSOLIDATE skipped candidate ${i} ("${candidates[i].content}") — defaulting to ADD`
          );
          localOps.push({ kind: 'add', subject: subjectStored, fact: candidates[i] });
        }
      }

      return { ops: localOps, oldFactClusterEntries: localOldFactCluster };
    }
  );

  for (const r of groupResults) {
    for (const [id, val] of r.oldFactClusterEntries) {
      oldFactCluster.set(id, val);
    }
    ops.push(...r.ops);
  }

  const toEmbed = ops.filter(
    (o): o is Extract<ResolvedOp, { kind: 'add' | 'connect' }> =>
      o.kind === 'add' || o.kind === 'connect'
  );
  const embedded = new Map<ResolvedOp, { vector: number[]; usage: { model: string; tokens_in: number; tokens_out: number } }>();
  const needsEmbed: typeof toEmbed = [];
  for (const o of toEmbed) {
    const reused = precomputedEmbeddings.get(o.fact);
    if (reused) embedded.set(o, reused);
    else needsEmbed.push(o);
  }
  const fresh = await mapWithConcurrency(needsEmbed, LLM_CONCURRENCY, (o) =>
    provider.embed(o.fact.content)
  );
  for (let i = 0; i < needsEmbed.length; i++) {
    embedded.set(needsEmbed[i], { vector: fresh[i].vector, usage: fresh[i].usage });
  }

  const result: BurstResult = {
    added: 0,
    updated: 0,
    deleted: 0,
    dropped: 0,
    guarded: guarded.length,
    clusters_refreshed: 0,
    clusters_deleted: 0,
  };

  const affectedClusters = new Set<string>();
  const recordCluster = (subject: string, category: string) => {
    affectedClusters.add(`${subject}|${category}`);
  };

  // Captured inside writeTx so the thread-assignment stage (post-tx, outside
  // the transaction) can attach each new fact to a memory_thread. Holds new
  // facts produced by ADD or CONNECT ops, grouped by subject.
  interface NewFactRecord {
    fact_id: number;
    category: ExtractedFact['category'];
    content: string;
    confidence: number;
  }
  const newFactsBySubject = new Map<string, NewFactRecord[]>();
  const recordNewFact = (subject: string, rec: NewFactRecord) => {
    if (!newFactsBySubject.has(subject)) newFactsBySubject.set(subject, []);
    newFactsBySubject.get(subject)!.push(rec);
  };

  const writeTx = db.transaction(() => {
    const graphJobs: GraphFactContext[] = [];
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
          event_ts: o.fact.event_ts ?? null,
          extracted_at: burst.end_ts,
        });
        insertEmbedding(db, factId, e.vector);
        recordNewFact(o.subject, {
          fact_id: factId,
          category: o.fact.category,
          content: o.fact.content,
          confidence: o.fact.confidence,
        });
        if (GRAPH_ENABLED) {
          graphJobs.push({
            fact_id: factId,
            subject: o.subject,
            category: o.fact.category,
            content: o.fact.content,
            confidence: o.fact.confidence,
            event_ts: o.fact.event_ts ?? null,
            source_burst_id: burst.id,
          });
        }
        logProcessing(db, {
          burst_id: burst.id,
          stage: 'embed',
          model: e.usage.model,
          tokens_in: e.usage.tokens_in,
          tokens_out: e.usage.tokens_out,
        });
        result.added++;
        recordCluster(o.subject, o.fact.category);
        log(`  burst ${burst.id} ADD [${o.fact.category}] about ${o.subject}: ${o.fact.content}`);
      } else if (o.kind === 'connect') {
        const e = embedded.get(o)!;
        const newId = insertFact(db, {
          source_burst_id: burst.id,
          source_msg_id: null,
          subject: o.subject,
          category: o.fact.category,
          content: o.fact.content,
          confidence: o.fact.confidence,
          event_ts: o.fact.event_ts ?? null,
          extracted_at: burst.end_ts,
        });
        insertEmbedding(db, newId, e.vector);
        recordNewFact(o.subject, {
          fact_id: newId,
          category: o.fact.category,
          content: o.fact.content,
          confidence: o.fact.confidence,
        });
        // Append-only: don't supersede or delete the old fact. Record a
        // typed connection from the new fact back to the old one. The old
        // row stays active and discoverable via retrieval; latestInChain()
        // walks update/state_change edges at read time.
        insertFactConnection(db, {
          from_fact_id: newId,
          to_fact_id: o.oldId,
          predicate: o.predicate,
          confidence: o.fact.confidence,
          reason: o.reason,
        });
        if (GRAPH_ENABLED) {
          graphJobs.push({
            fact_id: newId,
            subject: o.subject,
            category: o.fact.category,
            content: o.fact.content,
            confidence: o.fact.confidence,
            event_ts: o.fact.event_ts ?? null,
            source_burst_id: burst.id,
          });
        }
        logProcessing(db, {
          burst_id: burst.id,
          stage: 'embed',
          model: e.usage.model,
          tokens_in: e.usage.tokens_in,
          tokens_out: e.usage.tokens_out,
        });
        result.updated++;
        recordCluster(o.subject, o.fact.category);
        const old = oldFactCluster.get(o.oldId);
        if (old && old.category !== o.fact.category) recordCluster(old.subject, old.category);
        log(
          `  burst ${burst.id} CONNECT(${o.predicate}) fact ${newId} → ${o.oldId} ` +
            `about ${o.subject}: ${o.fact.content}`
        );
      } else {
        result.dropped++;
        log(`  burst ${burst.id} DROP candidate "${o.fact.content}" — ${o.reason}`);
      }
    }
    markBurstFiltered(db, burst.id, true);
    markBurstProcessed(db, burst.id);
    return graphJobs;
  });
  const graphJobs = writeTx();

  if (GRAPH_ENABLED && graphJobs.length > 0) {
    // Phase 1: parallel LLM extraction. Phase 2 below walks results in input
    // order and applies DB writes serially, since entity upserts and the
    // post-write trigger pass aren't safe to interleave.
    const llmResults = await mapWithConcurrency(graphJobs, LLM_CONCURRENCY, async (job) => {
      try {
        const graphResult = await provider.extractGraphFromFact(job);
        return { ok: true as const, job, graphResult };
      } catch (err) {
        return { ok: false as const, job, err };
      }
    });
    for (const r of llmResults) {
      if (!r.ok) {
        log(
          `  burst ${burst.id} GRAPH fact ${r.job.fact_id} ERROR: ${
            r.err instanceof Error ? r.err.message : r.err
          }`
        );
        continue;
      }
      const written = writeExtractedGraph(db, r.job, r.graphResult.graph);
      logProcessing(db, {
        burst_id: burst.id,
        stage: 'graph_extract',
        model: r.graphResult.usage.model,
        tokens_in: r.graphResult.usage.tokens_in,
        tokens_out: r.graphResult.usage.tokens_out,
      });
      log(
        `  burst ${burst.id} GRAPH fact ${r.job.fact_id}: entities=${written.entities} mentions=${written.mentions} edges=${written.edges}`
      );

      // Entity-signal trigger: if this fact's graph touches an established
      // entity via a type-defining predicate, queue a curator run. Cheap
      // (DB-only) — actual execution is deferred to drainPlannedAgentRuns.
      const triggered = planTriggeredRunsForFact(db, r.job.fact_id);
      if (triggered.length > 0) {
        log(
          `  burst ${burst.id} TRIGGER fact ${r.job.fact_id}: queued curator runs ${triggered.join(',')}`
        );
      }
    }
  }

  // Thread assignment: one LLM call per subject in this burst attaches each
  // new fact to existing memory_threads or creates new ones. Failures here
  // are logged but don't fail the burst — fact rows are already committed.
  // Subjects are owner-scoped on memory_threads, so per-subject writes don't
  // conflict across iterations — safe to fully parallelize.
  const threadAssignTargets = [...newFactsBySubject.entries()].filter(
    ([, facts]) => facts.length > 0
  );
  await mapWithConcurrency(threadAssignTargets, LLM_CONCURRENCY, async ([subject, newFacts]) => {
    try {
      const existing = listMemoryThreads(db, { owner_subject_wa_id: subject });
      const assignmentResult = await provider.assignThreads({
        subject,
        contactDisplayName: burst.chat_display_name ?? null,
        existing: existing.map((t) => ({
          id: t.id,
          name: t.name,
          description: t.description,
        })),
        facts: newFacts.map((f) => ({
          id: f.fact_id,
          category: f.category,
          content: f.content,
          confidence: f.confidence,
        })),
      });
      logProcessing(db, {
        burst_id: burst.id,
        stage: 'extract',
        model: assignmentResult.usage.model + ' [thread-assign]',
        tokens_in: assignmentResult.usage.tokens_in,
        tokens_out: assignmentResult.usage.tokens_out,
      });

      const localToThreadId = new Map<string, number>();
      for (const t of assignmentResult.result.new_threads) {
        const id = createMemoryThread(db, {
          name: t.name,
          description: t.description ?? null,
          owner_subject_wa_id: subject,
        });
        localToThreadId.set(t.local_id, id);
      }

      let attachedCount = 0;
      for (const a of assignmentResult.result.assignments) {
        const threadIds = new Set<number>(a.existing_thread_ids);
        for (const localId of a.new_thread_local_ids) {
          const id = localToThreadId.get(localId);
          if (id) threadIds.add(id);
        }
        for (const tid of threadIds) {
          if (addFactToThread(db, { fact_id: a.fact_id, thread_id: tid })) {
            attachedCount++;
          }
        }
      }
      log(
        `  burst ${burst.id} THREADS subject=${subject}: ` +
          `${newFacts.length} fact(s), ${assignmentResult.result.new_threads.length} new thread(s), ` +
          `${attachedCount} membership(s)`
      );
    } catch (err) {
      log(
        `  burst ${burst.id} THREADS subject=${subject} ERROR: ${
          err instanceof Error ? err.message : err
        }`
      );
    }
  });

  const clusterKeys = [...affectedClusters].map((key) => {
    const sep = key.indexOf('|');
    return { subject: key.slice(0, sep), category: key.slice(sep + 1) };
  });
  const clusterResults = await mapWithConcurrency(
    clusterKeys,
    LLM_CONCURRENCY,
    async ({ subject, category }) => {
      try {
        return await refreshClusterSummary(db, provider, subject, category, {
          burstId: burst.id,
          log,
        });
      } catch (err) {
        log(
          `  burst ${burst.id} CLUSTER ${subject}/${category} refresh ERROR: ${
            err instanceof Error ? err.message : err
          }`
        );
        return 'error' as const;
      }
    }
  );
  for (const r of clusterResults) {
    if (r === 'refreshed') result.clusters_refreshed++;
    else if (r === 'deleted') result.clusters_deleted++;
  }

  return result;
}

/**
 * Build the "existing facts" slate for the consolidator when a subject has
 * too many active facts to send them all. Embeds each candidate, runs a
 * subject-restricted vector search, unions the top-N per candidate, and
 * caches the embeddings so the writeTx can reuse them for ADD/UPDATE rows.
 */
async function selectExistingByVectorSim(
  db: DB,
  provider: LLMProvider,
  subject: string,
  candidates: ExtractedFact[],
  cache: Map<
    ExtractedFact,
    { vector: number[]; usage: { model: string; tokens_in: number; tokens_out: number } }
  >,
  burstId: number,
  log: (line: string) => void
): Promise<ActiveFactRow[]> {
  const embeds = await mapWithConcurrency(candidates, LLM_CONCURRENCY, (c) =>
    provider.embed(c.content)
  );
  const merged = new Map<number, ActiveFactRow>();
  for (let i = 0; i < candidates.length; i++) {
    const c = candidates[i];
    const r = embeds[i];
    cache.set(c, { vector: r.vector, usage: r.usage });
    logProcessing(db, {
      burst_id: burstId,
      stage: 'embed',
      model: r.usage.model + ' [pre-consolidate]',
      tokens_in: r.usage.tokens_in,
      tokens_out: r.usage.tokens_out,
    });
    const hits = searchFactsForSubjectByVector(
      db,
      r.vector,
      subject,
      CONSOLIDATION_SIMILAR_PER_CANDIDATE
    );
    for (const h of hits) merged.set(h.id, h);
  }
  log(
    `  burst ${burstId} CONSOLIDATE subject=${subject} vec-selected ${merged.size} existing for ${candidates.length} candidate(s)`
  );
  return [...merged.values()];
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
  | {
      kind: 'connect';
      subject: string;
      fact: ExtractedFact;
      oldId: number;
      predicate: ConnectionPredicate;
      reason: string;
    }
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
        // Consolidate doesn't see/return event_ts — pull from the original
        // candidate. The consolidate prompt copies content/category/confidence
        // verbatim, so the candidate's event_ts is the right anchor for the
        // resulting fact.
        event_ts: candidates[op.candidate_index]?.event_ts ?? null,
      },
    };
  }
  if (op.op === 'CONNECT') {
    if (consumedCandidates.has(op.candidate_index)) {
      log(`  burst ${burstId} CONSOLIDATE error: candidate ${op.candidate_index} consumed twice`);
      return null;
    }
    consumedCandidates.add(op.candidate_index);
    return {
      kind: 'connect',
      subject,
      oldId: op.old_fact_id,
      predicate: op.predicate,
      reason: op.reason,
      fact: {
        subject,
        category: op.category,
        content: op.content,
        confidence: op.confidence,
        event_ts: candidates[op.candidate_index]?.event_ts ?? null,
      },
    };
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
