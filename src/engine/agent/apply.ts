/**
 * Apply path for curator agent runs.
 *
 * `applyAgentRun(runId, approvedBy)` walks every action in 'proposed' status
 * for the run and turns it into a real fact mutation, in seq order:
 *   - update → insertFact + markFactSuperseded(target → new) + carry sources
 *   - delete → markFactDeleted(target)
 *   - merge  → insertFact + markFactSuperseded(each merged_id → new) + union sources
 *
 * Pre-flight: every action's target_fact_id, merge_fact_ids, and citing_fact_ids
 * must still be active. If anything is stale (already superseded/deleted by
 * another path between propose and apply), the action is marked 'skipped' with
 * a reason — we never apply a half-stale proposal.
 *
 * The fact mutations + action status writes happen in a single transaction so
 * the DB is never observed in a half-applied state. Outside the transaction
 * we then: embed each new fact (Cohere), refresh affected cluster summaries
 * (LLM), and finally mark the run 'applied'. If embedding or cluster refresh
 * fails for one cluster, the others still complete — the new fact rows stay
 * canonical regardless.
 *
 * Boundary: this module imports DB helpers and the LLM provider only. All web
 * / cli consumers go through src/engine/index.ts.
 */

import {
  copyFactSourcesUnion,
  getActiveFact,
  getAgentAction,
  getAgentRun,
  insertAgentDerivedFact,
  insertEmbedding,
  listAgentActionsForRun,
  logProcessing,
  markFactDeleted,
  markFactSuperseded,
  setAgentActionApplied,
  setAgentActionRejected,
  setAgentActionSkipped,
  setAgentRunApplied,
  setAgentRunRejected,
  type AgentActionRow,
  type AgentRunRow,
  type DB,
} from '../storage/db';
import { refreshClusterSummary } from '../cluster';
import type { LLMProvider } from '../../llm/provider';

export interface ApplyAgentRunOptions {
  approvedBy: string;
  log?: (line: string) => void;
}

export interface ApplyAgentRunResult {
  run: AgentRunRow;
  applied: number;
  skipped: number;
  rejected: number;
  clusters_refreshed: number;
  clusters_deleted: number;
  errors: string[];
}

interface PlannedApply {
  action: AgentActionRow;
  /** Resolved post-validation. New fact rows aren't created until inside the tx. */
  plan:
    | {
        kind: 'update';
        targetFact: { id: number; subject: string; category: string; event_ts: number | null };
        newContent: string;
        newCategory: string;
        sourceFactIdsForUnion: number[];
      }
    | {
        kind: 'delete';
        targetFact: { id: number; subject: string; category: string };
      }
    | {
        kind: 'merge';
        subject: string;
        canonicalContent: string;
        canonicalCategory: string;
        mergedFacts: Array<{ id: number; category: string; event_ts: number | null }>;
        sourceFactIdsForUnion: number[];
      };
}

interface SkipPlan {
  action: AgentActionRow;
  reason: string;
}

interface NewFactJob {
  factId: number;
  content: string;
  subject: string;
  category: string;
}

interface ClusterKey {
  subject: string;
  category: string;
}

export async function applyAgentRun(
  db: DB,
  provider: LLMProvider,
  runId: number,
  opts: ApplyAgentRunOptions
): Promise<ApplyAgentRunResult> {
  const log = opts.log ?? (() => {});
  const run = getAgentRun(db, runId);
  if (!run) throw new Error(`applyAgentRun: run ${runId} not found`);
  if (run.status !== 'proposed') {
    throw new Error(
      `applyAgentRun: run ${runId} is in status ${run.status}, expected 'proposed'`
    );
  }

  const allActions = listAgentActionsForRun(db, runId);
  const proposed = allActions.filter((a) => a.status === 'proposed');

  const planned: PlannedApply[] = [];
  const toSkip: SkipPlan[] = [];
  for (const action of proposed) {
    const result = planActionApply(db, action);
    if ('reason' in result) toSkip.push({ action, reason: result.reason });
    else planned.push({ action, plan: result });
  }

  const errors: string[] = [];
  const newFactJobs: NewFactJob[] = [];
  const affectedClusters = new Map<string, ClusterKey>();
  const recordCluster = (subject: string, category: string) => {
    affectedClusters.set(`${subject}|${category}`, { subject, category });
  };

  // ───── Phase 1: tx — fact mutations + per-action status updates ─────
  const writeTx = db.transaction(() => {
    // Skip stale actions.
    for (const s of toSkip) {
      setAgentActionSkipped(db, s.action.id, s.reason);
      log(`  action ${s.action.id} SKIPPED: ${s.reason}`);
    }

    // Apply planned actions.
    for (const p of planned) {
      try {
        if (p.plan.kind === 'update') {
          const newId = insertAgentDerivedFact(db, {
            subject: p.plan.targetFact.subject,
            category: p.plan.newCategory,
            content: p.plan.newContent,
            confidence: p.action.confidence,
            event_ts: p.plan.targetFact.event_ts,
          });
          copyFactSourcesUnion(db, p.plan.sourceFactIdsForUnion, newId);
          markFactSuperseded(db, p.plan.targetFact.id, newId);
          setAgentActionApplied(db, p.action.id, newId);
          recordCluster(p.plan.targetFact.subject, p.plan.targetFact.category);
          if (p.plan.newCategory !== p.plan.targetFact.category) {
            recordCluster(p.plan.targetFact.subject, p.plan.newCategory);
          }
          newFactJobs.push({
            factId: newId,
            content: p.plan.newContent,
            subject: p.plan.targetFact.subject,
            category: p.plan.newCategory,
          });
          log(
            `  action ${p.action.id} UPDATE fact ${p.plan.targetFact.id} → ${newId}`
          );
        } else if (p.plan.kind === 'delete') {
          markFactDeleted(db, p.plan.targetFact.id);
          setAgentActionApplied(db, p.action.id, null);
          recordCluster(p.plan.targetFact.subject, p.plan.targetFact.category);
          log(`  action ${p.action.id} DELETE fact ${p.plan.targetFact.id}`);
        } else {
          // merge
          const newId = insertAgentDerivedFact(db, {
            subject: p.plan.subject,
            category: p.plan.canonicalCategory,
            content: p.plan.canonicalContent,
            confidence: p.action.confidence,
            event_ts: null,
          });
          copyFactSourcesUnion(db, p.plan.sourceFactIdsForUnion, newId);
          for (const m of p.plan.mergedFacts) {
            markFactSuperseded(db, m.id, newId);
            recordCluster(p.plan.subject, m.category);
          }
          recordCluster(p.plan.subject, p.plan.canonicalCategory);
          setAgentActionApplied(db, p.action.id, newId);
          newFactJobs.push({
            factId: newId,
            content: p.plan.canonicalContent,
            subject: p.plan.subject,
            category: p.plan.canonicalCategory,
          });
          log(
            `  action ${p.action.id} MERGE [${p.plan.mergedFacts
              .map((m) => m.id)
              .join(',')}] → ${newId}`
          );
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        errors.push(`action ${p.action.id}: ${message}`);
        // If something throws inside the tx, the DB call will be rolled back
        // when we re-throw below. We mark the action skipped before throwing
        // so the audit trail records why.
        throw err;
      }
    }
  });

  try {
    writeTx();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    log(`apply tx FAILED, rolled back: ${message}`);
    setAgentRunRejected(db, runId, opts.approvedBy);
    return {
      run: getAgentRun(db, runId)!,
      applied: 0,
      skipped: 0,
      rejected: 0,
      clusters_refreshed: 0,
      clusters_deleted: 0,
      errors: [message],
    };
  }

  // ───── Phase 2: embed each new fact (outside tx) ─────
  for (const job of newFactJobs) {
    try {
      const r = await provider.embed(job.content, 'document');
      insertEmbedding(db, job.factId, r.vector);
      logProcessing(db, {
        burst_id: null,
        stage: 'embed',
        model: r.usage.model,
        tokens_in: r.usage.tokens_in,
        tokens_out: r.usage.tokens_out,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      errors.push(`embed fact ${job.factId}: ${message}`);
      log(`  embed fact ${job.factId} ERROR: ${message}`);
    }
  }

  // ───── Phase 3: refresh affected cluster summaries ─────
  let clustersRefreshed = 0;
  let clustersDeleted = 0;
  for (const { subject, category } of affectedClusters.values()) {
    try {
      const result = await refreshClusterSummary(db, provider, subject, category, {
        burstId: null,
        log,
      });
      if (result === 'refreshed') clustersRefreshed++;
      else if (result === 'deleted') clustersDeleted++;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      errors.push(`cluster ${subject}/${category}: ${message}`);
      log(`  CLUSTER ${subject}/${category} refresh ERROR: ${message}`);
    }
  }

  // ───── Phase 4: mark run applied ─────
  const finalActions = listAgentActionsForRun(db, runId);
  const appliedCount = finalActions.filter((a) => a.status === 'applied').length;
  const skippedCount = finalActions.filter((a) => a.status === 'skipped').length;
  const rejectedCount = finalActions.filter((a) => a.status === 'rejected').length;

  if (appliedCount > 0) {
    setAgentRunApplied(db, runId, opts.approvedBy);
  } else {
    setAgentRunRejected(db, runId, opts.approvedBy);
  }

  return {
    run: getAgentRun(db, runId)!,
    applied: appliedCount,
    skipped: skippedCount,
    rejected: rejectedCount,
    clusters_refreshed: clustersRefreshed,
    clusters_deleted: clustersDeleted,
    errors,
  };
}

/**
 * Apply a single proposed action without mutating run-level status. Useful
 * for piecemeal review/approval. The caller is responsible for eventually
 * calling applyAgentRun (which will skip already-applied/-rejected actions
 * and finalize run status) or finalizing the run themselves.
 */
export interface ApplyAgentActionOptions {
  approvedBy: string;
  log?: (line: string) => void;
}

export interface ApplyAgentActionResult {
  action: AgentActionRow;
  applied: boolean;
  skipped_reason?: string;
  new_fact_id?: number | null;
  errors: string[];
}

export async function applyAgentAction(
  db: DB,
  provider: LLMProvider,
  actionId: number,
  opts: ApplyAgentActionOptions
): Promise<ApplyAgentActionResult> {
  const log = opts.log ?? (() => {});
  const action = getAgentAction(db, actionId);
  if (!action) throw new Error(`applyAgentAction: action ${actionId} not found`);
  if (action.status !== 'proposed') {
    throw new Error(
      `applyAgentAction: action ${actionId} is in status ${action.status}, expected 'proposed'`
    );
  }

  const planResult = planActionApply(db, action);
  if ('reason' in planResult) {
    setAgentActionSkipped(db, actionId, planResult.reason);
    return {
      action: getAgentAction(db, actionId)!,
      applied: false,
      skipped_reason: planResult.reason,
      errors: [],
    };
  }

  const newFactJobs: NewFactJob[] = [];
  const affectedClusters = new Map<string, ClusterKey>();
  const recordCluster = (subject: string, category: string) => {
    affectedClusters.set(`${subject}|${category}`, { subject, category });
  };

  const tx = db.transaction(() => {
    if (planResult.kind === 'update') {
      const newId = insertAgentDerivedFact(db, {
        subject: planResult.targetFact.subject,
        category: planResult.newCategory,
        content: planResult.newContent,
        confidence: action.confidence,
        event_ts: planResult.targetFact.event_ts,
      });
      copyFactSourcesUnion(db, planResult.sourceFactIdsForUnion, newId);
      markFactSuperseded(db, planResult.targetFact.id, newId);
      setAgentActionApplied(db, actionId, newId);
      recordCluster(planResult.targetFact.subject, planResult.targetFact.category);
      if (planResult.newCategory !== planResult.targetFact.category) {
        recordCluster(planResult.targetFact.subject, planResult.newCategory);
      }
      newFactJobs.push({
        factId: newId,
        content: planResult.newContent,
        subject: planResult.targetFact.subject,
        category: planResult.newCategory,
      });
      return newId;
    } else if (planResult.kind === 'delete') {
      markFactDeleted(db, planResult.targetFact.id);
      setAgentActionApplied(db, actionId, null);
      recordCluster(planResult.targetFact.subject, planResult.targetFact.category);
      return null;
    } else {
      const newId = insertAgentDerivedFact(db, {
        subject: planResult.subject,
        category: planResult.canonicalCategory,
        content: planResult.canonicalContent,
        confidence: action.confidence,
        event_ts: null,
      });
      copyFactSourcesUnion(db, planResult.sourceFactIdsForUnion, newId);
      for (const m of planResult.mergedFacts) {
        markFactSuperseded(db, m.id, newId);
        recordCluster(planResult.subject, m.category);
      }
      recordCluster(planResult.subject, planResult.canonicalCategory);
      setAgentActionApplied(db, actionId, newId);
      newFactJobs.push({
        factId: newId,
        content: planResult.canonicalContent,
        subject: planResult.subject,
        category: planResult.canonicalCategory,
      });
      return newId;
    }
  });

  let newFactId: number | null = null;
  const errors: string[] = [];
  try {
    newFactId = tx();
    log(`  action ${actionId} APPLIED → fact ${newFactId ?? '(deleted)'}`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    errors.push(message);
    return {
      action: getAgentAction(db, actionId)!,
      applied: false,
      errors,
    };
  }

  for (const job of newFactJobs) {
    try {
      const r = await provider.embed(job.content, 'document');
      insertEmbedding(db, job.factId, r.vector);
      logProcessing(db, {
        burst_id: null,
        stage: 'embed',
        model: r.usage.model,
        tokens_in: r.usage.tokens_in,
        tokens_out: r.usage.tokens_out,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      errors.push(`embed fact ${job.factId}: ${message}`);
    }
  }
  for (const { subject, category } of affectedClusters.values()) {
    try {
      await refreshClusterSummary(db, provider, subject, category, {
        burstId: null,
        log,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      errors.push(`cluster ${subject}/${category}: ${message}`);
    }
  }

  return {
    action: getAgentAction(db, actionId)!,
    applied: true,
    new_fact_id: newFactId,
    errors,
  };
}

export function rejectAgentAction(db: DB, actionId: number, reason: string): AgentActionRow {
  const action = getAgentAction(db, actionId);
  if (!action) throw new Error(`rejectAgentAction: action ${actionId} not found`);
  if (action.status !== 'proposed') {
    throw new Error(
      `rejectAgentAction: action ${actionId} is in status ${action.status}, expected 'proposed'`
    );
  }
  setAgentActionRejected(db, actionId, reason);
  return getAgentAction(db, actionId)!;
}

// ───────────── planning / staleness ─────────────

/**
 * Validate that the action's referenced facts are all still active and
 * resolve the materialized info needed for the apply transaction. Returns
 * either a typed `plan` object or `{ reason }` indicating the action must
 * be skipped.
 */
function planActionApply(
  db: DB,
  action: AgentActionRow
):
  | NonNullable<PlannedApply['plan']>
  | { reason: string } {
  // Citing facts must still be active. If they're gone, the agent's rationale
  // has lost its source — better to skip than to apply a now-unsourced change.
  for (const cid of action.citing_fact_ids) {
    if (!getActiveFact(db, cid)) {
      return { reason: `citing_fact_id ${cid} is no longer active` };
    }
  }

  if (action.op === 'update') {
    if (!action.target_fact_id) {
      return { reason: 'update action has no target_fact_id' };
    }
    const target = getActiveFact(db, action.target_fact_id);
    if (!target) {
      return { reason: `target_fact_id ${action.target_fact_id} is no longer active` };
    }
    if (!action.new_content) {
      return { reason: 'update action has no new_content' };
    }
    const newCategory = action.new_category ?? target.category;
    return {
      kind: 'update',
      targetFact: {
        id: target.id,
        subject: target.subject_wa_id,
        category: target.category,
        event_ts: target.event_ts,
      },
      newContent: action.new_content,
      newCategory,
      sourceFactIdsForUnion: dedup([target.id, ...action.citing_fact_ids]),
    };
  }

  if (action.op === 'delete') {
    if (!action.target_fact_id) {
      return { reason: 'delete action has no target_fact_id' };
    }
    const target = getActiveFact(db, action.target_fact_id);
    if (!target) {
      return { reason: `target_fact_id ${action.target_fact_id} is no longer active` };
    }
    return {
      kind: 'delete',
      targetFact: {
        id: target.id,
        subject: target.subject_wa_id,
        category: target.category,
      },
    };
  }

  if (action.op === 'merge') {
    if (!action.merge_fact_ids || action.merge_fact_ids.length < 2) {
      return { reason: 'merge action has < 2 merge_fact_ids' };
    }
    if (!action.new_content || !action.new_category) {
      return { reason: 'merge action missing canonical content/category' };
    }
    const mergedFacts: Array<{ id: number; category: string; event_ts: number | null }> = [];
    let subject: string | null = null;
    for (const fid of action.merge_fact_ids) {
      const f = getActiveFact(db, fid);
      if (!f) {
        return { reason: `merge fact ${fid} is no longer active` };
      }
      if (subject === null) subject = f.subject_wa_id;
      else if (f.subject_wa_id !== subject) {
        return {
          reason: `merge fact ${fid} subject mismatch (expected ${subject}, got ${f.subject_wa_id})`,
        };
      }
      mergedFacts.push({ id: f.id, category: f.category, event_ts: f.event_ts });
    }
    return {
      kind: 'merge',
      subject: subject!,
      canonicalContent: action.new_content,
      canonicalCategory: action.new_category,
      mergedFacts,
      sourceFactIdsForUnion: dedup([
        ...action.merge_fact_ids,
        ...action.citing_fact_ids,
      ]),
    };
  }

  return { reason: `unknown action op: ${action.op}` };
}

function dedup<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}
