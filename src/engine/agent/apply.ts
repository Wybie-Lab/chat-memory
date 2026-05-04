/**
 * Apply path for curator agent runs (append-only memory model).
 *
 * The curator's job is to ORGANIZE existing memory, not mutate it. The apply
 * path therefore only writes graph-structure rows:
 *
 *   - connect       → fact_connections row between two existing facts
 *   - assign_thread → fact_thread_membership row
 *   - create_thread → memory_threads row + (optional) memberships
 *
 * No fact mutations, no embeddings, no cluster summary refresh — the underlying
 * facts are unchanged. Pre-flight checks each action's referenced facts/threads
 * are still active. Stale references → action 'skipped' with a reason.
 *
 * All writes for a single applyAgentRun() happen in one transaction so the run
 * is observable in only two states: pre-apply (all actions 'proposed') and
 * post-apply (each action 'applied' / 'skipped' / 'rejected'; run 'applied').
 */

import {
  addFactToThread,
  createMemoryThread,
  getActiveFact,
  getAgentAction,
  getAgentRun,
  getMemoryThread,
  insertFactConnection,
  listAgentActionsForRun,
  setAgentActionApplied,
  setAgentActionRejected,
  setAgentActionSkipped,
  setAgentRunApplied,
  setAgentRunRejected,
  type AgentActionOp,
  type AgentActionRow,
  type AgentRunRow,
  type ConnectionPredicate,
  type DB,
} from '../storage/db';
import { CONNECTION_PREDICATES } from '../storage/db';
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
  errors: string[];
}

interface PlannedConnect {
  kind: 'connect';
  action: AgentActionRow;
  fromFactId: number;
  toFactId: number;
  predicate: ConnectionPredicate;
}
interface PlannedAssignThread {
  kind: 'assign_thread';
  action: AgentActionRow;
  factId: number;
  threadId: number;
}
interface PlannedCreateThread {
  kind: 'create_thread';
  action: AgentActionRow;
  name: string;
  description: string | null;
  ownerSubjectWaId: string | null;
  attachedFactIds: number[];
}
type PlannedApply = PlannedConnect | PlannedAssignThread | PlannedCreateThread;

interface SkipPlan {
  action: AgentActionRow;
  reason: string;
}

export async function applyAgentRun(
  db: DB,
  // provider param kept for signature stability with older callers; the
  // new-model apply path doesn't need an LLM provider (no embeddings,
  // no cluster summarize). Accept any value, ignore it.
  _provider: LLMProvider,
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
    else planned.push(result);
  }

  const errors: string[] = [];

  // ───── Single transaction for all writes + per-action status ─────
  const writeTx = db.transaction(() => {
    for (const s of toSkip) {
      setAgentActionSkipped(db, s.action.id, s.reason);
      log(`  action ${s.action.id} SKIPPED: ${s.reason}`);
    }

    for (const p of planned) {
      try {
        if (p.kind === 'connect') {
          insertFactConnection(db, {
            from_fact_id: p.fromFactId,
            to_fact_id: p.toFactId,
            predicate: p.predicate,
            confidence: p.action.confidence,
            reason: p.action.reason,
            source_agent_action_id: p.action.id,
          });
          setAgentActionApplied(db, p.action.id, { fact_id: null, thread_id: null });
          log(
            `  action ${p.action.id} CONNECT(${p.predicate}) ` +
              `${p.fromFactId} → ${p.toFactId}`
          );
        } else if (p.kind === 'assign_thread') {
          addFactToThread(db, {
            fact_id: p.factId,
            thread_id: p.threadId,
            source_agent_action_id: p.action.id,
          });
          setAgentActionApplied(db, p.action.id, {
            fact_id: p.factId,
            thread_id: p.threadId,
          });
          log(
            `  action ${p.action.id} ASSIGN_THREAD fact ${p.factId} → thread ${p.threadId}`
          );
        } else {
          // create_thread
          const threadId = createMemoryThread(db, {
            name: p.name,
            description: p.description,
            owner_subject_wa_id: p.ownerSubjectWaId,
          });
          for (const fid of p.attachedFactIds) {
            addFactToThread(db, {
              fact_id: fid,
              thread_id: threadId,
              source_agent_action_id: p.action.id,
            });
          }
          setAgentActionApplied(db, p.action.id, { thread_id: threadId });
          log(
            `  action ${p.action.id} CREATE_THREAD "${p.name}" → ${threadId}` +
              (p.attachedFactIds.length
                ? ` (+${p.attachedFactIds.length} fact(s) attached)`
                : '')
          );
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        errors.push(`action ${p.action.id}: ${message}`);
        // Rethrow to roll back the whole transaction — partial application is
        // worse than none.
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
      errors: [message],
    };
  }

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
    errors,
  };
}

/** Apply a single proposed action. Useful for piecemeal review/approval. */
export interface ApplyAgentActionOptions {
  approvedBy: string;
  log?: (line: string) => void;
}

export interface ApplyAgentActionResult {
  action: AgentActionRow;
  applied: boolean;
  skipped_reason?: string;
  errors: string[];
}

export async function applyAgentAction(
  db: DB,
  _provider: LLMProvider,
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
  void opts.approvedBy; // currently no run-level finalization here

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

  const errors: string[] = [];
  const tx = db.transaction(() => {
    if (planResult.kind === 'connect') {
      insertFactConnection(db, {
        from_fact_id: planResult.fromFactId,
        to_fact_id: planResult.toFactId,
        predicate: planResult.predicate,
        confidence: action.confidence,
        reason: action.reason,
        source_agent_action_id: actionId,
      });
      setAgentActionApplied(db, actionId, {});
    } else if (planResult.kind === 'assign_thread') {
      addFactToThread(db, {
        fact_id: planResult.factId,
        thread_id: planResult.threadId,
        source_agent_action_id: actionId,
      });
      setAgentActionApplied(db, actionId, {
        fact_id: planResult.factId,
        thread_id: planResult.threadId,
      });
    } else {
      const threadId = createMemoryThread(db, {
        name: planResult.name,
        description: planResult.description,
        owner_subject_wa_id: planResult.ownerSubjectWaId,
      });
      for (const fid of planResult.attachedFactIds) {
        addFactToThread(db, {
          fact_id: fid,
          thread_id: threadId,
          source_agent_action_id: actionId,
        });
      }
      setAgentActionApplied(db, actionId, { thread_id: threadId });
    }
  });

  try {
    tx();
    log(`  action ${actionId} APPLIED (${planResult.kind})`);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    errors.push(message);
    return {
      action: getAgentAction(db, actionId)!,
      applied: false,
      errors,
    };
  }

  return {
    action: getAgentAction(db, actionId)!,
    applied: true,
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

// ───────────── plan / staleness ─────────────

function planActionApply(
  db: DB,
  action: AgentActionRow
): PlannedApply | { reason: string } {
  // Citing facts must still be active — the agent's rationale loses its
  // grounding otherwise.
  for (const cid of action.citing_fact_ids) {
    if (!getActiveFact(db, cid)) {
      return { reason: `citing_fact_id ${cid} is no longer active` };
    }
  }

  if (action.op === ('connect' as AgentActionOp)) {
    if (!action.target_fact_id) {
      return { reason: 'connect action has no target_fact_id (from-fact)' };
    }
    const secondary = action.extra.secondary_fact_id;
    const predicate = action.extra.predicate;
    if (typeof secondary !== 'number' || secondary <= 0) {
      return { reason: 'connect action extra.secondary_fact_id is missing or invalid' };
    }
    if (
      typeof predicate !== 'string' ||
      !(CONNECTION_PREDICATES as readonly string[]).includes(predicate)
    ) {
      return {
        reason: `connect action extra.predicate must be one of: ${CONNECTION_PREDICATES.join(', ')}`,
      };
    }
    if (!getActiveFact(db, action.target_fact_id)) {
      return { reason: `from_fact_id ${action.target_fact_id} is no longer active` };
    }
    if (!getActiveFact(db, secondary)) {
      return { reason: `to_fact_id ${secondary} is no longer active` };
    }
    return {
      kind: 'connect',
      action,
      fromFactId: action.target_fact_id,
      toFactId: secondary,
      predicate: predicate as ConnectionPredicate,
    };
  }

  if (action.op === ('assign_thread' as AgentActionOp)) {
    if (!action.target_fact_id) {
      return { reason: 'assign_thread action has no target_fact_id' };
    }
    const threadId = action.extra.thread_id;
    if (typeof threadId !== 'number' || threadId <= 0) {
      return { reason: 'assign_thread action extra.thread_id is missing or invalid' };
    }
    if (!getActiveFact(db, action.target_fact_id)) {
      return { reason: `fact_id ${action.target_fact_id} is no longer active` };
    }
    const thread = getMemoryThread(db, threadId);
    if (!thread || thread.deleted_at !== null) {
      return { reason: `thread_id ${threadId} not found or deleted` };
    }
    return {
      kind: 'assign_thread',
      action,
      factId: action.target_fact_id,
      threadId,
    };
  }

  if (action.op === ('create_thread' as AgentActionOp)) {
    const name = action.extra.name;
    if (typeof name !== 'string' || !name.trim()) {
      return { reason: 'create_thread action extra.name is missing' };
    }
    const description =
      typeof action.extra.description === 'string' ? action.extra.description : null;
    const ownerRaw = action.extra.owner_subject_wa_id;
    const owner =
      typeof ownerRaw === 'string'
        ? ownerRaw
        : ownerRaw === null || ownerRaw === undefined
          ? null
          : null;
    const attached = Array.isArray(action.extra.attached_fact_ids)
      ? (action.extra.attached_fact_ids as unknown[]).filter(
          (v): v is number => typeof v === 'number' && v > 0
        )
      : [];
    for (const fid of attached) {
      if (!getActiveFact(db, fid)) {
        return { reason: `attached_fact_id ${fid} is no longer active` };
      }
    }
    return {
      kind: 'create_thread',
      action,
      name: name.trim(),
      description,
      ownerSubjectWaId: owner,
      attachedFactIds: attached,
    };
  }

  return { reason: `unknown action op: ${action.op}` };
}
