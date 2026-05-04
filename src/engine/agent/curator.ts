/**
 * Curator agent loop.
 *
 * `planAgentRun` records intent (creates an `agent_runs` row in 'planned'
 * state). `runCurator` drives the LLM tool-call loop until the agent calls
 * `finish` or hits its budget. Each step the LLM sees the scope, the tool
 * catalog, and the running history of tool calls + results; it returns the
 * next batch of tool calls to dispatch.
 *
 * Important: this v1 NEVER mutates `facts`. Proposed actions land in
 * `agent_actions` (status='proposed') for separate review/apply. The whole
 * point of the run is auditability — every action carries a reason, a
 * confidence, and citing_fact_ids.
 */

import {
  countActiveFactsForSubject,
  countActiveMentionsForEntityExcluding,
  factsAboutSubject,
  getAgentRun,
  getEntityById,
  graphForFact,
  hasInFlightAgentRun,
  incrementAgentRunLlmCalls,
  insertAgentRun,
  listAgentActionsForRun,
  listFactsMentioningEntity,
  listPlannedAgentRuns,
  setAgentRunStatus,
  type AgentRunInput,
  type AgentRunRow,
  type DB,
} from '../storage/db';
import {
  buildToolCatalog,
  dispatchToolCall,
  type DispatchContext,
} from './tools';
import type {
  AgentHistoryEntry,
  AgentToolCall,
  LLMProvider,
} from '../../llm/provider';

const SYSTEM_PROMPT = `You are a memory curator. You audit a slice of long-term memory and propose targeted improvements. The slice is either one SUBJECT (a person — review all facts about them) or one ENTITY (a thing/place/relationship that newly arrived information has resolved or reframed — review facts that mention it).

Your job is NOT to add new facts from scratch — fresh facts arrive through a separate ingestion pipeline. Your job is to clean up what's already there:
- Update facts that are now better explained by newer context (e.g., a name in an old fact is now revealed to refer to a pet, a job, a place — rewrite the old fact to include that resolved context, citing the newer fact as your source).
- Delete facts that are clearly contradicted or made obsolete by newer facts.
- Merge near-duplicate facts about the same subject into one canonical fact.

Hard rules:
- Every proposed action MUST cite at least one existing fact id. You are not allowed to invent justification — every change must be grounded in something already in memory.
- For 'update', the new content must be more accurate or more contextual than the old, and you must cite the fact(s) that support that. Do not paraphrase for style alone.
- Prefer no action over a weak action. If you're not confident, finish without proposing it.
- You operate within a budget. When unsure, gather more context first (read tools are cheap; bad proposals are not).
- Call 'finish' as soon as you've completed your review. The summary should describe what you proposed and why, in 1-3 sentences.

Workflow guidance:
- Start by listing all facts about the subject (list_facts_for_subject).
- Look for: same person/pet/place mentioned across facts where one fact resolves their identity; pairs of facts that contradict; clusters of near-duplicates.
- Use get_fact_sources sparingly to disambiguate before proposing — only when the fact text alone is ambiguous.
- Use search_similar_facts when you suspect duplicates you haven't seen yet.

Return tool calls as JSON. The schema is { thinking?: string, tool_calls: [{ name, arguments }] }. You may issue multiple tool calls in one step.`;

// Predicates that "resolve" or "reframe" an entity strongly enough that
// older facts mentioning it might benefit from re-curation. We deliberately
// exclude `mentioned` (too generic), `knows` (too generic), and event-flavored
// predicates (attending/planning/visited/promised_to/needs/likes/dislikes/
// interested_in) which describe transient state, not identity.
const TYPE_DEFINING_PREDICATES = new Set<string>([
  'family_of',
  'friend_of',
  'partner_of',
  'works_at',
  'studies_at',
  'lives_in',
  'from_place',
  'located_in',
  'owns',
  'part_of',
]);

const ENTITY_TRIGGER_MIN_PRIOR_MENTIONS = 2;

const TRIGGER_BUDGET_OPS = 6;
const TRIGGER_BUDGET_LLM_CALLS = 6;

export interface PlanAgentRunInput {
  trigger: AgentRunInput['trigger'];
  scope_type: AgentRunInput['scope_type'];
  scope_ref: string;
  trigger_fact_id?: number | null;
  budget_ops?: number;
  budget_llm_calls?: number;
}

const DEFAULT_BUDGET_OPS = 8;
const DEFAULT_BUDGET_LLM_CALLS = 8;

export function planAgentRun(db: DB, input: PlanAgentRunInput): number {
  return insertAgentRun(db, {
    trigger: input.trigger,
    scope_type: input.scope_type,
    scope_ref: input.scope_ref,
    trigger_fact_id: input.trigger_fact_id ?? null,
    budget_ops: input.budget_ops ?? DEFAULT_BUDGET_OPS,
    budget_llm_calls: input.budget_llm_calls ?? DEFAULT_BUDGET_LLM_CALLS,
  });
}

/**
 * Inspect a fact's freshly-written graph projection and queue entity-scoped
 * curator runs for any entity it touches whose prior mentions cross the
 * threshold and whose predicate is type-defining. Returns the planned run
 * ids. The pipeline calls this synchronously (cheap — just DB inserts);
 * actual curator execution happens later via drainPlannedAgentRuns.
 *
 * Dedup: if a planned/running run already exists for the same entity, no
 * new run is queued — the existing one will pick up the latest state when
 * it drains.
 */
export function planTriggeredRunsForFact(db: DB, factId: number): number[] {
  const graph = graphForFact(db, factId);
  if (graph.edges.length === 0) return [];

  // Collect candidate entity ids from edges with type-defining predicates.
  const candidates = new Set<number>();
  for (const edge of graph.edges) {
    if (!TYPE_DEFINING_PREDICATES.has(edge.predicate)) continue;
    candidates.add(edge.source_entity_id);
    candidates.add(edge.target_entity_id);
  }
  if (candidates.size === 0) return [];

  const planned: number[] = [];
  for (const entityId of candidates) {
    const priorMentions = countActiveMentionsForEntityExcluding(db, entityId, factId);
    if (priorMentions < ENTITY_TRIGGER_MIN_PRIOR_MENTIONS) continue;
    const scopeRef = String(entityId);
    if (hasInFlightAgentRun(db, 'entity', scopeRef)) continue;

    const runId = planAgentRun(db, {
      trigger: 'entity_signal',
      scope_type: 'entity',
      scope_ref: scopeRef,
      trigger_fact_id: factId,
      budget_ops: TRIGGER_BUDGET_OPS,
      budget_llm_calls: TRIGGER_BUDGET_LLM_CALLS,
    });
    planned.push(runId);
  }
  return planned;
}

export interface DrainStats {
  drained: number;
  proposed_total: number;
  errors: string[];
}

export interface DrainOptions {
  limit?: number;
  log?: (line: string) => void;
}

/**
 * Process queued curator runs (status='planned'). Each run is executed in
 * sequence with the standard runCurator loop. Errors per-run are caught so
 * one bad run doesn't abort the rest. Returns aggregate stats.
 *
 * This is intentionally NOT called from the burst pipeline — entity signals
 * are queued cheaply during ingestion, and the operator (or a cron, or the
 * web UI) drains them when they're ready to spend the LLM budget.
 */
export async function drainPlannedAgentRuns(
  db: DB,
  provider: import('../../llm/provider').LLMProvider,
  opts: DrainOptions = {}
): Promise<DrainStats> {
  const log = opts.log ?? (() => {});
  const limit = opts.limit ?? 5;
  const stats: DrainStats = { drained: 0, proposed_total: 0, errors: [] };
  const runs = listPlannedAgentRuns(db, limit);
  for (const run of runs) {
    log(
      `▶ draining run ${run.id} ` +
        `(${run.trigger}, scope=${run.scope_type}:${run.scope_ref})`
    );
    try {
      const result = await runCurator(db, provider, run.id, { log });
      stats.drained++;
      stats.proposed_total += result.proposed_action_count;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      stats.errors.push(`run ${run.id}: ${message}`);
      log(`  run ${run.id} ERROR: ${message}`);
    }
  }
  return stats;
}

export interface RunCuratorResult {
  run: AgentRunRow;
  status: AgentRunRow['status'];
  proposed_action_count: number;
  llm_calls_used: number;
  reasoning: string | null;
  error: string | null;
}

export interface RunCuratorOptions {
  log?: (line: string) => void;
}

export async function runCurator(
  db: DB,
  provider: LLMProvider,
  runId: number,
  opts: RunCuratorOptions = {}
): Promise<RunCuratorResult> {
  const log = opts.log ?? (() => {});
  const run = getAgentRun(db, runId);
  if (!run) throw new Error(`runCurator: agent_runs row ${runId} not found`);
  if (run.status !== 'planned') {
    throw new Error(`runCurator: run ${runId} is in status ${run.status}, expected 'planned'`);
  }

  setAgentRunStatus(db, runId, 'running');

  const tools = buildToolCatalog(run);
  const history: AgentHistoryEntry[] = [];
  let seqCounter = 0;
  const ctx: DispatchContext = {
    db,
    provider,
    run,
    nextSeq: () => ++seqCounter,
  };

  try {
    const userPrompt = buildInitialUserPrompt(db, run);
    let finished = false;
    let finishedSummary: string | null = null;
    let llmCalls = 0;

    while (!finished && llmCalls < run.budget_llm_calls) {
      const { output, usage: _usage } = await provider.agentStep({
        systemPrompt: SYSTEM_PROMPT,
        userPrompt,
        tools,
        history,
      });
      llmCalls++;
      incrementAgentRunLlmCalls(db, runId);

      log(
        `step ${llmCalls}/${run.budget_llm_calls}` +
          (output.thinking ? ` thinking="${output.thinking.slice(0, 200)}"` : '')
      );

      // Echo the assistant's intent into the history for the next round.
      history.push({
        role: 'assistant',
        content: JSON.stringify({
          thinking: output.thinking,
          tool_calls: output.tool_calls,
        }),
      });

      const dispatchResults: Array<{
        index: number;
        name: string;
        result: ReturnType<typeof unwrapResult>;
      }> = [];
      let earlyTerminate = false;
      let pendingFinish: { summary: string | null } | null = null;
      let nonFinishFailed = false;

      for (let i = 0; i < output.tool_calls.length; i++) {
        const call = output.tool_calls[i] as AgentToolCall;
        const result = await dispatchToolCall(ctx, call);
        dispatchResults.push({ index: i, name: call.name, result: unwrapResult(result) });
        log(
          `  → ${call.name}(${JSON.stringify(call.arguments).slice(0, 600)}) ` +
            `ok=${result.ok}` +
            (result.error ? ` error="${result.error}"` : '')
        );
        if (result.finished) {
          // Defer the finish decision until we've seen every call this round.
          // If any non-finish call failed, force a retry instead of letting
          // the model "fire and forget".
          pendingFinish = {
            summary: (call.arguments?.summary as string | undefined) ?? null,
          };
        } else if (!result.ok) {
          nonFinishFailed = true;
        }
        if (result.terminate) {
          earlyTerminate = true;
        }
      }

      if (pendingFinish && !nonFinishFailed) {
        finished = true;
        finishedSummary = pendingFinish.summary;
      } else if (pendingFinish && nonFinishFailed) {
        log(
          '  finish() ignored: a non-finish call in the same batch failed; ' +
            'agent will retry with the error in history.'
        );
      }

      // Tool results back to the model — one entry per step, capturing every
      // dispatched call. Truncated/structured to keep prompts bounded.
      history.push({
        role: 'tool',
        content: JSON.stringify(
          dispatchResults.map((r) => ({
            tool: r.name,
            ok: r.result.ok,
            error: r.result.error ?? undefined,
            data: truncateForPrompt(r.result.data),
          }))
        ),
      });

      if (earlyTerminate) break;
    }

    const actions = listAgentActionsForRun(db, runId);
    const reasoning =
      finishedSummary ??
      (finished ? null : `loop ended without finish; budget llm_calls=${llmCalls}/${run.budget_llm_calls}`);
    setAgentRunStatus(db, runId, 'proposed', { reasoning });

    const finalRun = getAgentRun(db, runId)!;
    return {
      run: finalRun,
      status: finalRun.status,
      proposed_action_count: actions.length,
      llm_calls_used: finalRun.llm_calls_used,
      reasoning: finalRun.reasoning,
      error: finalRun.error,
    };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    setAgentRunStatus(db, runId, 'failed', { error: message });
    const finalRun = getAgentRun(db, runId)!;
    return {
      run: finalRun,
      status: finalRun.status,
      proposed_action_count: listAgentActionsForRun(db, runId).length,
      llm_calls_used: finalRun.llm_calls_used,
      reasoning: finalRun.reasoning,
      error: finalRun.error,
    };
  }
}

function buildInitialUserPrompt(db: DB, run: AgentRunRow): string {
  if (run.scope_type === 'subject') {
    const subject = run.scope_ref;
    const total = countActiveFactsForSubject(db, subject);
    const preview = factsAboutSubject(db, subject, 8)
      .map((f) => `  [${f.id}] (${f.category}, conf=${f.confidence.toFixed(2)}) ${f.content}`)
      .join('\n');
    return [
      `Run scope: subject = "${subject}" (${total} active facts).`,
      run.trigger_fact_id
        ? `Trigger fact id: ${run.trigger_fact_id}. The agent was invoked because this fact recently changed the picture.`
        : 'Trigger: manual review (no specific trigger fact).',
      `Budget: ≤ ${run.budget_ops} proposed actions, ≤ ${run.budget_llm_calls} LLM calls.`,
      '',
      'Top-confidence facts (preview — full list available via list_facts_for_subject):',
      preview || '  (none)',
      '',
      'Begin your review.',
    ].join('\n');
  }

  // entity scope
  const entityId = Number(run.scope_ref);
  const entity = Number.isFinite(entityId) ? getEntityById(db, entityId) : null;
  const mentions = entity ? listFactsMentioningEntity(db, entityId, 50) : [];
  const preview = mentions
    .slice(0, 12)
    .map(
      (m) =>
        `  [${m.id}] subject=${m.subject_wa_id} (${m.category}) ${m.content}` +
        (m.mention_role ? `   — role=${m.mention_role}${m.mention_text ? ` "${m.mention_text}"` : ''}` : '')
    )
    .join('\n');

  // Surface the trigger fact's content directly — without it the agent has
  // to guess what new info just landed.
  let triggerFactBlock = '';
  if (run.trigger_fact_id) {
    const tf = mentions.find((m) => m.id === run.trigger_fact_id);
    if (tf) {
      triggerFactBlock = `Trigger fact: [${tf.id}] (${tf.category}, conf=${tf.confidence.toFixed(2)}) ${tf.content}`;
    } else {
      triggerFactBlock = `Trigger fact id: ${run.trigger_fact_id} (not in current mention list — may have been superseded already).`;
    }
  }

  return [
    entity
      ? `Run scope: entity #${entityId} — "${entity.display_name}" (type=${entity.entity_type}).`
      : `Run scope: entity_id = ${run.scope_ref} (entity row not found).`,
    `Active facts mentioning this entity: ${mentions.length}.`,
    triggerFactBlock ||
      'Trigger: manual review of this entity (no specific trigger fact).',
    run.trigger_fact_id
      ? `The trigger fact resolved or reframed this entity. Earlier facts mentioning ${
          entity?.display_name ?? 'it'
        } may now read differently in light of it — for example, "took Nico for a walk" is more meaningful once we know Nico is a dog.`
      : '',
    `Budget: ≤ ${run.budget_ops} proposed actions, ≤ ${run.budget_llm_calls} LLM calls.`,
    '',
    'Mentions (preview — full list available via list_facts_mentioning_entity):',
    preview || '  (none)',
    '',
    'Your task: decide whether any of these earlier facts should be enriched (update with the new context), corrected, or merged. Each proposal must cite the trigger fact (and any others) as justification. If the existing facts are already adequate, finish without proposing changes.',
  ]
    .filter((line) => line !== '')
    .join('\n');
}

function unwrapResult<T extends { ok: boolean; error?: string; data: Record<string, unknown> }>(
  r: T
): { ok: boolean; error?: string; data: Record<string, unknown> } {
  return { ok: r.ok, error: r.error, data: r.data };
}

const PROMPT_PAYLOAD_CHAR_CAP = 4000;

function truncateForPrompt(data: Record<string, unknown>): Record<string, unknown> {
  const json = JSON.stringify(data);
  if (json.length <= PROMPT_PAYLOAD_CHAR_CAP) return data;
  return {
    _truncated: true,
    _original_size: json.length,
    preview: json.slice(0, PROMPT_PAYLOAD_CHAR_CAP),
  };
}
