/**
 * Curator agent loop, built on ai-sdk's native tool-calling.
 *
 * `planAgentRun` records intent (creates an `agent_runs` row in 'planned'
 * state). `runCurator` drives `generateText({ tools, stopWhen })` for that
 * run: the model picks tools, ai-sdk validates arguments against each tool's
 * Zod schema, the tool's `execute` runs against the DB (read tools return
 * data; propose tools insert into `agent_actions`). The loop terminates
 * when the model responds with text instead of calling another tool, or
 * when `stepCountIs(budget_llm_calls)` fires.
 *
 * `runCurator` NEVER mutates `facts` — that's the apply path's job. Every
 * proposed action carries reason, confidence, citing_fact_ids, and a back-
 * pointer to its run for the audit trail.
 */

import { generateText, stepCountIs } from 'ai';
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
  logProcessing,
  setAgentRunStatus,
  type AgentRunInput,
  type AgentRunRow,
  type DB,
} from '../storage/db';
import { buildCuratorTools, type CuratorContext, type CuratorToolSet } from './tools';
import { CURATOR_MODEL_NAME, getCuratorLanguageModel } from '../../llm';
import type { LLMProvider } from '../../llm/provider';

const SYSTEM_PROMPT = `You are a memory curator. Memory is APPEND-ONLY: facts are immutable. You don't rewrite or delete them. Your job is to ORGANIZE existing memory by adding structure — typed connections between related facts and topical "threads" that group facts about the same subject area.

You audit either one SUBJECT (a person) or one ENTITY (a thing/place/relationship newly resolved by an incoming fact). For each, look for:

1. **Connections the burst pipeline missed.** Two existing facts that should be linked via fact_connections. Predicates:
   - update        — same thing, new state                ("lives in Berlin" → "lives in Lisbon")
   - state_change  — discrete event changing state         ("has a dog Nico" → "Nico passed away")
   - expands       — same fact, more specific             ("has a pet" → "has a dog Nico")
   - qualifies     — adds a condition or nuance           ("works at the bank" → "works part-time at the bank")
   - contradicts   — irreconcilable, no clear winner       ("lives in Rome" ↔ "lives in Milan")
   - retracts      — older fact was wrong                  ("has a cat" → "actually a dog")
   - same_as       — duplicate facts about the same thing  (rare; prefer to fix at extraction)

2. **Thread organization.** Threads are topical buckets per subject ("Nico (her dog)", "career", "music tastes"). Look for facts that belong to existing threads they aren't in yet, or for clusters of related facts that need a brand-new thread.

Your propose-tools (which write to agent_actions, never mutating facts directly):
- propose_connect(from_fact_id, to_fact_id, predicate, ...)        — assert a typed edge between two existing facts
- propose_assign_thread(fact_id, thread_id, ...)                    — attach an existing fact to an existing thread
- propose_create_thread(name, description?, attached_fact_ids?, ...) — create a new thread (and optionally attach facts)

Hard rules:
- Every proposed action MUST cite ≥1 existing fact id. Justifications must be grounded in memory, not invented.
- Prefer no action over a weak action. If you're not confident a connection or thread is correct, skip it and finish.
- Read tools are cheap; bad proposals waste budget. Gather context first when unsure.
- You may NOT create or rewrite facts — those come from the ingestion pipeline.

Workflow guidance:
- Start with list_facts_for_subject (subject scope) or list_facts_mentioning_entity (entity scope) to see what's there.
- Then list_threads_for_subject to see existing threads and decide whether new ones are needed.
- Use get_fact_sources sparingly to disambiguate before proposing — only when the fact text alone is ambiguous.
- Use search_similar_facts when you suspect related facts you haven't seen.

When you're done, stop calling tools and respond with a short text summary (1–3 sentences) of what you proposed and why. If nothing needed changing, say so. Don't keep calling read tools once you've gathered enough context — that wastes budget.`;

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

  let seqCounter = 0;
  const ctx: CuratorContext = {
    db,
    provider,
    run,
    nextSeq: () => ++seqCounter,
  };
  const tools = buildCuratorTools(ctx);

  let stepIndex = 0;
  try {
    const userPrompt = buildInitialUserPrompt(db, run);

    const result = await generateText({
      model: getCuratorLanguageModel(),
      system: SYSTEM_PROMPT,
      prompt: userPrompt,
      tools,
      stopWhen: stepCountIs(run.budget_llm_calls),
      temperature: 0,
      maxRetries: 2,
      onStepFinish: (event) => {
        stepIndex++;
        incrementAgentRunLlmCalls(db, runId);
        logStep(log, run, stepIndex, event);
        logProcessing(db, {
          burst_id: null,
          stage: 'extract',
          model: CURATOR_MODEL_NAME + ' [curator]',
          tokens_in: event.usage.inputTokens ?? 0,
          tokens_out: event.usage.outputTokens ?? 0,
        });
      },
    });

    const reasoning =
      result.text.trim() ||
      `loop ended without a final summary (finishReason=${result.finishReason}, steps=${stepIndex}/${run.budget_llm_calls})`;
    setAgentRunStatus(db, runId, 'proposed', { reasoning });

    const finalRun = getAgentRun(db, runId)!;
    const actions = listAgentActionsForRun(db, runId);
    log(
      `● run ${runId} done: steps=${stepIndex}, proposals=${actions.length}, finishReason=${result.finishReason}`
    );
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

// `event` is StepResult<CuratorToolSet> from ai-sdk. We use `unknown` here on
// purpose: bringing the parameterized generic across the engine boundary made
// tsc OOM during instantiation. The shape we read (text, toolCalls,
// toolResults) is stable across ai-sdk versions and is narrowed locally.
function logStep(
  log: (line: string) => void,
  run: AgentRunRow,
  stepIndex: number,
  event: unknown
): void {
  const e = event as {
    text?: string;
    toolCalls?: Array<{ toolName: string; toolCallId: string; input: unknown }>;
    toolResults?: Array<{ toolCallId: string; output?: unknown }>;
  };
  log(`step ${stepIndex}/${run.budget_llm_calls}` + (e.text ? ` text="${e.text.slice(0, 200)}"` : ''));
  for (const call of e.toolCalls ?? []) {
    const argsPreview = JSON.stringify(call.input).slice(0, 600);
    const matchingResult = (e.toolResults ?? []).find((r) => r.toolCallId === call.toolCallId);
    const output = matchingResult?.output;
    const ok = isOkResult(output);
    const errorPart =
      !ok && output && typeof output === 'object' && 'error' in output
        ? ` error="${(output as { error: unknown }).error}"`
        : '';
    log(`  → ${call.toolName}(${argsPreview}) ok=${ok}${errorPart}`);
  }
}

function isOkResult(output: unknown): boolean {
  return (
    !!output &&
    typeof output === 'object' &&
    'ok' in output &&
    (output as { ok: unknown }).ok === true
  );
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

