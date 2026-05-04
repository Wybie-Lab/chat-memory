/**
 * Curator tool catalog + dispatcher.
 *
 * The agent loop in `curator.ts` calls `dispatchToolCall` for each tool the
 * LLM requests. Read tools execute against the DB and return JSON-friendly
 * results; propose tools insert rows into `agent_actions` (status='proposed')
 * — they NEVER mutate facts. The apply path is a separate, gated step
 * (not in this v1).
 *
 * Every propose tool requires `citing_fact_ids` (≥1) and validates that the
 * cited ids reference real, active facts. This makes hallucinated proposals
 * fail fast at the dispatcher rather than landing in agent_actions.
 */

import {
  countActiveFactsForSubject,
  countAgentActionsForRun,
  factsAboutSubject,
  getActiveFact,
  getFactSourceBursts,
  insertAgentAction,
  listFactsMentioningEntity,
  searchFactsForSubjectByVector,
  type ActiveFactRow,
  type AgentActionOp,
  type AgentRunRow,
  type DB,
} from '../storage/db';
import type { AgentToolDefinition, LLMProvider } from '../../llm/provider';

export interface DispatchResult {
  ok: boolean;
  /** JSON-serializable payload returned to the LLM as tool output. */
  data: Record<string, unknown>;
  /** Error message when ok=false. Surfaced to the LLM so it can adjust. */
  error?: string;
  /** True iff this was a successful 'finish' call ending the run. */
  finished?: boolean;
  /** When set, the loop must terminate (budget exhausted, internal error). */
  terminate?: boolean;
}

export interface DispatchContext {
  db: DB;
  provider: LLMProvider;
  run: AgentRunRow;
  /** seq counter for proposed actions within this run. */
  nextSeq: () => number;
}

const FACT_CATEGORIES = ['preference', 'event', 'commitment', 'fact', 'relationship'] as const;

export function buildToolCatalog(scope: AgentRunRow): AgentToolDefinition[] {
  const tools: AgentToolDefinition[] = [
    {
      name: 'list_facts_for_subject',
      description:
        'List active facts about a subject. If subject_wa_id is omitted, uses the run scope.',
      parameters: {
        type: 'object',
        properties: {
          subject_wa_id: { type: 'string' },
          limit: { type: 'integer', minimum: 1, maximum: 200 },
        },
        additionalProperties: false,
      },
    },
    {
      name: 'get_fact_sources',
      description:
        'Return the original burst messages backing a fact. Use this when you need the literal evidence before proposing a change.',
      parameters: {
        type: 'object',
        properties: {
          fact_id: { type: 'integer', minimum: 1 },
          burst_limit: { type: 'integer', minimum: 1, maximum: 5 },
        },
        required: ['fact_id'],
        additionalProperties: false,
      },
    },
    {
      name: 'search_similar_facts',
      description:
        'Vector search across active facts about the scope subject for facts semantically similar to the query text. Useful for finding duplicates or near-duplicates.',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          k: { type: 'integer', minimum: 1, maximum: 20 },
        },
        required: ['query'],
        additionalProperties: false,
      },
    },
    {
      name: 'list_facts_mentioning_entity',
      description:
        'List active facts whose entity-mention rows include this entity_id. Useful for cross-subject reasoning when the scope is an entity.',
      parameters: {
        type: 'object',
        properties: {
          entity_id: { type: 'integer', minimum: 1 },
          limit: { type: 'integer', minimum: 1, maximum: 200 },
        },
        required: ['entity_id'],
        additionalProperties: false,
      },
    },
    {
      name: 'propose_update',
      description:
        'Propose superseding an active fact with a richer/corrected version. The new content becomes a new fact row; the old one is marked superseded on apply. Requires at least one citing_fact_id (typically the target itself plus any fact that justifies the change).',
      parameters: {
        type: 'object',
        properties: {
          target_fact_id: { type: 'integer', minimum: 1 },
          new_content: { type: 'string', minLength: 1 },
          new_category: { type: 'string', enum: [...FACT_CATEGORIES] },
          citing_fact_ids: {
            type: 'array',
            items: { type: 'integer', minimum: 1 },
            minItems: 1,
          },
          reason: { type: 'string', minLength: 1 },
          confidence: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: ['target_fact_id', 'new_content', 'citing_fact_ids', 'reason', 'confidence'],
        additionalProperties: false,
      },
    },
    {
      name: 'propose_delete',
      description:
        'Propose soft-deleting an active fact that is now wrong or obsolete. Cite the fact(s) that contradict or supersede it.',
      parameters: {
        type: 'object',
        properties: {
          target_fact_id: { type: 'integer', minimum: 1 },
          citing_fact_ids: {
            type: 'array',
            items: { type: 'integer', minimum: 1 },
            minItems: 1,
          },
          reason: { type: 'string', minLength: 1 },
          confidence: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: ['target_fact_id', 'citing_fact_ids', 'reason', 'confidence'],
        additionalProperties: false,
      },
    },
    {
      name: 'propose_merge',
      description:
        'Propose collapsing N near-duplicate facts about the same subject into one canonical fact. All listed fact_ids will be superseded by a new fact with the canonical content. citing_fact_ids must include every fact_id being merged (the merge is its own evidence).',
      parameters: {
        type: 'object',
        properties: {
          fact_ids: {
            type: 'array',
            items: { type: 'integer', minimum: 1 },
            minItems: 2,
          },
          canonical_content: { type: 'string', minLength: 1 },
          canonical_category: { type: 'string', enum: [...FACT_CATEGORIES] },
          citing_fact_ids: {
            type: 'array',
            items: { type: 'integer', minimum: 1 },
            minItems: 2,
          },
          reason: { type: 'string', minLength: 1 },
          confidence: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: [
          'fact_ids',
          'canonical_content',
          'canonical_category',
          'citing_fact_ids',
          'reason',
          'confidence',
        ],
        additionalProperties: false,
      },
    },
    {
      name: 'finish',
      description:
        'End the run. Provide a one-paragraph summary of what you proposed and why. No further tool calls accepted after this.',
      parameters: {
        type: 'object',
        properties: {
          summary: { type: 'string', minLength: 1 },
        },
        required: ['summary'],
        additionalProperties: false,
      },
    },
  ];
  return tools;
}

export async function dispatchToolCall(
  ctx: DispatchContext,
  call: { name: string; arguments: Record<string, unknown> }
): Promise<DispatchResult> {
  switch (call.name) {
    case 'list_facts_for_subject':
      return readListFactsForSubject(ctx, call.arguments);
    case 'get_fact_sources':
      return readGetFactSources(ctx, call.arguments);
    case 'search_similar_facts':
      return readSearchSimilarFacts(ctx, call.arguments);
    case 'list_facts_mentioning_entity':
      return readListFactsMentioningEntity(ctx, call.arguments);
    case 'propose_update':
      return proposeUpdate(ctx, call.arguments);
    case 'propose_delete':
      return proposeDelete(ctx, call.arguments);
    case 'propose_merge':
      return proposeMerge(ctx, call.arguments);
    case 'finish':
      return finishRun(call.arguments);
    default:
      return {
        ok: false,
        error: `unknown tool: ${call.name}`,
        data: {},
      };
  }
}

// ───────────── Read tools ─────────────

function readListFactsForSubject(
  ctx: DispatchContext,
  args: Record<string, unknown>
): DispatchResult {
  const subjectArg = typeof args.subject_wa_id === 'string' ? args.subject_wa_id : null;
  const subject = subjectArg ?? (ctx.run.scope_type === 'subject' ? ctx.run.scope_ref : null);
  if (!subject) {
    return {
      ok: false,
      data: {},
      error:
        'subject_wa_id required (run scope is not a subject — pass it explicitly)',
    };
  }
  const limit = clampInt(args.limit, 1, 200, 100);
  const facts = factsAboutSubject(ctx.db, subject, limit);
  return {
    ok: true,
    data: {
      subject_wa_id: subject,
      total_active: countActiveFactsForSubject(ctx.db, subject),
      facts: facts.map(serializeFact),
    },
  };
}

function readGetFactSources(
  ctx: DispatchContext,
  args: Record<string, unknown>
): DispatchResult {
  const factId = toPositiveInt(args.fact_id);
  if (factId === null) return badArg('fact_id must be a positive integer');
  const burstLimit = clampInt(args.burst_limit, 1, 5, 3);
  const fact = getActiveFact(ctx.db, factId);
  if (!fact) {
    return {
      ok: false,
      data: { fact_id: factId },
      error: `fact ${factId} is not active (superseded, deleted, or unknown)`,
    };
  }
  const bursts = getFactSourceBursts(ctx.db, factId, burstLimit);
  return {
    ok: true,
    data: {
      fact: serializeFact(fact),
      bursts: bursts.map((b) => ({
        burst_id: b.burst_id,
        messages: b.messages.map((m) => ({
          ts: m.ts,
          direction: m.direction,
          body: m.body,
        })),
      })),
    },
  };
}

async function readSearchSimilarFacts(
  ctx: DispatchContext,
  args: Record<string, unknown>
): Promise<DispatchResult> {
  const query = typeof args.query === 'string' ? args.query.trim() : '';
  if (!query) return badArg('query is required');
  if (ctx.run.scope_type !== 'subject') {
    return badArg('search_similar_facts is only available for subject-scoped runs');
  }
  const k = clampInt(args.k, 1, 20, 10);
  const { vector } = await ctx.provider.embed(query, 'query');
  const hits = searchFactsForSubjectByVector(ctx.db, vector, ctx.run.scope_ref, k);
  return {
    ok: true,
    data: {
      query,
      hits: hits.map(serializeFact),
    },
  };
}

function readListFactsMentioningEntity(
  ctx: DispatchContext,
  args: Record<string, unknown>
): DispatchResult {
  const entityId = toPositiveInt(args.entity_id);
  if (entityId === null) return badArg('entity_id must be a positive integer');
  const limit = clampInt(args.limit, 1, 200, 50);
  const rows = listFactsMentioningEntity(ctx.db, entityId, limit);
  return {
    ok: true,
    data: {
      entity_id: entityId,
      facts: rows.map((r) => ({
        ...serializeFact(r),
        mention_role: r.mention_role,
        mention_text: r.mention_text,
      })),
    },
  };
}

// ───────────── Propose tools ─────────────

interface ProposeBase {
  citing_fact_ids: number[];
  reason: string;
  confidence: number;
}

function validateProposeBase(args: Record<string, unknown>): ProposeBase | string {
  const citing = toPositiveIntArray(args.citing_fact_ids);
  if (!citing || citing.length === 0) {
    return 'citing_fact_ids must be a non-empty array of positive integers';
  }
  const reason = typeof args.reason === 'string' ? args.reason.trim() : '';
  if (!reason) return 'reason is required';
  const confidence = toNumber(args.confidence);
  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    return 'confidence must be a number between 0 and 1';
  }
  return { citing_fact_ids: citing, reason, confidence };
}

function checkBudget(ctx: DispatchContext): string | null {
  const used = countAgentActionsForRun(ctx.db, ctx.run.id);
  if (used >= ctx.run.budget_ops) {
    return `op budget exhausted (${used}/${ctx.run.budget_ops} proposed actions)`;
  }
  return null;
}

function ensureCitingActive(ctx: DispatchContext, ids: number[]): string | null {
  for (const id of ids) {
    if (!getActiveFact(ctx.db, id)) {
      return `citing_fact_id ${id} is not active (superseded, deleted, or unknown)`;
    }
  }
  return null;
}

function proposeUpdate(
  ctx: DispatchContext,
  args: Record<string, unknown>
): DispatchResult {
  const budgetErr = checkBudget(ctx);
  if (budgetErr) return { ok: false, data: {}, error: budgetErr, terminate: true };

  const base = validateProposeBase(args);
  if (typeof base === 'string') return badArg(base);

  const target = toPositiveInt(args.target_fact_id);
  if (target === null) return badArg('target_fact_id must be a positive integer');
  const targetFact = getActiveFact(ctx.db, target);
  if (!targetFact) {
    return {
      ok: false,
      data: { target_fact_id: target },
      error: `target_fact_id ${target} is not active`,
    };
  }
  const newContent = typeof args.new_content === 'string' ? args.new_content.trim() : '';
  if (!newContent) return badArg('new_content is required');
  const newCategoryRaw = args.new_category;
  const newCategory =
    typeof newCategoryRaw === 'string' && (FACT_CATEGORIES as readonly string[]).includes(newCategoryRaw)
      ? newCategoryRaw
      : null;

  const citingErr = ensureCitingActive(ctx, base.citing_fact_ids);
  if (citingErr) return badArg(citingErr);

  const id = insertAgentAction(ctx.db, {
    run_id: ctx.run.id,
    seq: ctx.nextSeq(),
    op: 'update' as AgentActionOp,
    target_fact_id: target,
    new_content: newContent,
    new_category: newCategory,
    citing_fact_ids: base.citing_fact_ids,
    reason: base.reason,
    confidence: base.confidence,
  });
  return {
    ok: true,
    data: {
      action_id: id,
      op: 'update',
      target_fact_id: target,
      new_content: newContent,
      new_category: newCategory,
    },
  };
}

function proposeDelete(
  ctx: DispatchContext,
  args: Record<string, unknown>
): DispatchResult {
  const budgetErr = checkBudget(ctx);
  if (budgetErr) return { ok: false, data: {}, error: budgetErr, terminate: true };

  const base = validateProposeBase(args);
  if (typeof base === 'string') return badArg(base);

  const target = toPositiveInt(args.target_fact_id);
  if (target === null) return badArg('target_fact_id must be a positive integer');
  if (!getActiveFact(ctx.db, target)) {
    return {
      ok: false,
      data: { target_fact_id: target },
      error: `target_fact_id ${target} is not active`,
    };
  }
  const citingErr = ensureCitingActive(ctx, base.citing_fact_ids);
  if (citingErr) return badArg(citingErr);

  const id = insertAgentAction(ctx.db, {
    run_id: ctx.run.id,
    seq: ctx.nextSeq(),
    op: 'delete' as AgentActionOp,
    target_fact_id: target,
    citing_fact_ids: base.citing_fact_ids,
    reason: base.reason,
    confidence: base.confidence,
  });
  return {
    ok: true,
    data: { action_id: id, op: 'delete', target_fact_id: target },
  };
}

function proposeMerge(
  ctx: DispatchContext,
  args: Record<string, unknown>
): DispatchResult {
  const budgetErr = checkBudget(ctx);
  if (budgetErr) return { ok: false, data: {}, error: budgetErr, terminate: true };

  const base = validateProposeBase(args);
  if (typeof base === 'string') return badArg(base);

  const factIds = toPositiveIntArray(args.fact_ids);
  if (!factIds || factIds.length < 2) {
    return badArg('fact_ids must be an array of ≥2 positive integers');
  }
  // All facts must be active and share the same subject.
  let subject: string | null = null;
  for (const fid of factIds) {
    const f = getActiveFact(ctx.db, fid);
    if (!f) return badArg(`fact_id ${fid} in merge is not active`);
    if (subject === null) subject = f.subject_wa_id;
    else if (f.subject_wa_id !== subject) {
      return badArg(
        `merge facts must share the same subject — got ${subject} and ${f.subject_wa_id}`
      );
    }
  }

  const canonicalContent =
    typeof args.canonical_content === 'string' ? args.canonical_content.trim() : '';
  if (!canonicalContent) return badArg('canonical_content is required');
  const canonicalCategoryRaw = args.canonical_category;
  if (
    typeof canonicalCategoryRaw !== 'string' ||
    !(FACT_CATEGORIES as readonly string[]).includes(canonicalCategoryRaw)
  ) {
    return badArg(`canonical_category must be one of: ${FACT_CATEGORIES.join(', ')}`);
  }

  // citing_fact_ids must include every fact being merged so the audit trail is honest.
  const citingSet = new Set(base.citing_fact_ids);
  for (const fid of factIds) {
    if (!citingSet.has(fid)) {
      return badArg(`citing_fact_ids must include every merged fact_id; missing ${fid}`);
    }
  }
  const citingErr = ensureCitingActive(ctx, base.citing_fact_ids);
  if (citingErr) return badArg(citingErr);

  const id = insertAgentAction(ctx.db, {
    run_id: ctx.run.id,
    seq: ctx.nextSeq(),
    op: 'merge' as AgentActionOp,
    new_content: canonicalContent,
    new_category: canonicalCategoryRaw,
    merge_fact_ids: factIds,
    citing_fact_ids: base.citing_fact_ids,
    reason: base.reason,
    confidence: base.confidence,
  });
  return {
    ok: true,
    data: {
      action_id: id,
      op: 'merge',
      fact_ids: factIds,
      canonical_content: canonicalContent,
      canonical_category: canonicalCategoryRaw,
    },
  };
}

function finishRun(args: Record<string, unknown>): DispatchResult {
  const summary = typeof args.summary === 'string' ? args.summary.trim() : '';
  return {
    ok: true,
    finished: true,
    data: { finished: true, summary },
  };
}

// ───────────── helpers ─────────────

function serializeFact(f: ActiveFactRow): Record<string, unknown> {
  return {
    id: f.id,
    subject_wa_id: f.subject_wa_id,
    category: f.category,
    content: f.content,
    confidence: f.confidence,
    extracted_at: f.extracted_at,
    event_ts: f.event_ts,
  };
}

function badArg(error: string): DispatchResult {
  return { ok: false, data: {}, error };
}

// Coerce a value to a number, accepting either real numbers or numeric strings
// (e.g. Gemini Flash structured-output sometimes stringifies all argument
// values). Returns NaN on anything else.
function toNumber(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const n = Number(v);
    return Number.isFinite(n) ? n : NaN;
  }
  return NaN;
}

function clampInt(
  v: unknown,
  min: number,
  max: number,
  fallback: number
): number {
  const num = toNumber(v);
  if (!Number.isFinite(num)) return fallback;
  const n = Math.floor(num);
  if (n < min) return min;
  if (n > max) return max;
  return n;
}

function toPositiveInt(v: unknown): number | null {
  const num = toNumber(v);
  if (!Number.isFinite(num)) return null;
  const n = Math.floor(num);
  return n > 0 ? n : null;
}

function toPositiveIntArray(v: unknown): number[] | null {
  // Models (especially Gemini Flash via structured-output) serialize array
  // arguments inconsistently. We accept any of:
  //   [8, 7]            — real array (ideal)
  //   "[8, 7]"          — JSON-stringified array
  //   "8,7"  /  "8, 7"  — comma-separated string
  //   8                 — single number, wrapped to [8]
  //   "8"               — single numeric string, wrapped to [8]
  let arr: unknown = v;
  if (typeof arr === 'string') {
    const trimmed = arr.trim();
    if (trimmed === '') return null;
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      try {
        arr = JSON.parse(trimmed);
      } catch {
        return null;
      }
    } else if (trimmed.includes(',')) {
      arr = trimmed.split(',').map((s) => s.trim()).filter((s) => s.length > 0);
    } else {
      // Single numeric string — treat as a one-element array.
      const n = toPositiveInt(trimmed);
      if (n === null) return null;
      return [n];
    }
  } else if (typeof arr === 'number') {
    const n = toPositiveInt(arr);
    if (n === null) return null;
    return [n];
  }
  if (!Array.isArray(arr)) return null;
  const out: number[] = [];
  for (const item of arr) {
    const n = toPositiveInt(item);
    if (n === null) return null;
    out.push(n);
  }
  return out;
}
