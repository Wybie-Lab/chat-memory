/**
 * Curator tool catalog — built on top of ai-sdk's native tool() helper.
 *
 * Each tool defines its argument shape via Zod (the model is shown the JSON
 * schema; the SDK validates incoming calls against it before invoking
 * `execute`). Read tools query the DB and return JSON-friendly objects.
 * Propose tools insert rows into `agent_actions` with `status='proposed'` —
 * they NEVER mutate facts. The apply path is a separate, gated step.
 *
 * Errors are returned as structured objects (`{ ok: false, error: '...' }`)
 * rather than thrown, so the model sees them in the next-step tool result
 * and can correct its arguments. Throws are reserved for programmer errors
 * that shouldn't reach the model.
 */

import type { Tool, ToolSet } from 'ai';
import { z } from 'zod';
import {
  CONNECTION_PREDICATES,
  countActiveFactsForSubject,
  countAgentActionsForRun,
  factsAboutSubject,
  getActiveFact,
  getFactSourceBursts,
  getMemoryThread,
  insertAgentAction,
  listFactsInThread,
  listFactsMentioningEntity,
  listMemoryThreads,
  searchFactsForSubjectByVector,
  type ActiveFactRow,
  type AgentActionOp,
  type AgentRunRow,
  type DB,
} from '../storage/db';
import type { LLMProvider } from '../../llm/provider';

// Plain `Tool` objects — we deliberately skip the `tool({...})` helper from
// ai-sdk because its overloaded INPUT/OUTPUT inference made tsc OOM when
// called 7× in a single function body. The runtime shape is identical;
// inputSchema is still validated by the SDK against incoming tool calls.
function defineTool<I>(t: {
  description: string;
  inputSchema: z.ZodType<I>;
  execute: (input: I) => Promise<unknown>;
}): Tool {
  return t as unknown as Tool;
}

const FACT_CATEGORIES = ['preference', 'event', 'commitment', 'fact', 'relationship'] as const;
const FactCategoryZ = z.enum(FACT_CATEGORIES);

export interface CuratorContext {
  db: DB;
  provider: LLMProvider;
  run: AgentRunRow;
  /** Increments per propose_* call so agent_actions.seq reflects insertion order. */
  nextSeq: () => number;
}

export type CuratorToolSet = ToolSet;

export function buildCuratorTools(ctx: CuratorContext): CuratorToolSet {
  // Each tool is assigned to a `Tool`-typed const before being collected into
  // the returned object. The annotation widens each tool to the default
  // `Tool<any, any>`, isolating per-tool generic inference. Without it, tsc
  // accumulates instantiations across all 7 tools in a single object literal
  // and OOMs (~8GB+) — see issue notes in the curator README.
  const list_facts_for_subject = defineTool({
    description:
      'List active facts about a subject. If subject_wa_id is omitted, uses the run scope.',
    inputSchema: z.object({
      subject_wa_id: z.string().optional(),
      limit: z.number().int().min(1).max(200).optional(),
    }),
    execute: async ({ subject_wa_id, limit }) => {
      const subject =
        subject_wa_id ?? (ctx.run.scope_type === 'subject' ? ctx.run.scope_ref : null);
      if (!subject) {
        return errorResult(
          'subject_wa_id required (run scope is not a subject — pass it explicitly)'
        );
      }
      const cap = limit ?? 100;
      const facts = factsAboutSubject(ctx.db, subject, cap);
      return {
        ok: true,
        subject_wa_id: subject,
        total_active: countActiveFactsForSubject(ctx.db, subject),
        facts: facts.map(serializeFact),
      };
    },
  });

  const get_fact_sources = defineTool({
    description:
      'Return the original burst messages backing a fact. Use this when the fact text alone is ambiguous and you need the literal evidence before proposing a change.',
    inputSchema: z.object({
      fact_id: z.number().int().positive(),
      burst_limit: z.number().int().min(1).max(5).optional(),
    }),
    execute: async ({ fact_id, burst_limit }) => {
      const fact = getActiveFact(ctx.db, fact_id);
      if (!fact) {
        return errorResult(
          `fact ${fact_id} is not active (superseded, deleted, or unknown)`
        );
      }
      const bursts = getFactSourceBursts(ctx.db, fact_id, burst_limit ?? 3);
      return {
        ok: true,
        fact: serializeFact(fact),
        bursts: bursts.map((b) => ({
          burst_id: b.burst_id,
          messages: b.messages.map((m) => ({
            ts: m.ts,
            direction: m.direction,
            body: m.body,
          })),
        })),
      };
    },
  });

  const search_similar_facts = defineTool({
    description:
      'Vector search across active facts about the scope subject for facts semantically similar to the query text. Useful for finding duplicates or near-duplicates. Subject scope only.',
    inputSchema: z.object({
      query: z.string().min(1),
      k: z.number().int().min(1).max(20).optional(),
    }),
    execute: async ({ query, k }) => {
      if (ctx.run.scope_type !== 'subject') {
        return errorResult(
          'search_similar_facts is only available for subject-scoped runs'
        );
      }
      const trimmed = query.trim();
      if (!trimmed) return errorResult('query is required');
      const { vector } = await ctx.provider.embed(trimmed, 'query');
      const hits = searchFactsForSubjectByVector(
        ctx.db,
        vector,
        ctx.run.scope_ref,
        k ?? 10
      );
      return { ok: true, query: trimmed, hits: hits.map(serializeFact) };
    },
  });

  const list_facts_mentioning_entity = defineTool({
    description:
      'List active facts whose entity-mention rows include this entity_id. Useful for cross-subject reasoning when the scope is an entity.',
    inputSchema: z.object({
      entity_id: z.number().int().positive(),
      limit: z.number().int().min(1).max(200).optional(),
    }),
    execute: async ({ entity_id, limit }) => {
      const rows = listFactsMentioningEntity(ctx.db, entity_id, limit ?? 50);
      return {
        ok: true,
        entity_id,
        facts: rows.map((r) => ({
          ...serializeFact(r),
          mention_role: r.mention_role,
          mention_text: r.mention_text,
        })),
      };
    },
  });

  const list_threads_for_subject = defineTool({
    description:
      'List existing memory threads for a subject. Threads are topical buckets (e.g. "Rex (her dog)", "career", "music tastes"). If subject_wa_id is omitted, uses the run scope.',
    inputSchema: z.object({
      subject_wa_id: z.string().optional(),
    }),
    execute: async ({ subject_wa_id }) => {
      const subject =
        subject_wa_id ?? (ctx.run.scope_type === 'subject' ? ctx.run.scope_ref : null);
      if (!subject) {
        return errorResult(
          'subject_wa_id required (run scope is not a subject — pass it explicitly)'
        );
      }
      const threads = listMemoryThreads(ctx.db, { owner_subject_wa_id: subject });
      return {
        ok: true,
        subject_wa_id: subject,
        threads: threads.map((t) => ({
          id: t.id,
          name: t.name,
          description: t.description,
          created_at: t.created_at,
        })),
      };
    },
  });

  const list_facts_in_thread_tool = defineTool({
    description:
      'List active facts attached to a thread. Use this to inspect a thread before deciding whether to add or move facts.',
    inputSchema: z.object({
      thread_id: z.number().int().positive(),
      limit: z.number().int().min(1).max(200).optional(),
    }),
    execute: async ({ thread_id, limit }) => {
      const thread = getMemoryThread(ctx.db, thread_id);
      if (!thread) return errorResult(`thread ${thread_id} not found`);
      const facts = listFactsInThread(ctx.db, thread_id, limit ?? 50);
      return {
        ok: true,
        thread: {
          id: thread.id,
          name: thread.name,
          description: thread.description,
        },
        facts: facts.map(serializeFact),
      };
    },
  });

  const propose_connect = defineTool({
    description:
      'Propose a typed connection between two existing facts. The new edge will be inserted into fact_connections on apply. Use this when two existing facts relate to each other in a way the burst pipeline missed (e.g. one updates the other, expands it, contradicts it). Both facts must be active.',
    inputSchema: z.object({
      from_fact_id: z.number().int().positive(),
      to_fact_id: z.number().int().positive(),
      predicate: z.enum(CONNECTION_PREDICATES),
      citing_fact_ids: z.array(z.number().int().positive()).min(1),
      reason: z.string().min(1),
      confidence: z.number().min(0).max(1),
    }),
    execute: async (args) => {
      const budgetErr = checkBudget(ctx);
      if (budgetErr) return errorResult(budgetErr);
      if (args.from_fact_id === args.to_fact_id) {
        return errorResult('from_fact_id and to_fact_id must differ');
      }
      if (!getActiveFact(ctx.db, args.from_fact_id)) {
        return errorResult(`from_fact_id ${args.from_fact_id} is not active`);
      }
      if (!getActiveFact(ctx.db, args.to_fact_id)) {
        return errorResult(`to_fact_id ${args.to_fact_id} is not active`);
      }
      const citingErr = ensureCitingActive(ctx, args.citing_fact_ids);
      if (citingErr) return errorResult(citingErr);
      const id = insertAgentAction(ctx.db, {
        run_id: ctx.run.id,
        seq: ctx.nextSeq(),
        op: 'connect' as AgentActionOp,
        target_fact_id: args.from_fact_id,
        extra: {
          secondary_fact_id: args.to_fact_id,
          predicate: args.predicate,
        },
        citing_fact_ids: args.citing_fact_ids,
        reason: args.reason,
        confidence: args.confidence,
      });
      return {
        ok: true,
        action_id: id,
        op: 'connect',
        from_fact_id: args.from_fact_id,
        to_fact_id: args.to_fact_id,
        predicate: args.predicate,
      };
    },
  });

  const propose_assign_thread = defineTool({
    description:
      'Propose attaching an existing fact to an existing thread. Use this when a fact belongs to a thread it is currently not in (the burst pipeline\'s thread assignment missed it, or the thread was created later). The fact-thread membership is many-to-many; you can attach the same fact to multiple threads via multiple calls.',
    inputSchema: z.object({
      fact_id: z.number().int().positive(),
      thread_id: z.number().int().positive(),
      citing_fact_ids: z.array(z.number().int().positive()).min(1),
      reason: z.string().min(1),
      confidence: z.number().min(0).max(1),
    }),
    execute: async (args) => {
      const budgetErr = checkBudget(ctx);
      if (budgetErr) return errorResult(budgetErr);
      if (!getActiveFact(ctx.db, args.fact_id)) {
        return errorResult(`fact_id ${args.fact_id} is not active`);
      }
      const thread = getMemoryThread(ctx.db, args.thread_id);
      if (!thread || thread.deleted_at !== null) {
        return errorResult(`thread_id ${args.thread_id} not found or deleted`);
      }
      const citingErr = ensureCitingActive(ctx, args.citing_fact_ids);
      if (citingErr) return errorResult(citingErr);
      const id = insertAgentAction(ctx.db, {
        run_id: ctx.run.id,
        seq: ctx.nextSeq(),
        op: 'assign_thread' as AgentActionOp,
        target_fact_id: args.fact_id,
        extra: { thread_id: args.thread_id },
        citing_fact_ids: args.citing_fact_ids,
        reason: args.reason,
        confidence: args.confidence,
      });
      return {
        ok: true,
        action_id: id,
        op: 'assign_thread',
        fact_id: args.fact_id,
        thread_id: args.thread_id,
      };
    },
  });

  const propose_create_thread = defineTool({
    description:
      'Propose creating a new memory thread, optionally attaching one or more existing facts to it on apply. Use this when several facts belong to a topical bucket that doesn\'t exist yet (e.g. a new pet, project, trip). The owner_subject_wa_id defaults to the run scope when scope_type=subject; pass null explicitly for cross-subject threads.',
    inputSchema: z.object({
      name: z.string().min(1).max(80),
      description: z.string().max(500).optional(),
      owner_subject_wa_id: z.string().nullable().optional(),
      attached_fact_ids: z.array(z.number().int().positive()).default([]),
      citing_fact_ids: z.array(z.number().int().positive()).min(1),
      reason: z.string().min(1),
      confidence: z.number().min(0).max(1),
    }),
    execute: async (args) => {
      const budgetErr = checkBudget(ctx);
      if (budgetErr) return errorResult(budgetErr);
      // Default owner from run scope if not given.
      const owner: string | null =
        args.owner_subject_wa_id !== undefined
          ? args.owner_subject_wa_id
          : ctx.run.scope_type === 'subject'
            ? ctx.run.scope_ref
            : null;
      const attached = args.attached_fact_ids ?? [];
      // All attached facts must be active.
      for (const fid of attached) {
        if (!getActiveFact(ctx.db, fid)) {
          return errorResult(`attached_fact_id ${fid} is not active`);
        }
      }
      const citingErr = ensureCitingActive(ctx, args.citing_fact_ids);
      if (citingErr) return errorResult(citingErr);
      const id = insertAgentAction(ctx.db, {
        run_id: ctx.run.id,
        seq: ctx.nextSeq(),
        op: 'create_thread' as AgentActionOp,
        target_fact_id: attached.length > 0 ? attached[0] : null,
        extra: {
          name: args.name,
          description: args.description ?? null,
          owner_subject_wa_id: owner,
          attached_fact_ids: attached,
        },
        citing_fact_ids: args.citing_fact_ids,
        reason: args.reason,
        confidence: args.confidence,
      });
      return {
        ok: true,
        action_id: id,
        op: 'create_thread',
        name: args.name,
        owner_subject_wa_id: owner,
        attached_fact_ids: attached,
      };
    },
  });

  return {
    list_facts_for_subject,
    get_fact_sources,
    search_similar_facts,
    list_facts_mentioning_entity,
    list_threads_for_subject,
    list_facts_in_thread: list_facts_in_thread_tool,
    propose_connect,
    propose_assign_thread,
    propose_create_thread,
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

function errorResult(error: string): { ok: false; error: string } {
  return { ok: false, error };
}

function checkBudget(ctx: CuratorContext): string | null {
  const used = countAgentActionsForRun(ctx.db, ctx.run.id);
  if (used >= ctx.run.budget_ops) {
    return `op budget exhausted (${used}/${ctx.run.budget_ops} proposed actions). Call no further propose_* tools; finalize with a summary.`;
  }
  return null;
}

function ensureCitingActive(ctx: CuratorContext, ids: number[]): string | null {
  for (const id of ids) {
    if (!getActiveFact(ctx.db, id)) {
      return `citing_fact_id ${id} is not active (superseded, deleted, or unknown)`;
    }
  }
  return null;
}
