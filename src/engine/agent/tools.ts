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

  const propose_update = defineTool({
    description:
      'Propose superseding an active fact with a richer/corrected version. The new content becomes a new fact row; the old one is marked superseded on apply. Requires at least one citing_fact_id (typically the target itself plus any fact that justifies the change).',
    inputSchema: z.object({
      target_fact_id: z.number().int().positive(),
      new_content: z.string().min(1),
      new_category: FactCategoryZ.optional(),
      citing_fact_ids: z.array(z.number().int().positive()).min(1),
      reason: z.string().min(1),
      confidence: z.number().min(0).max(1),
    }),
    execute: async (args) => {
      const budgetErr = checkBudget(ctx);
      if (budgetErr) return errorResult(budgetErr);
      const targetFact = getActiveFact(ctx.db, args.target_fact_id);
      if (!targetFact) {
        return errorResult(`target_fact_id ${args.target_fact_id} is not active`);
      }
      const citingErr = ensureCitingActive(ctx, args.citing_fact_ids);
      if (citingErr) return errorResult(citingErr);
      const id = insertAgentAction(ctx.db, {
        run_id: ctx.run.id,
        seq: ctx.nextSeq(),
        op: 'update' as AgentActionOp,
        target_fact_id: args.target_fact_id,
        new_content: args.new_content,
        new_category: args.new_category ?? null,
        citing_fact_ids: args.citing_fact_ids,
        reason: args.reason,
        confidence: args.confidence,
      });
      return {
        ok: true,
        action_id: id,
        op: 'update',
        target_fact_id: args.target_fact_id,
        new_content: args.new_content,
        new_category: args.new_category ?? null,
      };
    },
  });

  const propose_delete = defineTool({
    description:
      'Propose soft-deleting an active fact that is now wrong or obsolete. Cite the fact(s) that contradict or supersede it.',
    inputSchema: z.object({
      target_fact_id: z.number().int().positive(),
      citing_fact_ids: z.array(z.number().int().positive()).min(1),
      reason: z.string().min(1),
      confidence: z.number().min(0).max(1),
    }),
    execute: async (args) => {
      const budgetErr = checkBudget(ctx);
      if (budgetErr) return errorResult(budgetErr);
      if (!getActiveFact(ctx.db, args.target_fact_id)) {
        return errorResult(`target_fact_id ${args.target_fact_id} is not active`);
      }
      const citingErr = ensureCitingActive(ctx, args.citing_fact_ids);
      if (citingErr) return errorResult(citingErr);
      const id = insertAgentAction(ctx.db, {
        run_id: ctx.run.id,
        seq: ctx.nextSeq(),
        op: 'delete' as AgentActionOp,
        target_fact_id: args.target_fact_id,
        citing_fact_ids: args.citing_fact_ids,
        reason: args.reason,
        confidence: args.confidence,
      });
      return { ok: true, action_id: id, op: 'delete', target_fact_id: args.target_fact_id };
    },
  });

  const propose_merge = defineTool({
    description:
      'Propose collapsing N near-duplicate facts about the same subject into one canonical fact. All listed fact_ids will be superseded by a new fact with the canonical content. citing_fact_ids must include every fact_id being merged.',
    inputSchema: z.object({
      fact_ids: z.array(z.number().int().positive()).min(2),
      canonical_content: z.string().min(1),
      canonical_category: FactCategoryZ,
      citing_fact_ids: z.array(z.number().int().positive()).min(2),
      reason: z.string().min(1),
      confidence: z.number().min(0).max(1),
    }),
    execute: async (args) => {
      const budgetErr = checkBudget(ctx);
      if (budgetErr) return errorResult(budgetErr);
      // All facts must be active and share the same subject.
      let subject: string | null = null;
      for (const fid of args.fact_ids) {
        const f = getActiveFact(ctx.db, fid);
        if (!f) return errorResult(`fact_id ${fid} in merge is not active`);
        if (subject === null) subject = f.subject_wa_id;
        else if (f.subject_wa_id !== subject) {
          return errorResult(
            `merge facts must share the same subject — got ${subject} and ${f.subject_wa_id}`
          );
        }
      }
      const citingSet = new Set(args.citing_fact_ids);
      for (const fid of args.fact_ids) {
        if (!citingSet.has(fid)) {
          return errorResult(
            `citing_fact_ids must include every merged fact_id; missing ${fid}`
          );
        }
      }
      const citingErr = ensureCitingActive(ctx, args.citing_fact_ids);
      if (citingErr) return errorResult(citingErr);

      const id = insertAgentAction(ctx.db, {
        run_id: ctx.run.id,
        seq: ctx.nextSeq(),
        op: 'merge' as AgentActionOp,
        new_content: args.canonical_content,
        new_category: args.canonical_category,
        merge_fact_ids: args.fact_ids,
        citing_fact_ids: args.citing_fact_ids,
        reason: args.reason,
        confidence: args.confidence,
      });
      return {
        ok: true,
        action_id: id,
        op: 'merge',
        fact_ids: args.fact_ids,
        canonical_content: args.canonical_content,
        canonical_category: args.canonical_category,
      };
    },
  });

  return {
    list_facts_for_subject,
    get_fact_sources,
    search_similar_facts,
    list_facts_mentioning_entity,
    propose_update,
    propose_delete,
    propose_merge,
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
