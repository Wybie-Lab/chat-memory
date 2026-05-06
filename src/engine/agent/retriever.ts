/**
 * Retrieval agent — answers a question by driving an ai-sdk `generateText`
 * loop with a read-only tool catalog over the memory graph. Replaces the
 * static one-shot `composeMemoryBlock` with a navigation surface: the
 * agent picks tools, results re-enter the prompt, the next step plans
 * against the new evidence.
 *
 * Tool catalog is shaped around how the structure was built:
 *   browse:    list_categories_for_subject, list_facts_by_category
 *   search:    search_facts (vector + filters), find_entity
 *   graph:     traverse_entity_neighborhood, list_entities_in_facts
 *   temporal:  list_events_in_window, as_of, get_fact_chain_with_history
 *   inspect:   get_fact_source
 *
 * The agent emits the final answer text directly (no separate chat call).
 * Tool results accumulate in seenFactIds so callers can show citations.
 */

import { generateText, stepCountIs, type Tool, type ToolSet } from 'ai';
import { z } from 'zod';
import {
  CONNECTION_PREDICATES,
  allActiveSubjects,
  entitiesForFacts,
  eventsInWindow,
  factCountsByCategoryForSubject,
  factsActiveAtTime,
  factsByCategoryForSubject,
  getActiveFactById,
  getFactSourceBursts,
  graphNeighborhood,
  latestInChain,
  listConnectionsToFact,
  listConnectionsFromFact,
  listFactsMentioningEntity,
  searchEntities,
  searchFactsHybrid,
  type ActiveFactRow,
  type DB,
} from '../storage/db';
import { CURATOR_MODEL_NAME, getCuratorLanguageModel } from '../../llm';
import type { LLMProvider } from '../../llm/provider';

const DEFAULT_MAX_STEPS = 10;

const FACT_CATEGORIES = ['preference', 'event', 'commitment', 'fact', 'relationship'] as const;
const FactCategoryZ = z.enum(FACT_CATEGORIES);

const FACTOID_SYSTEM_PROMPT = `You answer questions about people using a graph of facts. You navigate the memory by drilling down through structure.

What's in memory:
- Subjects: the people memory tracks (listed below).
- Facts: atomic, immutable statements about ONE subject. Categorized: preference | event | commitment | fact | relationship.
- Connections: typed edges between facts (update / state_change / expands / qualifies / contradicts / retracts / same_as). Walk update chains for the latest state.
- Threads: topical buckets per subject ("career", "Rex (her dog)").
- Entities (graph): people/places/orgs/things mentioned in facts. Edges between entities use predicates like family_of / works_at / lives_in / owns / from_place.
- Time: every fact has extracted_at (when learned in conversation) and optionally event_ts (when the event happened).

Recommended strategy (don't follow blindly — adapt to the question):
1. SEARCH FIRST — search_facts(question) is your default opening move. Vector search over all facts; works even when the question is short ("What was grandma's gift?", "How long has Mel been married?", "Where did Caroline move from?"). Filters are optional. Don't skip this for direct fact lookups.
2. SCOPE — list_categories_for_subject(subject) when search returned weak hits OR when the question asks to enumerate ("list all activities", "what categories of facts do we have"). Not the default first move.
3. NARROW — list_facts_by_category(subject, category) to drill into a specific slice once you know the right category.
4. PIVOT — list_entities_in_facts(fact_ids) when results suggest a graph relationship; then traverse_entity_neighborhood for multi-hop.
5. TIME-TRAVEL — list_events_in_window for "what happened between dates X and Y"; as_of(subject, category, ts) for "what was their X at time T"; get_fact_chain_with_history to reconstruct one fact's evolution.
6. DISAMBIGUATE — get_fact_source when fact text is ambiguous; you'll see the original conversation lines.

Don't refuse on the first weak search. If search_facts returned nothing relevant, try one rephrasing (different keywords, broader query) before answering "unknown".

Stop calling tools as soon as you have enough evidence. Don't keep gathering for confidence — gathering past usefulness wastes budget.

Inference questions ("Would X…", "What would X likely…", "Is X likely to…", "What … might Y say about Z", "What … leaning"):
The reference answer for these is itself an inference, not a literal quote from memory. Gather the related facts and COMMIT to the most likely answer. Examples:
- "Would Melanie be considered an ally to the trans community?" + facts about Melanie supporting Caroline's transition openly → Yes
- "Would Caroline pursue writing as a career option?" + facts about her firm intent to do counseling → Likely no
- "Would Melanie go on another roadtrip soon?" + facts about her son's accident on the last one → Likely no
"unknown" is wrong on these whenever relevant evidence exists. Only refuse if the question topic has no related facts at all.

Final answer rules:
- Output the SHORTEST correct phrase. A noun phrase, a date, a name, a list, or a "Yes/No (+ short reason)" for inference questions. Not a long sentence.
- NO citations like [fact:NN]. NO preamble ("Based on the facts:", "The answer is"). NO follow-up explanation.
- If memory doesn't contain enough evidence to answer or defensibly infer, output exactly: unknown
- Match capitalization to the question's expected form (proper nouns capitalized).

Examples:
Q: What is Caroline's identity?
A: Transgender woman

Q: When did Melanie paint a sunrise?
A: 2022

Q: Where has Melanie camped?
A: beach, mountains, forest

Q: Where did Caroline move from 4 years ago?
A: Sweden

Q: Would Caroline likely have Dr. Seuss books on her bookshelf?
A: Yes; she collects classic children's books

Q: Would Melanie go on another roadtrip soon?
A: Likely no; the last one ended badly`;

function defineTool<I>(t: {
  description: string;
  inputSchema: z.ZodType<I>;
  execute: (input: I) => Promise<unknown>;
}): Tool {
  // ai-sdk's tool() helper has overloads heavy enough to OOM tsc when called
  // many times in one builder. Same workaround as src/engine/agent/tools.ts.
  return t as unknown as Tool;
}

interface RetrieverContext {
  db: DB;
  provider: LLMProvider;
  /** Every fact id that surfaced in any tool result. Used as the citation set. */
  seenFactIds: Set<number>;
}

function buildRetrieverTools(ctx: RetrieverContext): ToolSet {
  // ─── browse ───────────────────────────────────────────────
  const list_categories_for_subject = defineTool({
    description:
      'Entry point for navigating one subject. Returns the per-category active fact counts and the latest temporal anchor in each category. Cheap; call this first when the question names a person to see what kind of facts exist before drilling.',
    inputSchema: z.object({
      subject_wa_id: z.string().min(1),
    }),
    execute: async ({ subject_wa_id }) => {
      const counts = factCountsByCategoryForSubject(ctx.db, subject_wa_id);
      return {
        ok: true,
        subject_wa_id,
        total_active: counts.reduce((a, c) => a + c.count, 0),
        categories: counts.map((c) => ({
          category: c.category,
          count: c.count,
          latest_extracted_at_iso: c.latest_extracted_at
            ? new Date(c.latest_extracted_at * 1000).toISOString().slice(0, 10)
            : null,
          latest_event_ts_iso: c.latest_event_ts
            ? new Date(c.latest_event_ts * 1000).toISOString().slice(0, 10)
            : null,
        })),
      };
    },
  });

  const list_facts_by_category = defineTool({
    description:
      'Drill into one (subject, category) slice. Optional time window (Unix seconds, inclusive). Facts ordered most-recent-first by event_ts (or extracted_at when missing).',
    inputSchema: z.object({
      subject_wa_id: z.string().min(1),
      category: FactCategoryZ,
      since_ts: z.number().int().nonnegative().optional(),
      until_ts: z.number().int().nonnegative().optional(),
      limit: z.number().int().min(1).max(200).optional(),
    }),
    execute: async ({ subject_wa_id, category, since_ts, until_ts, limit }) => {
      const facts = factsByCategoryForSubject(ctx.db, subject_wa_id, category, {
        sinceTs: since_ts,
        untilTs: until_ts,
        limit,
      });
      for (const f of facts) ctx.seenFactIds.add(f.id);
      return {
        ok: true,
        subject_wa_id,
        category,
        count: facts.length,
        facts: facts.map(serializeFact),
      };
    },
  });

  // ─── search (vector + filters) ───────────────────────────
  const search_facts = defineTool({
    description:
      'DEFAULT FIRST TOOL for direct fact-lookup questions. Hybrid vector search across ALL facts (every subject, every category). Pass the question itself as `query` — short queries work fine ("grandma gift", "how long married"). Optional filters narrow further; omit them unless you have a specific reason. Filters layer on top of the vector pre-fetch — restrictive filters may return fewer than k.',
    inputSchema: z.object({
      query: z.string().min(1),
      subject_wa_id: z.string().optional(),
      category: FactCategoryZ.optional(),
      since_ts: z.number().int().nonnegative().optional(),
      until_ts: z.number().int().nonnegative().optional(),
      k: z.number().int().min(1).max(30).optional(),
    }),
    execute: async ({ query, subject_wa_id, category, since_ts, until_ts, k }) => {
      const trimmed = query.trim();
      if (!trimmed) return errorResult('query is required');
      const { vector } = await ctx.provider.embed(trimmed, 'query');
      const hits = searchFactsHybrid(ctx.db, vector, {
        subjectWaId: subject_wa_id,
        category,
        sinceTs: since_ts,
        untilTs: until_ts,
        k,
      });
      for (const h of hits) ctx.seenFactIds.add(h.id);
      return {
        ok: true,
        query: trimmed,
        filters: { subject_wa_id, category, since_ts, until_ts },
        hits: hits.map(serializeFact),
      };
    },
  });

  const find_entity = defineTool({
    description:
      'Search the knowledge graph for entities (people, places, orgs, things) by display name. Returns matches with their entity_id for use in traverse_entity_neighborhood / list_facts_mentioning_entity. Empty if the graph wasn\'t built — fall back to subject/text search.',
    inputSchema: z.object({
      query: z.string().min(1),
      entity_type: z
        .enum(['person', 'place', 'organization', 'event', 'preference_topic', 'object', 'concept', 'date'])
        .optional(),
      limit: z.number().int().min(1).max(20).optional(),
    }),
    execute: async ({ query, entity_type, limit }) => {
      const matches = searchEntities(ctx.db, query, limit ?? 8);
      const filtered = entity_type
        ? matches.filter((e) => e.entity_type === entity_type)
        : matches;
      return {
        ok: true,
        query,
        entity_type_filter: entity_type ?? null,
        entities: filtered.map((e) => ({
          id: e.id,
          display_name: e.display_name,
          entity_type: e.entity_type,
          confidence: e.confidence,
        })),
      };
    },
  });

  // ─── graph ────────────────────────────────────────────────
  const traverse_entity_neighborhood = defineTool({
    description:
      'Walk knowledge-graph edges from/to an entity. Optional predicate filter (e.g. only family_of / lives_in). max_hops > 1 follows edges through intermediate entities to answer "Alex\'s sister\'s job" type questions. Empty if graph not built.',
    inputSchema: z.object({
      entity_id: z.number().int().positive(),
      predicate: z.string().optional(),
      max_hops: z.number().int().min(1).max(3).optional(),
    }),
    execute: async ({ entity_id, predicate, max_hops }) => {
      const hops = max_hops ?? 1;
      const visited = new Set<number>([entity_id]);
      let frontier = [entity_id];
      const allEdges: Array<Record<string, unknown>> = [];
      for (let depth = 0; depth < hops && frontier.length > 0; depth++) {
        const next: number[] = [];
        for (const eid of frontier) {
          const edges = graphNeighborhood(ctx.db, eid);
          for (const e of edges) {
            if (predicate && e.predicate !== predicate) continue;
            const sourceName = (e as unknown as { source_display_name?: string }).source_display_name;
            const targetName = (e as unknown as { target_display_name?: string }).target_display_name;
            allEdges.push({
              hop: depth + 1,
              predicate: e.predicate,
              source_entity_id: e.source_entity_id,
              source_display_name: sourceName,
              target_entity_id: e.target_entity_id,
              target_display_name: targetName,
              confidence: e.confidence,
              source_fact_id: e.source_fact_id,
              event_ts: e.event_ts,
            });
            const other = e.source_entity_id === eid ? e.target_entity_id : e.source_entity_id;
            if (!visited.has(other)) {
              visited.add(other);
              next.push(other);
            }
          }
        }
        frontier = next;
      }
      return {
        ok: true,
        entity_id,
        predicate_filter: predicate ?? null,
        max_hops: hops,
        edges: allEdges,
      };
    },
  });

  const list_entities_in_facts = defineTool({
    description:
      'Pivot from facts to graph: given a set of fact_ids, return the entities those facts mention with mention counts and roles. Use after a search/list result to find which graph entities are involved before traversing.',
    inputSchema: z.object({
      fact_ids: z.array(z.number().int().positive()).min(1).max(40),
    }),
    execute: async ({ fact_ids }) => {
      const rows = entitiesForFacts(ctx.db, fact_ids);
      return {
        ok: true,
        fact_ids,
        entities: rows,
      };
    },
  });

  // ─── temporal ─────────────────────────────────────────────
  const list_events_in_window = defineTool({
    description:
      'Events and commitments anchored in [since_ts, until_ts] (Unix seconds, inclusive). Anchor = event_ts when set, otherwise extracted_at. Optional subject filter. Use for "what happened between X and Y" questions.',
    inputSchema: z.object({
      since_ts: z.number().int().nonnegative(),
      until_ts: z.number().int().nonnegative(),
      subject_wa_id: z.string().optional(),
      categories: z.array(FactCategoryZ).optional(),
      limit: z.number().int().min(1).max(200).optional(),
    }),
    execute: async ({ since_ts, until_ts, subject_wa_id, categories, limit }) => {
      const events = eventsInWindow(ctx.db, since_ts, until_ts, {
        subjectWaId: subject_wa_id,
        categories,
        limit,
      });
      for (const f of events) ctx.seenFactIds.add(f.id);
      return {
        ok: true,
        since_ts,
        until_ts,
        subject_wa_id: subject_wa_id ?? null,
        count: events.length,
        events: events.map(serializeFact),
      };
    },
  });

  const as_of = defineTool({
    description:
      'Chain-aware time travel: "what was true about (subject, category) as of `ts`?" Walks update/state_change chains and returns the facts whose state was current at that time. Returns multiple rows on a fork (contradiction).',
    inputSchema: z.object({
      subject_wa_id: z.string().min(1),
      category: FactCategoryZ,
      ts: z.number().int().nonnegative(),
    }),
    execute: async ({ subject_wa_id, category, ts }) => {
      const facts = factsActiveAtTime(ctx.db, subject_wa_id, category, ts);
      for (const f of facts) ctx.seenFactIds.add(f.id);
      return {
        ok: true,
        subject_wa_id,
        category,
        ts,
        ts_iso: new Date(ts * 1000).toISOString().slice(0, 10),
        facts: facts.map(serializeFact),
      };
    },
  });

  const get_fact_chain_with_history = defineTool({
    description:
      'For one fact_id, return its update/state_change chain (older → newer) plus retract/contradict edges. Each chain member shows extracted_at and event_ts so you can read the timeline. Use for "how has X evolved?" or to verify a fact\'s currency.',
    inputSchema: z.object({
      fact_id: z.number().int().positive(),
    }),
    execute: async ({ fact_id }) => {
      const root = getActiveFactById(ctx.db, fact_id);
      if (!root) return errorResult(`fact ${fact_id} is not active`);
      ctx.seenFactIds.add(fact_id);

      // Walk forward (newer): collect every fact that updates `current` until
      // the leaf or a fork.
      const chain: ActiveFactRow[] = [root];
      const seen = new Set<number>([fact_id]);
      let current = fact_id;
      for (let i = 0; i < 32; i++) {
        const updaters = listConnectionsToFact(ctx.db, current).filter(
          (c) => c.predicate === 'update' || c.predicate === 'state_change'
        );
        if (updaters.length === 0) break;
        if (updaters.length > 1) break; // fork — leave the original chain
        const next = getActiveFactById(ctx.db, updaters[0].from_fact_id);
        if (!next || seen.has(next.id)) break;
        chain.push(next);
        seen.add(next.id);
        ctx.seenFactIds.add(next.id);
        current = next.id;
      }

      const leafId = latestInChain(ctx.db, fact_id);
      const incoming = listConnectionsToFact(ctx.db, leafId);
      const outgoing = listConnectionsFromFact(ctx.db, fact_id);

      return {
        ok: true,
        original_fact_id: fact_id,
        leaf_fact_id: leafId,
        chain_length: chain.length,
        chain: chain.map((f) => ({
          ...serializeFact(f),
          extracted_at_iso: new Date(f.extracted_at * 1000).toISOString().slice(0, 10),
          event_ts_iso: f.event_ts
            ? new Date(f.event_ts * 1000).toISOString().slice(0, 10)
            : null,
        })),
        edges_into_leaf: incoming,
        edges_out_of_original: outgoing,
        valid_predicates: CONNECTION_PREDICATES,
      };
    },
  });

  // ─── inspect ──────────────────────────────────────────────
  const get_fact_source = defineTool({
    description:
      'Return the original conversation messages backing a fact. Use only when fact text alone is ambiguous and the literal evidence matters.',
    inputSchema: z.object({
      fact_id: z.number().int().positive(),
      burst_limit: z.number().int().min(1).max(5).optional(),
    }),
    execute: async ({ fact_id, burst_limit }) => {
      const fact = getActiveFactById(ctx.db, fact_id);
      if (!fact) return errorResult(`fact ${fact_id} is not active`);
      ctx.seenFactIds.add(fact_id);
      const bursts = getFactSourceBursts(ctx.db, fact_id, burst_limit ?? 2);
      return {
        ok: true,
        fact: serializeFact(fact),
        bursts: bursts.map((b) => ({
          burst_id: b.burst_id,
          messages: b.messages.map((m) => ({
            ts: m.ts,
            ts_iso: new Date(m.ts * 1000).toISOString().slice(0, 16).replace('T', ' '),
            direction: m.direction,
            body: m.body,
          })),
        })),
      };
    },
  });

  // Optional — kept for graph-mediated subject queries when find_entity has
  // already produced an entity_id. Not in the strategy doc above; the model
  // will discover it from the description.
  const list_facts_mentioning_entity = defineTool({
    description:
      'List active facts whose entity-mentions include this entity_id. Use after find_entity to pull all facts about a graph entity (across subjects).',
    inputSchema: z.object({
      entity_id: z.number().int().positive(),
      limit: z.number().int().min(1).max(100).optional(),
    }),
    execute: async ({ entity_id, limit }) => {
      const rows = listFactsMentioningEntity(ctx.db, entity_id, limit ?? 30);
      for (const r of rows) ctx.seenFactIds.add(r.id);
      return {
        ok: true,
        entity_id,
        facts: rows.map(serializeFact),
      };
    },
  });

  return {
    // browse
    list_categories_for_subject,
    list_facts_by_category,
    // search
    search_facts,
    find_entity,
    // graph
    traverse_entity_neighborhood,
    list_entities_in_facts,
    list_facts_mentioning_entity,
    // temporal
    list_events_in_window,
    as_of,
    get_fact_chain_with_history,
    // inspect
    get_fact_source,
  };
}

export interface RetrieveAgenticResult {
  answer: string;
  citations: number[];
  steps: number;
  finishReason: string;
  toolCalls: Array<{ name: string; ok: boolean }>;
  usage: { tokens_in: number; tokens_out: number; model: string };
}

export interface RetrieveAgenticOptions {
  /** Hard cap on the agent's tool-call steps. Default 10. */
  maxSteps?: number;
  /** Per-step log line. */
  log?: (line: string) => void;
}

/**
 * Drive an agent loop that gathers evidence via read tools and emits a
 * factoid-style answer. Replaces both `composeMemoryBlock` + `chat()` for
 * one question — single LLM thread with tool access end-to-end.
 */
export async function retrieveAgentic(
  db: DB,
  provider: LLMProvider,
  question: string,
  opts: RetrieveAgenticOptions = {}
): Promise<RetrieveAgenticResult> {
  const maxSteps = opts.maxSteps ?? DEFAULT_MAX_STEPS;
  const log = opts.log ?? (() => {});

  const ctx: RetrieverContext = {
    db,
    provider,
    seenFactIds: new Set<number>(),
  };
  const tools = buildRetrieverTools(ctx);

  const subjects = allActiveSubjects(db);
  const subjectsList =
    subjects.length === 0
      ? '  (no active subjects — memory is empty)'
      : subjects
          .map(
            (s) =>
              `  - ${s.subject_wa_id}` +
              (s.display_name ? ` ("${s.display_name}")` : '') +
              ` — ${s.active_fact_count} facts`
          )
          .join('\n');

  const systemPrompt = `${FACTOID_SYSTEM_PROMPT}\n\nKnown subjects in memory:\n${subjectsList}`;

  const toolCalls: Array<{ name: string; ok: boolean }> = [];
  let totalIn = 0;
  let totalOut = 0;

  const result = await generateText({
    model: getCuratorLanguageModel(),
    system: systemPrompt,
    prompt: `Question: ${question}`,
    tools,
    stopWhen: stepCountIs(maxSteps),
    temperature: 0,
    maxRetries: 2,
    onStepFinish: (event) => {
      const e = event as {
        text?: string;
        toolCalls?: Array<{ toolName: string; toolCallId: string; input: unknown }>;
        toolResults?: Array<{ toolCallId: string; output?: unknown }>;
        usage?: { inputTokens?: number; outputTokens?: number };
      };
      totalIn += e.usage?.inputTokens ?? 0;
      totalOut += e.usage?.outputTokens ?? 0;
      for (const c of e.toolCalls ?? []) {
        const matched = (e.toolResults ?? []).find((r) => r.toolCallId === c.toolCallId);
        const ok = isOkResult(matched?.output);
        toolCalls.push({ name: c.toolName, ok });
        log(`  → ${c.toolName} ok=${ok}`);
      }
      if (e.text) log(`  text="${e.text.slice(0, 200)}"`);
    },
  });

  return {
    answer: result.text.trim(),
    citations: [...ctx.seenFactIds],
    steps: result.steps.length,
    finishReason: result.finishReason,
    toolCalls,
    usage: { tokens_in: totalIn, tokens_out: totalOut, model: CURATOR_MODEL_NAME },
  };
}

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

function isOkResult(output: unknown): boolean {
  return (
    !!output &&
    typeof output === 'object' &&
    'ok' in output &&
    (output as { ok: unknown }).ok === true
  );
}
