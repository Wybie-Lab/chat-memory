import { generateObject, generateText } from 'ai';
import { createOpenRouter } from '@openrouter/ai-sdk-provider';
import { z } from 'zod';
import { ENTITY_ROLES, ENTITY_TYPES, GRAPH_PREDICATES } from './provider';
import type {
  LLMProvider,
  BurstInput,
  ChatInput,
  ConsolidateInput,
  ConsolidationOp,
  EmbedMode,
  GraphFactInput,
  SummarizeClusterInput,
  UsageMetadata,
} from './provider';

const DEFAULT_OPENROUTER_MODEL = process.env.OPENROUTER_MODEL ?? 'google/gemini-2.5-flash';
const FILTER_MODEL = process.env.OPENROUTER_FILTER_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const EXTRACT_MODEL = process.env.OPENROUTER_EXTRACT_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const CONSOLIDATE_MODEL = process.env.OPENROUTER_CONSOLIDATE_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const SUMMARIZE_MODEL = process.env.OPENROUTER_SUMMARIZE_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const GRAPH_MODEL = process.env.OPENROUTER_GRAPH_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const CHAT_MODEL = process.env.OPENROUTER_CHAT_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const CURATOR_MODEL = process.env.OPENROUTER_CURATOR_MODEL ?? DEFAULT_OPENROUTER_MODEL;
const EMBED_MODEL = 'embed-multilingual-v3.0';

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY,
  appName: process.env.OPENROUTER_APP_NAME ?? 'manila-memory',
  appUrl: process.env.OPENROUTER_APP_URL,
});

/**
 * Language model used by the curator agent loop. Exposed so the engine
 * (which drives ai-sdk's generateText with native tools) can pick up the
 * same OpenRouter wiring without re-instantiating the client.
 */
export function getCuratorLanguageModel() {
  return openrouter(CURATOR_MODEL);
}

export const CURATOR_MODEL_NAME = CURATOR_MODEL;

const CHAT_SYSTEM_PROMPT = `You answer questions about the user's WhatsApp contacts using only the structured <memory> block provided.

The <memory> block has up to four sections:
- <preferences> — durable preferences about the user, always relevant.
- <known_facts> — atomic facts ranked by relevance to the question. Each line ends with [fact:ID].
- <subject_summaries> — rolled-up prose summaries per (subject, category). Use these for context; cite the underlying facts via [fact:ID] from <known_facts> when applicable.
- <recent_episodes> — recent events and commitments, time-anchored.

Rules:
- If <memory> doesn't contain enough info to answer, say so honestly. Don't invent details, names, or dates.
- Cite supporting facts inline using [fact:ID]. Cite at least one fact per non-trivial claim. Multiple citations are fine.
- A fact's confidence is implied by how it's phrased in <memory>. If wording is hedged ("may have", "reportedly"), echo that hedging in your answer.
- Keep answers concise — one paragraph unless the user asks for detail.
- Match the user's language (English/Italian/etc.).

Return only the answer text — no preamble like "Based on the facts:".`;

const FILTER_SYSTEM_PROMPT = `You are a memory curator. Your job is to decide whether a WhatsApp conversation burst — a contiguous run of messages between two people — contains anything worth remembering long-term, as part of building a personal memory layer about the people in someone's life.

A burst is the WHOLE unit of decision: if even ONE message in the burst contains something durable, KEEP the whole burst.

KEEP if the burst contains:
- Personal facts about a person (job, family, location, health, preferences, opinions)
- ANY mention of family members, friends, partners, or recurring people in the contact's life — even passing references like "say hi to your mom" or "I'm with my grandma" count
- Personality traits or self-descriptions, even if said casually or jokingly ("I'm always late", "I complain a lot")
- Scheduled events with dates or times (appointments, trips, birthdays, plans)
- Commitments (a promise — to act, to deliver, to meet)
- Significant emotional state changes (good news, bad news, milestones)
- Specific concrete information that could be referenced later (names, places, organizations, preferences)

DROP only if the ENTIRE burst is:
- Pure greetings, acknowledgments, reactions ("hi", "hey", "ok", "thanks", "lol", emoji-only)
- Pure ephemeral logistics ("on my way", "5 min late", "where are you", "one sec")
- Generic small talk, idioms, or jokes with no informational content

Lean toward KEEP. False positives are easier to clean later than false negatives are to recover. Drop only when the burst clearly contains nothing durable across all of its lines.

Return your decision strictly as JSON matching the provided schema. The reason field must be one short sentence under 15 words.`;

const EXTRACT_SYSTEM_PROMPT = `You are extracting durable, structured facts about people from a WhatsApp conversation burst — a contiguous run of messages between two people — to build a long-term memory.

You are given the FULL burst, with each line labeled by sender ("me:" or the contact's name). Earlier lines in the burst provide context for later ones (idioms, references, ongoing topics). Extract facts from anywhere in the burst.

For each fact you extract, classify it:
- preference: things they like/dislike, opinions, tastes
- event: scheduled or happened events with dates ("doctor appt next Tuesday", "got promoted last week")
- commitment: a promise (you to them, them to you, them to a third party)
- fact: stable info (job, family member name, address, hometown)
- relationship: who-knows-who connections

Rules:
- Ground every fact in the literal text of the burst. If you can't quote the supporting words, don't extract the fact.
- Resolve relative dates ("tomorrow", "next week") using the burst date provided. If a message says "tomorrow" without indicating what's happening, do NOT extract a fact — the date alone is not the fact.
- Treat colloquialisms, idioms, sarcasm, and dialect phrases as expressive language, not literal facts.
- SUBJECT ATTRIBUTION (critical): each line is labeled with its SENDER. Use the per-line sender to resolve first-person and possessives:
  • First-person ("io", "I", "sono", "I'm", "ho", "I have") → subject is the SENDER OF THAT LINE. So "io sono stanco" sent by me → subject is "me"; sent by the contact → subject is the contact.
  • First-person possessives ("my mom", "i miei", "mio padre", "mia sorella", "my X") refer to that line's SENDER's relations.
  • Second-person ("tu", "you", "sei") → subject is the OTHER party in the burst.
  • Explicitly named subject in the line → use that name.
  • Earlier lines in the burst unambiguously establish the third-person subject by name → use that name.

- UNNAMED THIRD-PERSON RULE (strict, no exceptions): a line that uses a third-person pronoun ("lei", "lui", "loro", "she", "he", "they") OR a dropped third-person subject (Italian/Spanish/Portuguese "deve partire", "se va", "vanno via", "está cansado") with NO named antecedent earlier in this burst is AMBIGUOUS.

  When you cannot name the subject from this burst alone:
  → DROP the fact entirely. Do not extract anything from those lines.
  → DO NOT use the chat label / contact name as a fallback subject.
  → DO NOT rewrite as "<contact> mentioned/discovered/told me that someone is doing X" — that fabricates a meta-fact the contact never asserted, and floods memory with junk.

  The chat label is NOT a default subject. The subject of a fact must be a real, identifiable person. If you can't name them from this burst, skip.

  WORKED EXAMPLE — burst with unresolved third-person "her":
    Friend: did you hear about her promotion
    Friend: she's flying to Tokyo next week
    Friend: lucky her

    WRONG: extract "Friend mentioned someone is flying to Tokyo" (subject=Friend). Friend is the speaker, not the subject; the actual subject ("her" / "she") has no antecedent in this burst.
    WRONG: extract "the contact's colleague is flying to Tokyo" (subject=<chat label>). Adding qualifiers doesn't make the subject identifiable.
    RIGHT: extract nothing from these lines. Return facts: [] (or only facts from OTHER lines in the burst that do have resolvable subjects).
- Confidence: 0.0–1.0. Use values below 0.7 when the fact requires inference. Skip facts you'd rate below 0.4.
- DEDUPE within the burst: if the same fact is restated, extract it once.
- Each fact should be self-contained (readable without the original burst).
- Return an empty list if the burst contains no durable, grounded facts.

- TIME ANCHOR (event_ts) — for category 'event' or 'commitment' ONLY. If the burst contains an unambiguous specific calendar date for when the event takes place / the commitment is due, return event_ts as Unix seconds (UTC midnight is fine if only the date is known; pick the message-time hour if a specific clock time was given). The burst date is provided to you — use it to anchor relative phrases like "tomorrow", "next Tuesday", "this Sunday" only when the day is unambiguously determinable.
  Return event_ts ONLY when:
    - the date is fully specified (year + month + day), OR
    - a relative phrase resolves to a specific day given the burst date (e.g. "tomorrow" → burst_date + 1d; "next Tuesday" → the upcoming Tuesday).
  OMIT event_ts (return null or leave the field out) when:
    - the date is partial ("in the summer", "later this year", "next month"),
    - the date is ambiguous ("when she gets back", "after the project"),
    - the event has no date at all (durable facts about a person's life).
  Wrong dates are worse than missing dates. Be conservative.

Return your output strictly as JSON matching the provided schema.`;

const FILTER_SCHEMA_JSON = {
  type: 'object',
  properties: {
    keep: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['keep', 'reason'],
  additionalProperties: false,
} as const;

const FilterZod = z.object({
  keep: z.boolean(),
  reason: z.string(),
});

const FACT_CATEGORIES = ['preference', 'event', 'commitment', 'fact', 'relationship'] as const;

const EXTRACT_SCHEMA_JSON = {
  type: 'object',
  properties: {
    facts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          subject: { type: 'string' },
          category: { type: 'string', enum: [...FACT_CATEGORIES] },
          content: { type: 'string' },
          confidence: { type: 'number' },
          // Optional. Unix seconds. Set ONLY for unambiguous event/commitment
          // dates. Omit (or null) otherwise.
          event_ts: { type: ['number', 'null'] },
        },
        required: ['subject', 'category', 'content', 'confidence'],
        additionalProperties: false,
      },
    },
  },
  required: ['facts'],
  additionalProperties: false,
} as const;

const ExtractZod = z.object({
  facts: z.array(
    z.object({
      subject: z.string(),
      category: z.enum(FACT_CATEGORIES),
      content: z.string(),
      confidence: z.number().min(0).max(1),
      event_ts: z.number().int().nullable().optional(),
    })
  ),
});

const CONSOLIDATE_SYSTEM_PROMPT = `You are merging new candidate facts into an existing memory about ONE subject. For each candidate, decide one of:

- ADD: candidate is genuinely new info not covered by any existing fact. Insert as a new fact.
- UPDATE: candidate refines, replaces, or contradicts ONE specific existing fact (named by old_fact_id). The existing fact will be marked superseded; the candidate becomes the new active fact.
- DELETE: an existing fact is now known to be wrong or obsolete, and the candidate provides no replacement worth keeping. Remove the existing fact; do not add the candidate.
- DROP: the candidate is fully redundant with an existing fact (same info, no new detail). Ignore the candidate; keep memory unchanged.

Rules:
- Each candidate must appear in exactly one op (ADD, UPDATE, or DROP). DELETE ops do not consume a candidate.
- An existing fact may appear in at most one op (UPDATE or DELETE) — never both.
- Existing facts not mentioned in any op are kept as-is.
- Prefer UPDATE over ADD+DELETE pairs when a candidate clearly supersedes one specific fact (e.g. "lives in Berlin" → "lives in Lisbon").
- Prefer DROP over UPDATE when the candidate adds no real information ("works at Stripe" candidate when existing fact already says "works at Stripe as a backend engineer").
- For UPDATE: copy the candidate's content/category/confidence verbatim into the op (these become the new fact's stored values). Do NOT merge or paraphrase.
- For ADD: same — content/category/confidence come from the candidate.
- For DELETE: provide a one-sentence reason citing the contradicting candidate.
- For DROP: provide a one-sentence reason naming the redundant existing fact.

WORKED EXAMPLE.
Subject: Sam
Existing facts:
  [3] (preference, conf=0.85, 30d) Sam loves jazz music, especially Coltrane.
  [7] (fact, conf=0.90, 60d) Sam lives in Berlin and works at Stripe.
  [12] (event, conf=0.75, 5d) Sam is going to Umbria Jazz festival in July.

New candidates:
  [0] (fact, conf=0.92) Sam lives in Lisbon now (moved last month).
  [1] (preference, conf=0.80) Sam likes jazz.
  [2] (commitment, conf=0.85) Sam promised to send me her Umbria Jazz photos.

Correct ops:
  - UPDATE: candidate_index=0, old_fact_id=7 (Sam moved Berlin → Lisbon — the Stripe employer detail is also part of fact 7, but the candidate is silent on it; choose UPDATE because candidate clearly supersedes the location, and re-asserting Stripe without evidence would invent info.)
  - DROP: candidate_index=1, reason="redundant with fact 3 which already records jazz preference with more detail."
  - ADD: candidate_index=2 (new commitment, no existing fact covers it).

Return ops strictly as JSON matching the schema.`;

const CONSOLIDATE_SCHEMA_JSON = {
  type: 'object',
  properties: {
    ops: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          op: { type: 'string', enum: ['ADD', 'UPDATE', 'DELETE', 'DROP'] },
          candidate_index: { type: 'integer' },
          old_fact_id: { type: 'integer' },
          content: { type: 'string' },
          category: { type: 'string', enum: [...FACT_CATEGORIES] },
          confidence: { type: 'number' },
          reason: { type: 'string' },
        },
        required: ['op'],
        additionalProperties: false,
      },
    },
  },
  required: ['ops'],
  additionalProperties: false,
} as const;

const ConsolidateZod = z.object({
  ops: z.array(
    z.object({
      op: z.enum(['ADD', 'UPDATE', 'DELETE', 'DROP']),
      candidate_index: z.number().int().optional(),
      old_fact_id: z.number().int().optional(),
      content: z.string().optional(),
      category: z.enum(FACT_CATEGORIES).optional(),
      confidence: z.number().min(0).max(1).optional(),
      reason: z.string().optional(),
    })
  ),
});

const SUMMARIZE_SYSTEM_PROMPT = `You are writing a one-paragraph rolled-up summary of what we know about ONE person within ONE category, drawn from a list of atomic facts. The summary will be injected into a memory block that an AI assistant reads at chat time, instead of the raw facts. The AI reads prose better than triples — your job is to make these facts readable as one coherent picture.

Rules:
- Output a single short paragraph: 1–4 sentences, ≤ 350 characters total.
- Synthesize: integrate overlapping facts. Don't list each one separately.
- Stay grounded: every claim must be supported by the input facts. Do NOT invent details, dates, or relationships.
- Hedge low-confidence content. If a fact has confidence < 0.6, qualify it ("may have", "reportedly") or omit it.
- Use the subject's name as given. If contactDisplayName is provided, prefer it; otherwise use the subject id verbatim.
- Match the dominant language of the source facts (Italian/English/etc.). If mixed, pick whichever is the majority.
- Plain prose only. No bullet points, no headings, no fact IDs, no preamble like "Here is a summary:".`;

const SUMMARIZE_SCHEMA_JSON = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
  },
  required: ['summary'],
  additionalProperties: false,
} as const;

const SummarizeZod = z.object({
  summary: z.string(),
});

const GRAPH_SYSTEM_PROMPT = `You extract a small, source-backed knowledge graph from ONE already-consolidated memory fact.

The input fact is already cleaned and grounded. Your job is to identify explicit entities and explicit relationships supported by that fact. The graph is a projection of the fact, not a place for inference.

Allowed entity types:
- person
- place
- organization
- event
- preference_topic
- object
- concept
- date

Allowed entity roles:
- subject
- object
- person
- place
- organization
- event
- date
- topic
- source
- recipient

Allowed predicates:
- knows
- friend_of
- family_of
- partner_of
- works_at
- studies_at
- lives_in
- from_place
- located_in
- likes
- dislikes
- interested_in
- attending
- planning
- visited
- promised_to
- needs
- owns
- part_of
- mentioned

Rules:
- Extract only what the fact literally supports. Do not infer hidden relationships.
- Prefer no edge over a weak edge.
- Use the fact subject as an entity when the fact is about that subject.
- Every edge endpoint must refer to an entity in the entities array.
- Use stable local_id values like "subject", "venice", "riley"; local_id only needs to be unique within this response.
- For relationship facts, capture the actual relationship if explicit.
- For vague references where no stronger predicate is supported, use mentioned sparingly.
- For planned future events, prefer planning or attending. Do not convert plans into completed facts.
- For promises, use promised_to with the promiser as source and recipient as target when both are explicit.
- Do not create located_in edges unless the fact explicitly says one thing is located in another.
- Return empty arrays if the fact has no explicit graph structure.

Return JSON only, matching the provided schema.`;

const GRAPH_SCHEMA_JSON = {
  type: 'object',
  properties: {
    entities: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          local_id: { type: 'string' },
          type: { type: 'string', enum: [...ENTITY_TYPES] },
          display_name: { type: 'string' },
          aliases: { type: 'array', items: { type: 'string' } },
          confidence: { type: 'number' },
        },
        required: ['local_id', 'type', 'display_name', 'confidence'],
        additionalProperties: false,
      },
    },
    mentions: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          entity_local_id: { type: 'string' },
          role: { type: 'string', enum: [...ENTITY_ROLES] },
          mention_text: { type: 'string' },
          confidence: { type: 'number' },
        },
        required: ['entity_local_id', 'role', 'confidence'],
        additionalProperties: false,
      },
    },
    edges: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          source_local_id: { type: 'string' },
          predicate: { type: 'string', enum: [...GRAPH_PREDICATES] },
          target_local_id: { type: 'string' },
          confidence: { type: 'number' },
          event_ts: { type: ['number', 'null'] },
          valid_from_ts: { type: ['number', 'null'] },
          valid_to_ts: { type: ['number', 'null'] },
          qualifiers: { type: 'object', additionalProperties: true },
        },
        required: ['source_local_id', 'predicate', 'target_local_id', 'confidence'],
        additionalProperties: false,
      },
    },
  },
  required: ['entities', 'mentions', 'edges'],
  additionalProperties: false,
} as const;

const GraphZod = z.object({
  entities: z.array(
    z.object({
      local_id: z.string().min(1),
      type: z.enum(ENTITY_TYPES),
      display_name: z.string().min(1),
      aliases: z.array(z.string()).optional(),
      confidence: z.number().min(0).max(1),
    })
  ),
  mentions: z.array(
    z.object({
      entity_local_id: z.string().min(1),
      role: z.enum(ENTITY_ROLES),
      mention_text: z.string().optional(),
      confidence: z.number().min(0).max(1),
    })
  ),
  edges: z.array(
    z.object({
      source_local_id: z.string().min(1),
      predicate: z.enum(GRAPH_PREDICATES),
      target_local_id: z.string().min(1),
      confidence: z.number().min(0).max(1),
      event_ts: z.number().int().nullable().optional(),
      valid_from_ts: z.number().int().nullable().optional(),
      valid_to_ts: z.number().int().nullable().optional(),
      qualifiers: z.record(z.unknown()).optional(),
    })
  ),
});

const LooseGraphZod = z.object({
  entities: z
    .array(
      z.object({
        local_id: z.string().min(1).optional(),
        id: z.string().min(1).optional(),
        type: z.string().min(1).optional(),
        display_name: z.string().min(1).optional(),
        name: z.string().min(1).optional(),
        labels: z.array(z.string()).optional(),
        aliases: z.array(z.string()).optional(),
        confidence: z.number().min(0).max(1).optional(),
      }).passthrough()
    )
    .default([]),
  mentions: z
    .array(
      z.object({
        entity_local_id: z.string().min(1).optional(),
        entity_id: z.string().min(1).optional(),
        role: z.string().min(1).optional(),
        mention_text: z.string().optional(),
        text: z.string().optional(),
        confidence: z.number().min(0).max(1).optional(),
      }).passthrough()
    )
    .default([]),
  edges: z
    .array(
      z.object({
        source_local_id: z.string().min(1).optional(),
        target_local_id: z.string().min(1).optional(),
        source: z.string().min(1).optional(),
        target: z.string().min(1).optional(),
        subject: z.string().min(1).optional(),
        object: z.string().min(1).optional(),
        predicate: z.string().min(1).optional(),
        confidence: z.number().min(0).max(1).optional(),
        event_ts: z.number().int().nullable().optional(),
        valid_from_ts: z.number().int().nullable().optional(),
        valid_to_ts: z.number().int().nullable().optional(),
        qualifiers: z.record(z.unknown()).optional(),
      }).passthrough()
    )
    .default([]),
});

interface AgentCallResult {
  structured: unknown;
  tokens_in: number;
  tokens_out: number;
}

interface TextCallResult {
  text: string;
  tokens_in: number;
  tokens_out: number;
}

async function callAgent(args: {
  systemPrompt: string;
  userPrompt: string;
  model: string;
  outputSchema: z.ZodTypeAny;
  outputSchemaJson?: Record<string, unknown>;
}): Promise<AgentCallResult> {
  const generate = generateObject as unknown as (options: {
    model: unknown;
    system: string;
    prompt: string;
    schema: z.ZodTypeAny;
    temperature: number;
    maxRetries: number;
  }) => Promise<{ object: unknown; usage: { inputTokens?: number; outputTokens?: number } }>;

  try {
    const result = await generate({
      model: openrouter(args.model),
      system: args.systemPrompt,
      prompt: args.userPrompt,
      schema: args.outputSchema,
      temperature: 0,
      maxRetries: 2,
    });
    return {
      structured: result.object,
      tokens_in: result.usage.inputTokens ?? 0,
      tokens_out: result.usage.outputTokens ?? 0,
    };
  } catch (err) {
    const fallback = await callText({
      model: args.model,
      systemPrompt: `${args.systemPrompt}

Return ONLY valid JSON. Do not wrap it in markdown.${
        args.outputSchemaJson
          ? `\nThe JSON must match this schema:\n${JSON.stringify(args.outputSchemaJson)}`
          : ''
      }`,
      userPrompt: args.userPrompt,
    });
    return {
      structured: parseStructuredJson(fallback.text),
      tokens_in: fallback.tokens_in,
      tokens_out: fallback.tokens_out,
    };
  }
}

async function callText(args: {
  systemPrompt: string;
  userPrompt: string;
  model: string;
}): Promise<TextCallResult> {
  const result = await generateText({
    model: openrouter(args.model),
    system: args.systemPrompt,
    prompt: args.userPrompt,
    temperature: 0,
    maxRetries: 2,
  });
  return {
    text: result.text,
    tokens_in: result.usage.inputTokens ?? 0,
    tokens_out: result.usage.outputTokens ?? 0,
  };
}

function parseStructuredJson(content: string): unknown {
  const trimmed = content.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fenced) return JSON.parse(fenced[1]);
    const first = trimmed.indexOf('{');
    const last = trimmed.lastIndexOf('}');
    if (first >= 0 && last > first) return JSON.parse(trimmed.slice(first, last + 1));
    throw new Error(`openrouter returned non-JSON structured output: ${trimmed.slice(0, 200)}`);
  }
}

export class OpenRouterProvider implements LLMProvider {
  async filterBurst(input: BurstInput) {
    const result = await callAgent({
      systemPrompt: FILTER_SYSTEM_PROMPT,
      userPrompt: formatBurstMessage(input, 'filter'),
      model: FILTER_MODEL,
      outputSchema: FilterZod,
      outputSchemaJson: FILTER_SCHEMA_JSON as unknown as Record<string, unknown>,
    });
    const parsed = FilterZod.parse(result.structured);
    return {
      keep: parsed.keep,
      reason: parsed.reason,
      usage: usageOf(FILTER_MODEL, result),
    };
  }

  async extractBurst(input: BurstInput) {
    const result = await callAgent({
      systemPrompt: EXTRACT_SYSTEM_PROMPT,
      userPrompt: formatBurstMessage(input, 'extract'),
      model: EXTRACT_MODEL,
      outputSchema: ExtractZod,
      outputSchemaJson: EXTRACT_SCHEMA_JSON as unknown as Record<string, unknown>,
    });
    const parsed = ExtractZod.parse(result.structured);
    return {
      facts: parsed.facts,
      usage: usageOf(EXTRACT_MODEL, result),
    };
  }

  async consolidate(input: ConsolidateInput) {
    const result = await callAgent({
      systemPrompt: CONSOLIDATE_SYSTEM_PROMPT,
      userPrompt: formatConsolidateMessage(input),
      model: CONSOLIDATE_MODEL,
      outputSchema: ConsolidateZod,
      outputSchemaJson: CONSOLIDATE_SCHEMA_JSON as unknown as Record<string, unknown>,
    });
    const parsed = ConsolidateZod.parse(result.structured);

    const ops: ConsolidationOp[] = [];
    for (const raw of parsed.ops) {
      const validated = validateOp(raw, input);
      if (validated) ops.push(validated);
    }
    return {
      ops,
      usage: usageOf(CONSOLIDATE_MODEL, result),
    };
  }

  async summarizeCluster(input: SummarizeClusterInput) {
    const result = await callAgent({
      systemPrompt: SUMMARIZE_SYSTEM_PROMPT,
      userPrompt: formatSummarizeMessage(input),
      model: SUMMARIZE_MODEL,
      outputSchema: SummarizeZod,
      outputSchemaJson: SUMMARIZE_SCHEMA_JSON as unknown as Record<string, unknown>,
    });
    const parsed = SummarizeZod.parse(result.structured);
    return {
      summary: parsed.summary.trim(),
      usage: usageOf(SUMMARIZE_MODEL, result),
    };
  }

  async extractGraphFromFact(input: GraphFactInput) {
    const result = await callText({
      systemPrompt: `${GRAPH_SYSTEM_PROMPT}

Return ONLY a JSON object with this exact shape:
{"entities":[],"mentions":[],"edges":[]}
Do not wrap it in markdown.`,
      userPrompt: formatGraphFactMessage(input),
      model: GRAPH_MODEL,
    });
    const parsed = normalizeGraph(LooseGraphZod.parse(parseStructuredJson(result.text)));
    const localIds = new Set(parsed.entities.map((e) => e.local_id));
    return {
      graph: {
        entities: parsed.entities,
        mentions: parsed.mentions.filter((m) => localIds.has(m.entity_local_id)),
        edges: parsed.edges.filter(
          (e) => localIds.has(e.source_local_id) && localIds.has(e.target_local_id)
        ),
      },
      usage: { model: GRAPH_MODEL, tokens_in: result.tokens_in, tokens_out: result.tokens_out },
    };
  }

  async embed(text: string, mode: EmbedMode = 'document') {
    const apiKey = process.env.COHERE_API_KEY;
    if (!apiKey) {
      throw new Error('COHERE_API_KEY not set');
    }

    const r = await fetch('https://api.cohere.com/v2/embed', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        texts: [text],
        model: EMBED_MODEL,
        input_type: mode === 'query' ? 'search_query' : 'search_document',
        embedding_types: ['float'],
      }),
    });

    if (!r.ok) {
      throw new Error(`cohere embed failed (${r.status}): ${await r.text()}`);
    }

    const data = (await r.json()) as {
      embeddings: { float: number[][] };
      meta?: { billed_units?: { input_tokens?: number } };
    };

    return {
      vector: data.embeddings.float[0],
      usage: {
        model: EMBED_MODEL,
        tokens_in: data.meta?.billed_units?.input_tokens ?? 0,
        tokens_out: 0,
      },
    };
  }

  async chat(input: ChatInput) {
    const result = await callText({
      model: CHAT_MODEL,
      systemPrompt: CHAT_SYSTEM_PROMPT,
      userPrompt: formatChatMessage(input),
    });
    return {
      answer: result.text.trim(),
      usage: {
        model: CHAT_MODEL,
        tokens_in: result.tokens_in,
        tokens_out: result.tokens_out,
      },
    };
  }
}

export function createLLMProvider(): LLMProvider {
  const kind = (process.env.LLM_PROVIDER ?? 'openrouter').toLowerCase();
  switch (kind) {
    case 'openrouter':
      return new OpenRouterProvider();
    default:
      throw new Error(`unknown LLM provider: ${kind}`);
  }
}

function formatBurstMessage(input: BurstInput, mode: 'filter' | 'extract'): string {
  const contactLabel = input.contactName ?? 'unknown';
  const startISO = new Date(input.startTs * 1000).toISOString().slice(0, 16).replace('T', ' ');
  const endISO = new Date(input.endTs * 1000).toISOString().slice(0, 16).replace('T', ' ');
  const sameDay = startISO.slice(0, 10) === endISO.slice(0, 10);
  const span = sameDay ? `${startISO} → ${endISO.slice(11)}` : `${startISO} → ${endISO}`;

  const lines = [
    `Contact: ${contactLabel}${input.contactNotes ? ` — ${input.contactNotes}` : ''}`,
    `Burst span: ${span} UTC (${input.lines.length} messages)`,
  ];
  if (input.isGroup) lines.push('Group chat');
  lines.push('', 'Conversation burst:');
  for (const line of input.lines) {
    const sender = line.direction === 'out' ? 'me' : contactLabel;
    lines.push(`  ${sender}: ${line.body}`);
  }
  lines.push('');
  if (mode === 'filter') {
    lines.push('Does this burst contain anything worth remembering?');
  } else {
    lines.push('Extract facts from this burst.');
  }
  return lines.join('\n');
}

function usageOf(model: string, result: AgentCallResult): UsageMetadata {
  return {
    model,
    tokens_in: result.tokens_in,
    tokens_out: result.tokens_out,
  };
}

function normalizeGraph(raw: z.infer<typeof LooseGraphZod>): z.infer<typeof GraphZod> {
  const entityTypes = new Set<string>(ENTITY_TYPES);
  const entityRoles = new Set<string>(ENTITY_ROLES);
  const predicates = new Set<string>(GRAPH_PREDICATES);

  const entities = raw.entities
    .map((e) => {
      const localId = e.local_id ?? e.id ?? e.name ?? e.display_name ?? e.labels?.[0];
      const displayName = e.display_name ?? e.name ?? e.labels?.[0] ?? e.id ?? e.local_id;
      return {
        local_id: localId ?? '',
        type: normalizeEntityType(e.type),
        display_name: displayName ?? '',
        aliases: e.aliases ?? e.labels?.filter((label) => label !== displayName),
        confidence: e.confidence ?? 0.7,
      };
    })
    .filter((e) => e.local_id && e.display_name && entityTypes.has(e.type));

  const localIds = new Set(entities.map((e) => e.local_id));
  const entityTypeById = new Map(entities.map((e) => [e.local_id, e.type]));

  const mentions = raw.mentions
    .map((m) => {
      const entityId = m.entity_local_id ?? m.entity_id;
      return {
        entity_local_id: entityId ?? '',
        role: normalizeRole(m.role, entityId ? entityTypeById.get(entityId) : undefined),
        mention_text: m.mention_text ?? m.text,
        confidence: m.confidence ?? 0.7,
      };
    })
    .filter((m) => localIds.has(m.entity_local_id) && entityRoles.has(m.role));

  const edges = raw.edges
    .map((e) => ({
      source_local_id: e.source_local_id ?? e.source ?? e.subject ?? '',
      predicate: normalizePredicate(e.predicate),
      target_local_id: e.target_local_id ?? e.target ?? e.object ?? '',
      confidence: e.confidence ?? 0.7,
      event_ts: e.event_ts,
      valid_from_ts: e.valid_from_ts,
      valid_to_ts: e.valid_to_ts,
      qualifiers: e.qualifiers,
    }))
    .filter(
      (e) =>
        localIds.has(e.source_local_id) &&
        localIds.has(e.target_local_id) &&
        predicates.has(e.predicate)
    );

  return GraphZod.parse({ entities, mentions, edges });
}

function normalizeToken(value: string | undefined): string {
  return (value ?? '').trim().toLowerCase().replace(/[\s-]+/g, '_');
}

function normalizeEntityType(value: string | undefined): string {
  const token = normalizeToken(value);
  if (token === 'location' || token === 'city' || token === 'country') return 'place';
  if (token === 'company' || token === 'school' || token === 'university') return 'organization';
  if (token === 'transportation_method' || token === 'vehicle') return 'object';
  if (token === 'topic') return 'preference_topic';
  return token || 'concept';
}

function normalizeRole(value: string | undefined, entityType: string | undefined): string {
  const token = normalizeToken(value);
  if (token && token !== 'unknown') return token;
  if (entityType === 'place') return 'place';
  if (entityType === 'organization') return 'organization';
  if (entityType === 'event') return 'event';
  if (entityType === 'date') return 'date';
  if (entityType === 'preference_topic' || entityType === 'concept') return 'topic';
  if (entityType === 'person') return 'person';
  return 'object';
}

function normalizePredicate(value: string | undefined): string {
  const token = normalizeToken(value);
  if (token === 'is_visiting' || token === 'visiting' || token === 'goes_to') return 'visited';
  if (token === 'leaving' || token === 'leaving_from' || token === 'departing_from') {
    return 'from_place';
  }
  if (token === 'arriving_by' || token === 'travels_by' || token === 'using') return 'mentioned';
  if (token === 'works_for') return 'works_at';
  if (token === 'lives_at') return 'lives_in';
  return token;
}

function formatConsolidateMessage(input: ConsolidateInput): string {
  const lines = [`Subject: ${input.subject}`, '', 'Existing facts:'];
  if (input.existing.length === 0) {
    lines.push('  (none — every candidate should be ADD)');
  } else {
    for (const f of input.existing) {
      lines.push(
        `  [${f.id}] (${f.category}, conf=${f.confidence.toFixed(2)}, ${Math.round(f.age_days)}d) ${f.content}`
      );
    }
  }
  lines.push('', 'New candidates:');
  for (let i = 0; i < input.candidates.length; i++) {
    const c = input.candidates[i];
    lines.push(`  [${i}] (${c.category}, conf=${c.confidence.toFixed(2)}) ${c.content}`);
  }
  lines.push('', 'Decide ops for each candidate. Return JSON.');
  return lines.join('\n');
}

/**
 * Reject malformed ops (missing required fields for the chosen op type) and
 * out-of-range candidate/fact references. Returning null silently drops one
 * op rather than failing the whole burst — better than nothing, and any
 * dropped candidate will simply be re-extracted next time the same content
 * appears.
 */
function validateOp(
  raw: z.infer<typeof ConsolidateZod>['ops'][number],
  input: ConsolidateInput
): ConsolidationOp | null {
  const candIdxOk =
    raw.candidate_index !== undefined &&
    raw.candidate_index >= 0 &&
    raw.candidate_index < input.candidates.length;
  const oldIdOk =
    raw.old_fact_id !== undefined &&
    input.existing.some((e) => e.id === raw.old_fact_id);

  switch (raw.op) {
    case 'ADD':
      if (!candIdxOk || !raw.content || !raw.category || raw.confidence === undefined) return null;
      return {
        op: 'ADD',
        candidate_index: raw.candidate_index!,
        content: raw.content,
        category: raw.category,
        confidence: raw.confidence,
      };
    case 'UPDATE':
      if (!candIdxOk || !oldIdOk || !raw.content || !raw.category || raw.confidence === undefined) {
        return null;
      }
      return {
        op: 'UPDATE',
        candidate_index: raw.candidate_index!,
        old_fact_id: raw.old_fact_id!,
        content: raw.content,
        category: raw.category,
        confidence: raw.confidence,
      };
    case 'DELETE':
      if (!oldIdOk) return null;
      return {
        op: 'DELETE',
        old_fact_id: raw.old_fact_id!,
        reason: raw.reason ?? '(no reason given)',
      };
    case 'DROP':
      if (!candIdxOk) return null;
      return {
        op: 'DROP',
        candidate_index: raw.candidate_index!,
        reason: raw.reason ?? '(no reason given)',
      };
  }
}

function formatSummarizeMessage(input: SummarizeClusterInput): string {
  const subjectLabel = input.contactDisplayName
    ? `${input.contactDisplayName} (id: ${input.subject})`
    : input.subject;
  const lines = [
    `Subject: ${subjectLabel}`,
    `Category: ${input.category}`,
    '',
    'Active facts:',
  ];
  for (const f of input.facts) {
    lines.push(
      `  [${f.id}] (conf=${f.confidence.toFixed(2)}, ${Math.round(f.age_days)}d old) ${f.content}`
    );
  }
  lines.push('', `Write the rolled-up summary for "${subjectLabel}" (${input.category}).`);
  return lines.join('\n');
}

function formatGraphFactMessage(input: GraphFactInput): string {
  const lines = [
    `Fact ID: ${input.fact_id}`,
    `Subject: ${input.subject}`,
    `Category: ${input.category}`,
    `Confidence: ${input.confidence.toFixed(2)}`,
  ];
  if (input.event_ts) {
    lines.push(`Event time: ${new Date(input.event_ts * 1000).toISOString()}`);
  }
  lines.push('', 'Fact:', input.content, '', 'Extract the explicit graph projection for this fact.');
  return lines.join('\n');
}

function formatChatMessage(input: ChatInput): string {
  return [input.memoryBlock, '', `Question: ${input.question}`].join('\n');
}
