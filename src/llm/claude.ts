import { query } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import type {
  LLMProvider,
  BurstInput,
  ChatInput,
  ConsolidateInput,
  ConsolidationOp,
  EmbedMode,
  UsageMetadata,
} from './provider';

const FILTER_MODEL = 'claude-haiku-4-5';
const EXTRACT_MODEL = 'claude-sonnet-4-6';
const CONSOLIDATE_MODEL = 'claude-sonnet-4-6';
const CHAT_MODEL = 'claude-sonnet-4-6';
const EMBED_MODEL = 'embed-multilingual-v3.0';

const CHAT_SYSTEM_PROMPT = `You answer questions about the user's WhatsApp contacts using only the provided facts. Each fact has a confidence score (0-1) and an ID.

Rules:
- If the facts don't contain enough info to answer, say so honestly. Don't invent details.
- Cite specific facts inline using the format [fact:ID]. Use multiple citations when supporting a claim.
- Lower-confidence facts (<0.7) should be hedged ("she may have mentioned..." rather than "she said...").
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
Subject: Anna
Existing facts:
  [3] (preference, conf=0.85, 30d) Anna loves jazz music, especially Coltrane.
  [7] (fact, conf=0.90, 60d) Anna lives in Berlin and works at Stripe.
  [12] (event, conf=0.75, 5d) Anna is going to Umbria Jazz festival in July.

New candidates:
  [0] (fact, conf=0.92) Anna lives in Lisbon now (moved last month).
  [1] (preference, conf=0.80) Anna likes jazz.
  [2] (commitment, conf=0.85) Anna promised to send me her Umbria Jazz photos.

Correct ops:
  - UPDATE: candidate_index=0, old_fact_id=7 (Anna moved Berlin → Lisbon — the Stripe employer detail is also part of fact 7, but the candidate is silent on it; choose UPDATE because candidate clearly supersedes the location, and re-asserting Stripe without evidence would invent info.)
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

interface AgentCallResult {
  structured: unknown;
  tokens_in: number;
  tokens_out: number;
}

async function callAgent(args: {
  systemPrompt: string;
  userPrompt: string;
  model: string;
  outputSchema: Record<string, unknown>;
}): Promise<AgentCallResult> {
  const stream = query({
    prompt: args.userPrompt,
    options: {
      systemPrompt: args.systemPrompt,
      model: args.model,
      outputFormat: { type: 'json_schema', schema: args.outputSchema },
      allowedTools: [],
      maxTurns: 5,
      settingSources: [],
    },
  });

  let structured: unknown = undefined;
  let tokens_in = 0;
  let tokens_out = 0;
  const trace: string[] = [];

  for await (const message of stream) {
    if (message.type === 'assistant') {
      const blocks = message.message.content
        .map((b: { type: string; text?: string }) =>
          b.type === 'text' && typeof b.text === 'string' ? `text(${b.text.slice(0, 80)})` : b.type
        )
        .join(',');
      trace.push(`assistant[${blocks}]${message.error ? ` ERR:${message.error}` : ''}`);
      continue;
    }
    if (message.type !== 'result') continue;

    if (message.subtype !== 'success') {
      const errs = message.errors.length ? message.errors.join('; ') : '(no error message)';
      throw new Error(
        `claude-agent-sdk ${message.subtype}: ${errs} | turns=${message.num_turns} | trace=${trace.join(' → ')}`
      );
    }

    structured = message.structured_output;
    for (const u of Object.values(message.modelUsage)) {
      tokens_in += u.inputTokens;
      tokens_out += u.outputTokens;
    }
  }

  if (structured === undefined) {
    throw new Error(`claude-agent-sdk returned no structured output. trace=${trace.join(' → ')}`);
  }

  return { structured, tokens_in, tokens_out };
}

export class ClaudeProvider implements LLMProvider {
  async filterBurst(input: BurstInput) {
    const result = await callAgent({
      systemPrompt: FILTER_SYSTEM_PROMPT,
      userPrompt: formatBurstMessage(input, 'filter'),
      model: FILTER_MODEL,
      outputSchema: FILTER_SCHEMA_JSON as unknown as Record<string, unknown>,
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
      outputSchema: EXTRACT_SCHEMA_JSON as unknown as Record<string, unknown>,
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
      outputSchema: CONSOLIDATE_SCHEMA_JSON as unknown as Record<string, unknown>,
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
    const userPrompt = formatChatMessage(input);
    const stream = query({
      prompt: userPrompt,
      options: {
        systemPrompt: CHAT_SYSTEM_PROMPT,
        model: CHAT_MODEL,
        allowedTools: [],
        maxTurns: 5,
        settingSources: [],
      },
    });

    let answer = '';
    let tokens_in = 0;
    let tokens_out = 0;

    for await (const message of stream) {
      if (message.type === 'assistant') {
        for (const block of message.message.content) {
          const b = block as { type: string; text?: string };
          if (b.type === 'text' && typeof b.text === 'string') {
            answer += b.text;
          }
        }
      } else if (message.type === 'result') {
        if (message.subtype !== 'success') {
          const errs = message.errors.length ? message.errors.join('; ') : '(no error message)';
          throw new Error(`claude-agent-sdk chat ${message.subtype}: ${errs}`);
        }
        for (const u of Object.values(message.modelUsage)) {
          tokens_in += u.inputTokens;
          tokens_out += u.outputTokens;
        }
      }
    }

    return {
      answer: answer.trim(),
      usage: { model: CHAT_MODEL, tokens_in, tokens_out },
    };
  }
}

export function createLLMProvider(): LLMProvider {
  const kind = (process.env.LLM_PROVIDER ?? 'claude').toLowerCase();
  switch (kind) {
    case 'claude':
      return new ClaudeProvider();
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

function formatChatMessage(input: ChatInput): string {
  const lines = [`Question: ${input.question}`, '', 'Facts (sorted by relevance):'];
  if (input.facts.length === 0) {
    lines.push('  (none)');
  } else {
    for (const f of input.facts) {
      lines.push(
        `  [fact:${f.id}] (${f.category}, confidence=${f.confidence.toFixed(2)}, about=${f.subject}) ${f.content}`
      );
    }
  }
  return lines.join('\n');
}
