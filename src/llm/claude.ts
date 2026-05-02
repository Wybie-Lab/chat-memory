import { query } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import type {
  LLMProvider,
  FilterInput,
  ExtractInput,
  ChatInput,
  EmbedMode,
  UsageMetadata,
} from './provider';

const FILTER_MODEL = 'claude-haiku-4-5';
const EXTRACT_MODEL = 'claude-sonnet-4-6';
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

const FILTER_SYSTEM_PROMPT = `You are a memory curator. Your job is to decide whether a single WhatsApp message is worth remembering long-term, as part of building a personal memory layer about the people in someone's life.

You may be given recent messages from the same chat as conversational context. Use that context to understand what the current message means (resolving idioms, references, ongoing topics). The decision applies ONLY to the current message — not to the context.

KEEP messages that contain:
- Personal facts about a person (job, family, location, health, preferences, opinions)
- ANY mention of family members, friends, partners, or recurring people in the contact's life — even passing references like "say hi to your mom" or "I'm with my grandma" count as relationship facts worth knowing
- Personality traits or self-descriptions, even if said casually or jokingly ("I'm always late", "I complain a lot", "I'm bad at remembering names")
- Scheduled events with dates or times (appointments, trips, birthdays, plans)
- Commitments (a promise — to act, to deliver, to meet)
- Significant emotional state changes (good news, bad news, milestones)
- Specific concrete information that could be referenced later (names, places, organizations, preferences)

DROP messages that are:
- Pure greetings without context ("hi", "hey", "ciao", "good morning")
- Pure acknowledgments ("ok", "thanks", "got it")
- Pure reactions ("lol", "haha", "nice", emoji-only, single-word excitement)
- Ephemeral logistics ("on my way", "5 min late", "where are you", "one sec")
- Generic small talk, idioms, or jokes with no informational content (use the context to recognize idioms — e.g., a Venetian dialect phrase used colloquially is NOT a personal fact)

Lean toward KEEP. False positives are easier to clean later than false negatives are to recover. Drop only when the message clearly contains nothing durable — when in doubt, keep.

Return your decision strictly as JSON matching the provided schema. The reason field must be one short sentence under 15 words.`;

const EXTRACT_SYSTEM_PROMPT = `You are extracting durable, structured facts about people from WhatsApp messages, to build a long-term memory.

You may be given recent messages from the same chat as conversational context. Use the context to disambiguate references, idioms, and ongoing topics in the current message. CRITICAL: Only extract facts FROM the current message — never extract facts that appear only in the context. The context is read-only background.

For each fact you extract, classify it:
- preference: things they like/dislike, opinions, tastes
- event: scheduled or happened events with dates ("doctor appt next Tuesday", "got promoted last week")
- commitment: a promise (you to them, them to you, them to a third party)
- fact: stable info (job, family member name, address, hometown)
- relationship: who-knows-who connections

Rules:
- Ground every fact in the literal text of the current message. If you can't quote the supporting words, don't extract the fact.
- Resolve relative dates ("tomorrow", "next week") using the message date. If the message just says "tomorrow" without indicating what's happening, do NOT extract a fact — the date alone is not the fact.
- Treat colloquialisms, idioms, sarcasm, and dialect phrases as expressive language, not literal facts. Use context to recognize them.
- SUBJECT ATTRIBUTION (critical): the message is labelled with its SENDER. Use the sender to resolve first-person and possessives:
  • First-person ("io", "I", "sono", "I'm", "ho", "I have") → subject is the SENDER. So "io sono stanco" sent by me → subject is "me"; sent by the contact → subject is the contact.
  • First-person possessives ("my mom", "i miei", "mio padre", "mia sorella", "my X") refer to the SENDER's relations. "i miei" sent by me means MY parents; "i miei" sent by the contact means the CONTACT's parents.
  • Second-person ("tu", "you", "sei") → subject is the addressee (the contact in a DM if I'm sending; me if the contact is sending).
  • Explicitly named subject in the current message → use that name.
  • Recent context unambiguously establishes the third-person subject → use that.
  Italian, Spanish, and many other languages drop subjects ("deve partire", "se va"). A third-person verb with NO named subject and NO clear contextual antecedent is ambiguous — DO NOT extract the fact. Skip it. Never default to the contact.
- Confidence: 0.0–1.0. Use values below 0.7 when the fact requires inference. Skip facts you'd rate below 0.4.
- Each fact should be self-contained (readable without the original message).
- Return an empty list if the message contains no durable, grounded facts.

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
  async filter(input: FilterInput) {
    const result = await callAgent({
      systemPrompt: FILTER_SYSTEM_PROMPT,
      userPrompt: formatFilterMessage(input),
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

  async extract(input: ExtractInput) {
    const result = await callAgent({
      systemPrompt: EXTRACT_SYSTEM_PROMPT,
      userPrompt: formatExtractMessage(input),
      model: EXTRACT_MODEL,
      outputSchema: EXTRACT_SCHEMA_JSON as unknown as Record<string, unknown>,
    });
    const parsed = ExtractZod.parse(result.structured);
    return {
      facts: parsed.facts,
      usage: usageOf(EXTRACT_MODEL, result),
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

function formatFilterMessage(input: FilterInput): string {
  const sender = input.senderName ?? input.contactName ?? 'unknown';
  const lines = [
    `Contact: ${input.contactName ?? 'unknown'}${input.contactNotes ? ` — ${input.contactNotes}` : ''}`,
  ];
  if (input.isGroup) lines.push('Group chat');
  if (input.recentContext && input.recentContext.length > 0) {
    lines.push('', 'Recent conversation:');
    for (const line of input.recentContext) lines.push(`  ${line}`);
  }
  lines.push('', `Current message (sent by ${sender}): "${input.body}"`);
  lines.push('', 'Should this be remembered?');
  return lines.join('\n');
}

function formatExtractMessage(input: ExtractInput): string {
  const dateISO = new Date(input.ts * 1000).toISOString().slice(0, 10);
  const sender = input.senderName ?? input.contactName ?? 'unknown';
  const lines = [
    `Date: ${dateISO}`,
    `Contact: ${input.contactName ?? 'unknown'}${input.contactNotes ? ` — ${input.contactNotes}` : ''}`,
  ];
  if (input.isGroup) lines.push('Group chat');
  if (input.recentContext && input.recentContext.length > 0) {
    lines.push('', 'Recent conversation:');
    for (const line of input.recentContext) lines.push(`  ${line}`);
  }
  lines.push('', `Current message (sent by ${sender}): "${input.body}"`);
  lines.push('', 'Extract facts from the current message.');
  return lines.join('\n');
}

function usageOf(model: string, result: AgentCallResult): UsageMetadata {
  return {
    model,
    tokens_in: result.tokens_in,
    tokens_out: result.tokens_out,
  };
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
