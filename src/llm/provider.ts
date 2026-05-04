export interface BurstLine {
  direction: 'in' | 'out';
  body: string;
  ts: number;
}

export interface BurstInput {
  contactName?: string | null;
  contactNotes?: string | null;
  isGroup?: boolean;
  startTs: number;
  endTs: number;
  lines: BurstLine[];
}

export interface FilterResult {
  keep: boolean;
  reason: string;
}

export type FactCategory = 'preference' | 'event' | 'commitment' | 'fact' | 'relationship';

export interface ExtractedFact {
  subject: string;
  category: FactCategory;
  content: string;
  confidence: number;
  /**
   * Unix seconds. Set ONLY for event/commitment when the burst contains an
   * unambiguous specific date for the event itself. Null/undefined for
   * non-events, or when the date is ambiguous, partial, or unclear.
   * The extractor must be conservative: a missing event_ts is much better
   * than a wrong one.
   */
  event_ts?: number | null;
}

export interface UsageMetadata {
  model: string;
  tokens_in: number;
  tokens_out: number;
}

export interface ChatInput {
  question: string;
  memoryBlock: string;
}

export type EmbedMode = 'document' | 'query';

export interface ExistingFact {
  id: number;
  content: string;
  category: string;
  confidence: number;
  age_days: number;
}

export interface ConsolidateInput {
  subject: string;
  existing: ExistingFact[];
  candidates: ExtractedFact[];
}

export interface SummarizeClusterInput {
  subject: string;
  contactDisplayName?: string | null;
  category: FactCategory;
  facts: Array<{
    id: number;
    content: string;
    confidence: number;
    age_days: number;
  }>;
}

export const ENTITY_TYPES = [
  'person',
  'place',
  'organization',
  'event',
  'preference_topic',
  'object',
  'concept',
  'date',
] as const;

export type EntityType = (typeof ENTITY_TYPES)[number];

export const ENTITY_ROLES = [
  'subject',
  'object',
  'person',
  'place',
  'organization',
  'event',
  'date',
  'topic',
  'source',
  'recipient',
] as const;

export type EntityRole = (typeof ENTITY_ROLES)[number];

export const GRAPH_PREDICATES = [
  'knows',
  'friend_of',
  'family_of',
  'partner_of',
  'works_at',
  'studies_at',
  'lives_in',
  'from_place',
  'located_in',
  'likes',
  'dislikes',
  'interested_in',
  'attending',
  'planning',
  'visited',
  'promised_to',
  'needs',
  'owns',
  'part_of',
  'mentioned',
] as const;

export type GraphPredicate = (typeof GRAPH_PREDICATES)[number];

export interface ExtractedGraph {
  entities: Array<{
    local_id: string;
    type: EntityType;
    display_name: string;
    aliases?: string[];
    confidence: number;
  }>;
  mentions: Array<{
    entity_local_id: string;
    role: EntityRole;
    mention_text?: string;
    confidence: number;
  }>;
  edges: Array<{
    source_local_id: string;
    predicate: GraphPredicate;
    target_local_id: string;
    confidence: number;
    event_ts?: number | null;
    valid_from_ts?: number | null;
    valid_to_ts?: number | null;
    qualifiers?: Record<string, unknown>;
  }>;
}

export interface GraphFactInput {
  fact_id: number;
  subject: string;
  category: FactCategory;
  content: string;
  confidence: number;
  event_ts?: number | null;
}

export type ConsolidationOp =
  | { op: 'ADD'; candidate_index: number; content: string; category: FactCategory; confidence: number }
  | { op: 'UPDATE'; candidate_index: number; old_fact_id: number; content: string; category: FactCategory; confidence: number }
  | { op: 'DELETE'; old_fact_id: number; reason: string }
  | { op: 'DROP'; candidate_index: number; reason: string };

// ───────────── Curator agent step ─────────────
// One round-trip in the agent loop. The curator builds the prompt + tool
// catalog from engine state; the provider produces the next batch of tool
// calls. Tool dispatch and bookkeeping live in the engine, not here — the
// provider only knows how to ask the model "what next?".

export interface AgentToolDefinition {
  name: string;
  description: string;
  /** JSON schema for the tool's arguments object. */
  parameters: Record<string, unknown>;
}

export interface AgentToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface AgentHistoryEntry {
  role: 'assistant' | 'tool';
  /** Free-form text. Assistant entries echo the previous tool_calls + thinking;
   *  tool entries are JSON-serialized results keyed by the call index. */
  content: string;
}

export interface AgentStepInput {
  systemPrompt: string;
  userPrompt: string;
  tools: AgentToolDefinition[];
  history: AgentHistoryEntry[];
}

export interface AgentStepOutput {
  thinking?: string;
  tool_calls: AgentToolCall[];
}

export interface LLMProvider {
  filterBurst(input: BurstInput): Promise<FilterResult & { usage: UsageMetadata }>;
  extractBurst(input: BurstInput): Promise<{ facts: ExtractedFact[]; usage: UsageMetadata }>;
  consolidate(
    input: ConsolidateInput
  ): Promise<{ ops: ConsolidationOp[]; usage: UsageMetadata }>;
  summarizeCluster(
    input: SummarizeClusterInput
  ): Promise<{ summary: string; usage: UsageMetadata }>;
  extractGraphFromFact(
    input: GraphFactInput
  ): Promise<{ graph: ExtractedGraph; usage: UsageMetadata }>;
  embed(text: string, mode?: EmbedMode): Promise<{ vector: number[]; usage: UsageMetadata }>;
  chat(input: ChatInput): Promise<{ answer: string; usage: UsageMetadata }>;
  agentStep(
    input: AgentStepInput
  ): Promise<{ output: AgentStepOutput; usage: UsageMetadata }>;
}
