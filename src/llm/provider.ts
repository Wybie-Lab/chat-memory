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

/**
 * The append-only memory model uses three consolidation ops:
 *
 *  - ADD     — candidate is a new fact, unrelated to anything existing.
 *              Insert the fact. No connection.
 *  - CONNECT — candidate is a new fact that relates to an existing one
 *              via a typed connection. Insert the fact AND insert a
 *              fact_connections row from the new fact to old_fact_id.
 *              Predicate captures the kind of relationship:
 *                update       — same thing, new state
 *                state_change — discrete event changing state
 *                expands      — adds detail / specificity
 *                qualifies    — adds a condition or nuance
 *                contradicts  — mutual exclusion (unresolved)
 *                retracts     — old fact was wrong
 *                same_as      — restating; mostly used by curator dedupe
 *  - DROP    — candidate adds nothing; ignore it. Memory unchanged.
 *
 * Note: there is intentionally no DELETE op anymore. The append-only model
 * preserves history; corrections happen via CONNECT(predicate=retracts)
 * with a new replacement fact.
 */
export type ConnectionPredicate =
  | 'update'
  | 'state_change'
  | 'expands'
  | 'qualifies'
  | 'contradicts'
  | 'retracts'
  | 'same_as';

export type ConsolidationOp =
  | { op: 'ADD'; candidate_index: number; content: string; category: FactCategory; confidence: number }
  | {
      op: 'CONNECT';
      candidate_index: number;
      old_fact_id: number;
      predicate: ConnectionPredicate;
      content: string;
      category: FactCategory;
      confidence: number;
      reason: string;
    }
  | { op: 'DROP'; candidate_index: number; reason: string };

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
}
