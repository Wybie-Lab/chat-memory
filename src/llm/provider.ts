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

export type ConsolidationOp =
  | { op: 'ADD'; candidate_index: number; content: string; category: FactCategory; confidence: number }
  | { op: 'UPDATE'; candidate_index: number; old_fact_id: number; content: string; category: FactCategory; confidence: number }
  | { op: 'DELETE'; old_fact_id: number; reason: string }
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
  embed(text: string, mode?: EmbedMode): Promise<{ vector: number[]; usage: UsageMetadata }>;
  chat(input: ChatInput): Promise<{ answer: string; usage: UsageMetadata }>;
}
