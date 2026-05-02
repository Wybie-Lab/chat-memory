export interface FilterInput {
  body: string;
  contactName?: string | null;
  contactNotes?: string | null;
  isGroup?: boolean;
  senderName?: string | null;
  recentContext?: string[];
}

export interface FilterResult {
  keep: boolean;
  reason: string;
}

export interface ExtractInput {
  body: string;
  contactName?: string | null;
  contactNotes?: string | null;
  ts: number;
  isGroup?: boolean;
  senderName?: string | null;
  recentContext?: string[];
}

export type FactCategory = 'preference' | 'event' | 'commitment' | 'fact' | 'relationship';

export interface ExtractedFact {
  subject: string;
  category: FactCategory;
  content: string;
  confidence: number;
}

export interface UsageMetadata {
  model: string;
  tokens_in: number;
  tokens_out: number;
}

export interface LLMProvider {
  filter(input: FilterInput): Promise<FilterResult & { usage: UsageMetadata }>;
  extract(input: ExtractInput): Promise<{ facts: ExtractedFact[]; usage: UsageMetadata }>;
  embed(text: string): Promise<{ vector: number[]; usage: UsageMetadata }>;
}
