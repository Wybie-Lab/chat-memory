export interface TestTurn {
  /** ISO 8601 timestamp. */
  ts: string;
  /** "in" = the contact, "out" = the user. */
  direction: 'in' | 'out';
  body: string;
}

export interface ExpectedFact {
  subject: string;
  /** Optional category constraint; if omitted, any category counts. */
  category?: 'preference' | 'event' | 'commitment' | 'fact' | 'relationship';
  /**
   * Substring(s) that must appear in the extracted fact's content. All must
   * match (AND). Case-insensitive.
   */
  content_contains: string[];
}

export interface TestQuery {
  q: string;
  /** Substring(s) the model's answer should include. Case-insensitive. */
  expected_answer_substrings?: string[];
  /**
   * Subjects that should be entity-matched in the retrieval context — proves
   * the entity matcher fires. Optional.
   */
  expected_matched_subjects?: string[];
  /**
   * Whether to score retrieval relevance via LLM-as-judge (costs one Sonnet
   * call per query). Default true.
   */
  judge_retrieval?: boolean;
}

export interface TestCase {
  id: string;
  description?: string;
  /** Display name for the contact in this conversation. */
  contact_name: string;
  /** wa_id-style id for the contact. */
  contact_wa_id: string;
  /** The user's display label, used as the subject for "out" first-person facts. */
  me?: string;
  transcript: TestTurn[];
  expected_facts?: ExpectedFact[];
  queries?: TestQuery[];
}

export interface FactRecallResult {
  expected: number;
  matched: number;
  missing: ExpectedFact[];
}

export interface FactPrecisionResult {
  total_extracted: number;
  /**
   * Facts that don't loosely correspond to anything in the expected set.
   * Loose because real bursts often surface valid extras the test didn't list.
   */
  unanchored: number;
}

export interface QueryResult {
  q: string;
  matched_subjects: string[];
  retrieval_relevance_score?: number;
  retrieval_judge_reasoning?: string;
  answer: string;
  answer_substring_hits: number;
  answer_substring_total: number;
  matched_subjects_ok: boolean;
  citations_count: number;
}

export interface RunResult {
  test_id: string;
  description?: string;
  ingested_turns: number;
  bursts_processed: number;
  fact_recall: FactRecallResult;
  fact_precision: FactPrecisionResult;
  queries: QueryResult[];
  ingestion_ms: number;
  total_ms: number;
  errors: string[];
}
