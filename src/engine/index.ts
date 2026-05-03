/**
 * Public API for the memory engine.
 *
 * Boundary rule: this is the ONLY path consumers (sources/, web/, cli/, eval/,
 * scripts/) should import from. Reaching directly into engine/storage/db.ts,
 * engine/pipeline.ts, engine/retrieval/* from outside src/engine/ is a
 * code-review smell — if a consumer needs something not exported here, add
 * it to this barrel rather than opening the boundary.
 *
 * LLM types (LLMProvider, ExtractedFact, etc.) deliberately do NOT come from
 * this module — they live in src/llm/provider.ts because they're the provider
 * contract the engine depends on, not engine-owned types.
 */

// ───────────── lifecycle ─────────────
export { openDb } from './storage/db';

// ───────────── ingestion (sources call these) ─────────────
// Sources (whatsapp live, chat-export importer) write raw_messages and
// assign them to bursts. The engine then picks up the bursts.
export {
  upsertContact,
  insertRawMessage,
  assignMessageToBurst,
  rebuildAllBursts,
  setLiveCutoff,
  getLiveCutoff,
} from './storage/db';

// ───────────── pipeline (consumers drain bursts) ─────────────
export { processBatch, processUntilDrained } from './pipeline';

// ───────────── retrieval (consumers compose memory for a query) ─────────────
export { composeMemoryBlock } from './retrieval/memory-block';
export {
  hybridRetrieve,
  buildRetrievalContext,
  scoreCandidates,
  matchSubjectsInQuery,
  DEFAULT_WEIGHTS,
} from './retrieval/score';

// ───────────── inspection / browsing (used by web UI + eval scoring) ─────────────
export {
  listFacts,
  listCategories,
  searchFactsByVector,
  searchFactsForSubjectByVector,
  allActiveSubjects,
  factsAboutSubject,
  listClusterSummariesForSubject,
  listClusterSummariesForSubjects,
  getClusterSummary,
  activePreferences,
  recentEpisodes,
  countFactSources,
  getFactSources,
  attachFactSource,
  activeFactsForCluster,
  existingFactsForSubject,
  countActiveFactsForSubject,
  insertFact,
  insertEmbedding,
  markFactSuperseded,
  markFactDeleted,
  upsertClusterSummary,
  deleteClusterSummary,
  logProcessing,
  listUnprocessedBursts,
  getBurstMessages,
  markBurstFiltered,
  markBurstProcessed,
} from './storage/db';

// ───────────── types ─────────────
export type {
  DB,
  RawMessageInput,
  FactInput,
  FactRow,
  FactSearchResult,
  FactSourceRow,
  FactListFilters,
  ActiveFactRow,
  ClusterSummaryRow,
  SubjectInfo,
  UnprocessedBurst,
  BurstMessage,
  BurstRow,
  ProcessingLogInput,
} from './storage/db';

export { BURST_GAP_SECONDS } from './storage/db';

export type { ProcessStats, ProcessOptions } from './pipeline';

export type {
  ScoredFact,
  ScoreWeights,
  ScoreOptions,
  RetrievalContext,
} from './retrieval/score';

export type {
  ComposedMemoryBlock,
  ComposeOptions,
} from './retrieval/memory-block';
