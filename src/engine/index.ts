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

// ───────────── knowledge graph (derived projection over facts) ─────────────
export {
  writeExtractedGraph,
  canonicalKeyForEntity,
  type GraphFactContext,
  type GraphWriteResult,
} from './graph';

// ───────────── curator agent (scoped LLM curation) ─────────────
// The agent reads existing memory and writes proposed mutations to
// agent_actions. The apply path turns those proposals into real fact
// mutations (insertFact + markFactSuperseded / markFactDeleted), refreshes
// affected cluster summaries, and logs evidence trails via fact_sources.
export {
  planAgentRun,
  runCurator,
  drainPlannedAgentRuns,
  planTriggeredRunsForFact,
  type RunCuratorResult,
  type DrainStats,
  type DrainOptions,
} from './agent/curator';
export {
  applyAgentRun,
  applyAgentAction,
  rejectAgentAction,
  type ApplyAgentRunOptions,
  type ApplyAgentRunResult,
  type ApplyAgentActionOptions,
  type ApplyAgentActionResult,
} from './agent/apply';
export {
  getAgentRun,
  getAgentAction,
  listAgentRuns,
  listAgentActionsForRun,
  countAgentActionsForRun,
  type AgentRunInput,
  type AgentRunRow,
  type AgentActionInput,
  type AgentActionRow,
  type AgentTrigger,
  type AgentScopeType,
  type AgentRunStatus,
  type AgentActionOp,
  type AgentActionStatus,
  type ListAgentRunsFilter,
} from './storage/db';

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
  getSubjectInfo,
  supersededFactsForSubject,
  factsAboutSubject,
  factsAboutSubjectWithSource,
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
  insertAgentDerivedFact,
  insertEmbedding,
  markFactSuperseded,
  markFactDeleted,
  upsertClusterSummary,
  deleteClusterSummary,
  clearGraphTables,
  listActiveFactsForGraph,
  upsertEntity,
  insertFactEntityMention,
  upsertKnowledgeEdge,
  attachEdgeSource,
  deactivateGraphForFact,
  graphCounts,
  searchEntities,
  graphNeighborhood,
  listGraphEntitiesWithStats,
  listKnowledgeEdges,
  graphForFact,
  logProcessing,
  listUnprocessedBursts,
  getBurstMessages,
  getBurstQueueStats,
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
  GraphEntityInput,
  GraphEntityRow,
  FactEntityMentionInput,
  FactEntityMentionRow,
  KnowledgeEdgeInput,
  KnowledgeEdgeRow,
  GraphFactRow,
  GraphBuildStats,
  GraphEntityWithStatsRow,
  SubjectInfo,
  SupersededFactRow,
  UnprocessedBurst,
  BurstMessage,
  BurstRow,
  BurstQueueStats,
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
