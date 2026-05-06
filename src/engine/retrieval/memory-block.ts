import {
  activePreferences,
  recentEpisodes,
  listClusterSummariesForSubjects,
  listMemoryThreads,
  listFactsInThread,
  listFactThreads,
  listConnectionsToFact,
  latestInChain,
  getActiveFactById,
  type ActiveFactRow,
  type ClusterSummaryRow,
  type FactRow,
  type MemoryThreadRow,
  type SubjectInfo,
  type DB,
} from '../storage/db';
import type { LLMProvider, UsageMetadata } from '../../llm/provider';
import {
  buildRetrievalContext,
  hybridRetrieve,
  type ScoredFact,
} from './score';

/**
 * Approximate chars per token for our content mix (Italian + English, plain
 * prose). Used only to bound the memory block — it doesn't have to be exact.
 */
const CHARS_PER_TOKEN = 3.5;

const TOTAL_TOKEN_BUDGET = 1500;
const SECTION_TOKEN_BUDGET = {
  preferences: 300,
  top_facts: 600,
  threads: 300,
  cluster_summaries: 400,
  recent_episodes: 200,
};

const RECENT_EPISODE_DAYS = 60;

export interface ThreadSection {
  thread: MemoryThreadRow;
  facts: ActiveFactRow[];
}

export interface ComposedMemoryBlock {
  block: string;
  citations: ScoredFact[];
  preferences: ActiveFactRow[];
  cluster_summaries: ClusterSummaryRow[];
  episodes: ActiveFactRow[];
  threads: ThreadSection[];
  matched_subjects: string[];
  embed_usage: UsageMetadata;
  budget: {
    total_chars: number;
    sections: Record<keyof typeof SECTION_TOKEN_BUDGET, { used_chars: number; dropped: number }>;
  };
}

export interface ComposeOptions {
  /** Topic or empty if none — purely for the section header. */
  preferenceSubject?: string;
  /** Override for tests; otherwise read from DEFAULT. */
  totalTokenBudget?: number;
}

/**
 * Build the structured <memory> block injected into the chat agent's context
 * for one user question. Combines always-on preferences, hybrid-ranked facts
 * (chain-walked + retract-filtered), threads matched to the query, cluster
 * summaries, and recent episodes — all inside a hard token cap.
 */
export async function composeMemoryBlock(
  db: DB,
  provider: LLMProvider,
  question: string,
  opts: ComposeOptions = {}
): Promise<ComposedMemoryBlock> {
  const totalBudget = opts.totalTokenBudget ?? TOTAL_TOKEN_BUDGET;
  const totalCharsBudget = Math.floor(totalBudget * CHARS_PER_TOKEN);

  const embedRes = await provider.embed(question, 'query');

  const ctx = buildRetrievalContext(db, question);
  const subjectLabels = buildSubjectLabelMap(ctx.subjects);

  const rawScored = hybridRetrieve(db, embedRes.vector, ctx, { k: 30, rerankTopK: 30 });
  // Prefer the leaf of any update/state_change chain; drop retracted; dedupe.
  const topScored = walkChainsAndDedupe(db, rawScored).slice(0, 20);

  const allPrefs = activePreferences(db, opts.preferenceSubject);

  // Subjects we care about for the question: explicit query mentions ∪ subjects
  // of top-ranked facts.
  const subjectsForClusters = new Set<string>(ctx.matchedSubjectIds);
  for (const f of topScored.slice(0, 5)) subjectsForClusters.add(f.subject_wa_id);
  const clusters = listClusterSummariesForSubjects(db, [...subjectsForClusters]);

  // Threads: union of two cheap signals.
  //  (a) the query string substring-matches the thread name (explicit topic)
  //  (b) one of the top-ranked facts is in the thread (implicit relevance)
  // Either is enough — semantic match on thread names is dominated by (b)
  // here because we already paid a vector embedding to rank facts.
  const threads = matchedThreadsForQuery(db, question, subjectsForClusters, topScored);

  const allEpisodes = recentEpisodes(db, RECENT_EPISODE_DAYS).filter((ep) =>
    subjectsForClusters.size === 0 ? true : subjectsForClusters.has(ep.subject_wa_id)
  );

  const prefsSection = packLines(
    allPrefs,
    Math.floor(SECTION_TOKEN_BUDGET.preferences * CHARS_PER_TOKEN),
    (p) => formatPreference(p, subjectLabels)
  );
  const topFactsSection = packLines(
    topScored,
    Math.floor(SECTION_TOKEN_BUDGET.top_facts * CHARS_PER_TOKEN),
    (f) => formatScoredFact(f, subjectLabels)
  );
  const threadsSection = packThreads(
    threads,
    Math.floor(SECTION_TOKEN_BUDGET.threads * CHARS_PER_TOKEN),
    subjectLabels
  );
  const clustersSection = packLines(
    clusters,
    Math.floor(SECTION_TOKEN_BUDGET.cluster_summaries * CHARS_PER_TOKEN),
    (c) => formatCluster(c, subjectLabels)
  );
  const episodesSection = packLines(
    allEpisodes,
    Math.floor(SECTION_TOKEN_BUDGET.recent_episodes * CHARS_PER_TOKEN),
    (e) => formatEpisode(e, subjectLabels)
  );

  const sectionLines: SectionLines = {
    preferences: prefsSection.lines,
    top_facts: topFactsSection.lines,
    threads: threadsSection.lines,
    cluster_summaries: clustersSection.lines,
    recent_episodes: episodesSection.lines,
  };
  const block = renderBlock(sectionLines);

  let finalBlock = block;
  if (block.length > totalCharsBudget) {
    finalBlock = enforceGlobalBudget(sectionLines, totalCharsBudget);
  }

  return {
    block: finalBlock,
    citations: topScored.slice(0, topFactsSection.lines.length),
    preferences: allPrefs.slice(0, prefsSection.lines.length),
    cluster_summaries: clusters.slice(0, clustersSection.lines.length),
    episodes: allEpisodes.slice(0, episodesSection.lines.length),
    threads: threadsSection.kept,
    matched_subjects: [...ctx.matchedSubjectIds],
    embed_usage: embedRes.usage,
    budget: {
      total_chars: finalBlock.length,
      sections: {
        preferences: { used_chars: prefsSection.usedChars, dropped: prefsSection.dropped },
        top_facts: { used_chars: topFactsSection.usedChars, dropped: topFactsSection.dropped },
        threads: { used_chars: threadsSection.usedChars, dropped: threadsSection.dropped },
        cluster_summaries: {
          used_chars: clustersSection.usedChars,
          dropped: clustersSection.dropped,
        },
        recent_episodes: {
          used_chars: episodesSection.usedChars,
          dropped: episodesSection.dropped,
        },
      },
    },
  };
}

/**
 * Replace each scored candidate with the leaf of its update/state_change
 * chain, drop any whose leaf has been retracted, and dedupe. Order is
 * preserved by score; the leaf inherits the candidate's score (the chain
 * member that matched the query is a valid relevance signal for the leaf).
 */
function walkChainsAndDedupe(db: DB, scored: ScoredFact[]): ScoredFact[] {
  const seen = new Set<number>();
  const out: ScoredFact[] = [];
  for (const c of scored) {
    const leafId = latestInChain(db, c.id);
    if (seen.has(leafId)) continue;
    if (listConnectionsToFact(db, leafId, 'retracts').length > 0) continue;
    if (leafId === c.id) {
      seen.add(leafId);
      out.push(c);
      continue;
    }
    const leaf = getActiveFactById(db, leafId);
    if (!leaf) continue;
    seen.add(leafId);
    out.push({
      ...leaf,
      distance: c.distance,
      score: c.score,
      components: c.components,
      matched_subject_ids: c.matched_subject_ids,
    });
  }
  return out;
}

/**
 * Pick threads worth surfacing for a question. Two signals, unioned:
 *   - explicit: a thread name token (≥3 chars) appears in the question
 *   - implicit: one of the top-N ranked facts is a member of the thread
 * Threads owned by an unmatched subject (and not surfaced by a top fact) are
 * skipped to keep the section on-topic.
 */
function matchedThreadsForQuery(
  db: DB,
  question: string,
  subjects: Set<string>,
  topScored: ScoredFact[]
): ThreadSection[] {
  const lowerQ = question.toLowerCase();
  const out: ThreadSection[] = [];
  const seen = new Set<number>();

  // Implicit: walk the top scored facts, collect their thread memberships.
  // Cap how deep we go so we don't pull in every tangentially-related thread.
  const TOP_FACTS_FOR_THREAD_SURFACE = 8;
  for (const f of topScored.slice(0, TOP_FACTS_FOR_THREAD_SURFACE)) {
    for (const t of listFactThreads(db, f.id)) {
      if (seen.has(t.id)) continue;
      // Only include threads whose owner is a query-matched subject, OR
      // cross-subject threads (owner null). Keeps the section focused.
      if (t.owner_subject_wa_id !== null && !subjects.has(t.owner_subject_wa_id)) continue;
      const facts = listFactsInThread(db, t.id, 8);
      if (facts.length === 0) continue;
      seen.add(t.id);
      out.push({ thread: t, facts });
    }
  }

  // Explicit: substring match on thread names of matched subjects.
  for (const subject of subjects) {
    const threads = listMemoryThreads(db, { owner_subject_wa_id: subject });
    for (const t of threads) {
      if (seen.has(t.id)) continue;
      const tokens = t.name.toLowerCase().split(/[\s\-,()'’]+/).filter((s) => s.length >= 3);
      if (!tokens.some((tok) => lowerQ.includes(tok))) continue;
      const facts = listFactsInThread(db, t.id, 8);
      if (facts.length === 0) continue;
      seen.add(t.id);
      out.push({ thread: t, facts });
    }
  }
  return out;
}

interface PackedSection {
  lines: string[];
  usedChars: number;
  dropped: number;
}

function packLines<T>(items: T[], charBudget: number, format: (t: T) => string): PackedSection {
  const lines: string[] = [];
  let used = 0;
  let i = 0;
  for (; i < items.length; i++) {
    const line = format(items[i]);
    if (used + line.length + 1 > charBudget) break;
    lines.push(line);
    used += line.length + 1;
  }
  return { lines, usedChars: used, dropped: items.length - i };
}

interface PackedThreadSection extends PackedSection {
  kept: ThreadSection[];
}

/**
 * Render threads as a header line per thread followed by indented fact lines.
 * Greedy fit: include whole threads while they fit; truncate per-thread fact
 * lists if needed; drop excess threads at the end.
 */
function packThreads(
  threads: ThreadSection[],
  charBudget: number,
  subjectLabels: Map<string, string>
): PackedThreadSection {
  const lines: string[] = [];
  const kept: ThreadSection[] = [];
  let used = 0;
  let dropped = 0;
  for (const t of threads) {
    const header = formatThreadHeader(t.thread, subjectLabels);
    let chunkLines: string[] = [header];
    let chunkChars = header.length + 1;
    const keptFacts: ActiveFactRow[] = [];
    for (const f of t.facts) {
      const line = '  ' + formatActiveFact(f, subjectLabels);
      if (chunkChars + line.length + 1 > charBudget - used && chunkLines.length > 1) break;
      chunkLines.push(line);
      chunkChars += line.length + 1;
      keptFacts.push(f);
    }
    if (used + chunkChars > charBudget) {
      dropped++;
      continue;
    }
    lines.push(...chunkLines);
    used += chunkChars;
    kept.push({ thread: t.thread, facts: keptFacts });
  }
  dropped += threads.length - kept.length;
  return { lines, usedChars: used, dropped, kept };
}

function buildSubjectLabelMap(subjects: SubjectInfo[]): Map<string, string> {
  const m = new Map<string, string>();
  for (const s of subjects) {
    m.set(s.subject_wa_id, displayLabel(s.subject_wa_id, s.display_name));
  }
  return m;
}

/**
 * Pick a human-readable label for a subject:
 *   - prefer the contact's display_name
 *   - fall back to the wa_id stripped of its "@c.us" / "@g.us" suffix
 *   - "me" stays "me"
 */
function displayLabel(waId: string, displayName: string | null): string {
  if (displayName && displayName.trim().length > 0) return displayName.trim();
  const at = waId.indexOf('@');
  return at > 0 ? waId.slice(0, at) : waId;
}

function labelFor(waId: string, labels: Map<string, string>): string {
  return labels.get(waId) ?? displayLabel(waId, null);
}

function formatPreference(p: ActiveFactRow, labels: Map<string, string>): string {
  const subjectLabel = labelFor(p.subject_wa_id, labels);
  return `- (${subjectLabel}) ${p.content} [fact:${p.id}]`;
}

function formatScoredFact(f: ScoredFact, labels: Map<string, string>): string {
  return `- (${labelFor(f.subject_wa_id, labels)}) ${f.content} [fact:${f.id}]`;
}

function formatActiveFact(f: ActiveFactRow, labels: Map<string, string>): string {
  return `- (${labelFor(f.subject_wa_id, labels)}) ${f.content} [fact:${f.id}]`;
}

function formatCluster(c: ClusterSummaryRow, labels: Map<string, string>): string {
  return `- ${labelFor(c.subject_wa_id, labels)} (${c.category}): ${c.summary}`;
}

function formatEpisode(e: ActiveFactRow, labels: Map<string, string>): string {
  // Anchor on event_ts when set (the actual event date) — otherwise on
  // extracted_at (when we learned about it). The "→" marker flags upcoming
  // items so the chat agent can answer "what's planned" vs "what happened"
  // without parsing dates from prose.
  const anchorTs = e.event_ts ?? e.extracted_at;
  const date = new Date(anchorTs * 1000).toISOString().slice(0, 10);
  const isFuture = e.event_ts !== null && e.event_ts > Date.now() / 1000;
  const marker = isFuture ? '→ ' : '';
  return `- ${marker}${date} (${labelFor(e.subject_wa_id, labels)}, ${e.category}): ${e.content} [fact:${e.id}]`;
}

function formatThreadHeader(t: MemoryThreadRow, labels: Map<string, string>): string {
  const owner = t.owner_subject_wa_id ? labelFor(t.owner_subject_wa_id, labels) : 'shared';
  const desc = t.description ? ` — ${t.description}` : '';
  return `- ${owner}: ${t.name}${desc}`;
}

interface SectionLines {
  preferences: string[];
  top_facts: string[];
  threads: string[];
  cluster_summaries: string[];
  recent_episodes: string[];
}

function renderBlock(s: SectionLines): string {
  const parts: string[] = ['<memory>'];
  if (s.preferences.length > 0) {
    parts.push('<preferences>');
    parts.push(...s.preferences);
    parts.push('</preferences>');
  }
  if (s.top_facts.length > 0) {
    parts.push('<known_facts>');
    parts.push(...s.top_facts);
    parts.push('</known_facts>');
  }
  if (s.threads.length > 0) {
    parts.push('<threads>');
    parts.push(...s.threads);
    parts.push('</threads>');
  }
  if (s.cluster_summaries.length > 0) {
    parts.push('<subject_summaries>');
    parts.push(...s.cluster_summaries);
    parts.push('</subject_summaries>');
  }
  if (s.recent_episodes.length > 0) {
    parts.push('<recent_episodes>');
    parts.push(...s.recent_episodes);
    parts.push('</recent_episodes>');
  }
  parts.push('</memory>');
  return parts.join('\n');
}

/**
 * Last-resort trim: drop sections in priority order until under the global
 * char budget. Priority (kept longest): preferences > top_facts > threads >
 * cluster_summaries > recent_episodes.
 */
function enforceGlobalBudget(s: SectionLines, charBudget: number): string {
  const dropOrder: Array<keyof SectionLines> = [
    'recent_episodes',
    'cluster_summaries',
    'threads',
    'top_facts',
    'preferences',
  ];
  const working: SectionLines = { ...s };
  for (const section of dropOrder) {
    while (working[section].length > 0) {
      const block = renderBlock(working);
      if (block.length <= charBudget) return block;
      working[section] = working[section].slice(0, -1);
    }
  }
  return renderBlock(working);
}

// Internal-only — exported for tests / future debugging.
export type { FactRow };
