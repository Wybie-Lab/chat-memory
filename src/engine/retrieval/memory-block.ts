import {
  activePreferences,
  recentEpisodes,
  listClusterSummariesForSubjects,
  factsAboutSubject,
  type ActiveFactRow,
  type ClusterSummaryRow,
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
  cluster_summaries: 400,
  recent_episodes: 200,
};

const RECENT_EPISODE_DAYS = 60;

export interface ComposedMemoryBlock {
  block: string;
  citations: ScoredFact[];
  preferences: ActiveFactRow[];
  cluster_summaries: ClusterSummaryRow[];
  episodes: ActiveFactRow[];
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
 * for one user question. Combines always-on preferences, hybrid-ranked facts,
 * cluster summaries for entity-matched subjects, and recent episodes — all
 * inside a hard token cap.
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

  const topScored = hybridRetrieve(db, embedRes.vector, ctx, { k: 30, rerankTopK: 20 });

  // Always-on preferences. Ordered by confidence DESC inside activePreferences.
  const allPrefs = activePreferences(db, opts.preferenceSubject);

  // Cluster summaries: pull for any subject that appeared either in entity
  // match or in the top scored facts (whichever subjects are most likely
  // relevant to this question).
  const subjectsForClusters = new Set<string>(ctx.matchedSubjectIds);
  for (const f of topScored.slice(0, 5)) subjectsForClusters.add(f.subject_wa_id);
  const clusters = listClusterSummariesForSubjects(db, [...subjectsForClusters]);

  // Episodes (event/commitment) within the trailing window. Filter to subjects
  // that matched or scored highly so the section stays on-topic.
  const allEpisodes = recentEpisodes(db, RECENT_EPISODE_DAYS).filter((ep) =>
    subjectsForClusters.size === 0 ? true : subjectsForClusters.has(ep.subject_wa_id)
  );

  // Pack each section into its own char budget. Items already arrive
  // priority-sorted, so a simple greedy fit is correct.
  const prefsSection = packLines(
    allPrefs,
    Math.floor(SECTION_TOKEN_BUDGET.preferences * CHARS_PER_TOKEN),
    formatPreference
  );
  const topFactsSection = packLines(
    topScored,
    Math.floor(SECTION_TOKEN_BUDGET.top_facts * CHARS_PER_TOKEN),
    formatScoredFact
  );
  const clustersSection = packLines(
    clusters,
    Math.floor(SECTION_TOKEN_BUDGET.cluster_summaries * CHARS_PER_TOKEN),
    formatCluster
  );
  const episodesSection = packLines(
    allEpisodes,
    Math.floor(SECTION_TOKEN_BUDGET.recent_episodes * CHARS_PER_TOKEN),
    formatEpisode
  );

  // Stitch sections, then enforce the global cap by trimming the lowest-priority
  // section first (episodes → clusters → top_facts → preferences).
  const block = renderBlock({
    preferences: prefsSection.lines,
    top_facts: topFactsSection.lines,
    cluster_summaries: clustersSection.lines,
    recent_episodes: episodesSection.lines,
  });

  let finalBlock = block;
  if (block.length > totalCharsBudget) {
    finalBlock = enforceGlobalBudget(
      {
        preferences: prefsSection.lines,
        top_facts: topFactsSection.lines,
        cluster_summaries: clustersSection.lines,
        recent_episodes: episodesSection.lines,
      },
      totalCharsBudget
    );
  }

  return {
    block: finalBlock,
    citations: topScored.slice(0, topFactsSection.lines.length),
    preferences: allPrefs.slice(0, prefsSection.lines.length),
    cluster_summaries: clusters.slice(0, clustersSection.lines.length),
    episodes: allEpisodes.slice(0, episodesSection.lines.length),
    matched_subjects: [...ctx.matchedSubjectIds],
    embed_usage: embedRes.usage,
    budget: {
      total_chars: finalBlock.length,
      sections: {
        preferences: { used_chars: prefsSection.usedChars, dropped: prefsSection.dropped },
        top_facts: { used_chars: topFactsSection.usedChars, dropped: topFactsSection.dropped },
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

function formatPreference(p: ActiveFactRow): string {
  return `- ${p.content} [fact:${p.id}]`;
}

function formatScoredFact(f: ScoredFact): string {
  const subjectLabel = f.subject_wa_id;
  return `- (${subjectLabel}) ${f.content} [fact:${f.id}]`;
}

function formatCluster(c: ClusterSummaryRow): string {
  return `- ${c.subject_wa_id} (${c.category}): ${c.summary}`;
}

function formatEpisode(e: ActiveFactRow): string {
  // Anchor on event_ts when set (the actual event date) — otherwise on
  // extracted_at (when we learned about it). The "→" marker flags upcoming
  // items so the chat agent can answer "what's planned" vs "what happened"
  // without parsing dates from prose.
  const anchorTs = e.event_ts ?? e.extracted_at;
  const date = new Date(anchorTs * 1000).toISOString().slice(0, 10);
  const isFuture = e.event_ts !== null && e.event_ts > Date.now() / 1000;
  const marker = isFuture ? '→ ' : '';
  return `- ${marker}${date} (${e.subject_wa_id}, ${e.category}): ${e.content} [fact:${e.id}]`;
}

interface SectionLines {
  preferences: string[];
  top_facts: string[];
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
 * char budget. Priority (kept longest): preferences > top_facts >
 * cluster_summaries > recent_episodes.
 */
function enforceGlobalBudget(s: SectionLines, charBudget: number): string {
  const dropOrder: Array<keyof SectionLines> = [
    'recent_episodes',
    'cluster_summaries',
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
