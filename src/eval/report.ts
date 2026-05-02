import type { RunResult } from './types';

export interface AggregateMetrics {
  fact_recall_pct: number;
  unanchored_per_test: number;
  avg_retrieval_relevance: number | null;
  answer_substring_hit_rate: number | null;
  matched_subjects_hit_rate: number | null;
  total_tests: number;
  total_errors: number;
}

export function aggregate(results: RunResult[]): AggregateMetrics {
  let recallNum = 0;
  let recallDen = 0;
  let unanchoredSum = 0;
  let judgeSum = 0;
  let judgeCount = 0;
  let subHits = 0;
  let subTotal = 0;
  let matchedHits = 0;
  let matchedTotal = 0;
  let errors = 0;

  for (const r of results) {
    recallNum += r.fact_recall.matched;
    recallDen += r.fact_recall.expected;
    unanchoredSum += r.fact_precision.unanchored;
    errors += r.errors.length;
    for (const q of r.queries) {
      if (typeof q.retrieval_relevance_score === 'number') {
        judgeSum += q.retrieval_relevance_score;
        judgeCount++;
      }
      if (q.answer_substring_total > 0) {
        subHits += q.answer_substring_hits;
        subTotal += q.answer_substring_total;
      }
      matchedHits += q.matched_subjects_ok ? 1 : 0;
      matchedTotal += 1;
    }
  }

  return {
    fact_recall_pct: recallDen === 0 ? 0 : recallNum / recallDen,
    unanchored_per_test: results.length === 0 ? 0 : unanchoredSum / results.length,
    avg_retrieval_relevance: judgeCount === 0 ? null : judgeSum / judgeCount,
    answer_substring_hit_rate: subTotal === 0 ? null : subHits / subTotal,
    matched_subjects_hit_rate: matchedTotal === 0 ? null : matchedHits / matchedTotal,
    total_tests: results.length,
    total_errors: errors,
  };
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function num(n: number | null, digits = 2): string {
  return n === null ? '—' : n.toFixed(digits);
}

export function renderMarkdown(results: RunResult[]): string {
  const agg = aggregate(results);
  const lines: string[] = [];

  lines.push('# manila eval report');
  lines.push('');
  lines.push(`Generated ${new Date().toISOString()} — ${results.length} test(s)`);
  lines.push('');

  lines.push('## Aggregate metrics');
  lines.push('');
  lines.push('| Metric | Value |');
  lines.push('|---|---|');
  lines.push(`| Fact recall | ${pct(agg.fact_recall_pct)} |`);
  lines.push(`| Unanchored facts / test (loose precision proxy) | ${num(agg.unanchored_per_test, 1)} |`);
  lines.push(
    `| Avg retrieval relevance (LLM-as-judge, 0–10) | ${num(agg.avg_retrieval_relevance, 2)} |`
  );
  lines.push(
    `| Answer substring hit rate | ${
      agg.answer_substring_hit_rate === null ? '—' : pct(agg.answer_substring_hit_rate)
    } |`
  );
  lines.push(
    `| Entity match hit rate | ${
      agg.matched_subjects_hit_rate === null ? '—' : pct(agg.matched_subjects_hit_rate)
    } |`
  );
  lines.push(`| Errors | ${agg.total_errors} |`);
  lines.push('');

  lines.push('## Per-test detail');
  lines.push('');
  for (const r of results) {
    lines.push(`### ${r.test_id}`);
    if (r.description) lines.push(r.description);
    lines.push('');
    lines.push(
      `Ingested **${r.ingested_turns}** turns → **${r.bursts_processed}** bursts processed in ${r.total_ms}ms.`
    );
    lines.push('');
    lines.push(
      `**Recall:** ${r.fact_recall.matched}/${r.fact_recall.expected}` +
        (r.fact_recall.missing.length > 0 ? ` — missing:` : '')
    );
    for (const m of r.fact_recall.missing) {
      lines.push(
        `  - subject=${m.subject}` +
          (m.category ? ` category=${m.category}` : '') +
          ` contains=[${m.content_contains.join(', ')}]`
      );
    }
    lines.push('');
    lines.push(
      `**Extracted:** ${r.fact_precision.total_extracted} total, ${r.fact_precision.unanchored} about unexpected subjects.`
    );
    lines.push('');
    if (r.queries.length > 0) {
      lines.push('**Queries:**');
      lines.push('');
      lines.push('| Query | Judge | Subs | Entity | Cites |');
      lines.push('|---|---|---|---|---|');
      for (const q of r.queries) {
        const judge =
          typeof q.retrieval_relevance_score === 'number'
            ? q.retrieval_relevance_score.toFixed(1)
            : '—';
        const subs =
          q.answer_substring_total > 0
            ? `${q.answer_substring_hits}/${q.answer_substring_total}`
            : '—';
        const ent = q.matched_subjects_ok ? '✓' : '✗';
        lines.push(
          `| ${escapeMd(q.q.slice(0, 60))} | ${judge} | ${subs} | ${ent} | ${q.citations_count} |`
        );
      }
      lines.push('');
    }
    if (r.errors.length > 0) {
      lines.push('**Errors:**');
      for (const e of r.errors) lines.push(`- ${e}`);
      lines.push('');
    }
  }

  return lines.join('\n');
}

function escapeMd(s: string): string {
  return s.replace(/\|/g, '\\|').replace(/\n/g, ' ');
}
