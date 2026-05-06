import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { mapWithConcurrency } from '../src/engine';
import { judgeAnswer, type JudgeResult } from '../src/eval/locomo/judge';

interface QaResult {
  index: number;
  category: number;
  question: string;
  goldAnswer: string;
  predicted: string;
  score: { exactMatch: number; tokenF1: number };
  judge?: JudgeResult;
}

interface RunResult {
  sampleId: string;
  qa: QaResult[];
  aggregates?: unknown;
}

interface Args {
  inPath: string;
  outPath: string;
  concurrency: number;
  limit: number | null;
  onlyCategories: Set<number> | null;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    if (i < 0 || i === args.length - 1) return undefined;
    return args[i + 1];
  };
  const inPath = get('--in');
  if (!inPath) {
    console.error('usage: tsx scripts/judge-locomo.ts --in <results.json> [--out <judged.json>] [--concurrency 5] [--limit N] [--category 1,2,3]');
    process.exit(2);
  }
  const outPath = get('--out') ?? inPath.replace(/\.json$/, '.judged.json');
  const concurrency = Number(get('--concurrency') ?? '5');
  const limitStr = get('--limit');
  const limit = limitStr ? Number(limitStr) : null;
  const onlyCatsRaw = get('--category');
  const onlyCategories = onlyCatsRaw
    ? new Set<number>(onlyCatsRaw.split(',').map((s) => Number(s.trim())))
    : null;
  return { inPath, outPath, concurrency, limit, onlyCategories };
}

function aggregate(qa: QaResult[]) {
  const judged = qa.filter((q) => q.judge);
  const n = judged.length;
  if (n === 0) return { n: 0, judgeAccuracy: 0, meanF1: 0, byCategory: {} as Record<number, { n: number; judgeAccuracy: number; meanF1: number }> };
  const correct = judged.reduce((s, q) => s + (q.judge!.correct ? 1 : 0), 0);
  const buckets = new Map<number, QaResult[]>();
  for (const q of judged) {
    const list = buckets.get(q.category) ?? [];
    list.push(q);
    buckets.set(q.category, list);
  }
  const byCategory: Record<number, { n: number; judgeAccuracy: number; meanF1: number }> = {};
  for (const [cat, list] of buckets) {
    byCategory[cat] = {
      n: list.length,
      judgeAccuracy: list.reduce((s, q) => s + (q.judge!.correct ? 1 : 0), 0) / list.length,
      meanF1: list.reduce((s, q) => s + q.score.tokenF1, 0) / list.length,
    };
  }
  return {
    n,
    judgeAccuracy: correct / n,
    meanF1: judged.reduce((s, q) => s + q.score.tokenF1, 0) / n,
    byCategory,
  };
}

const CAT_NAMES: Record<number, string> = {
  1: 'single-hop',
  2: 'temporal',
  3: 'multi-hop',
  4: 'open-domain',
  5: 'adversarial',
};

async function main() {
  const args = parseArgs();
  const absIn = path.resolve(args.inPath);
  if (!fs.existsSync(absIn)) {
    console.error(`input not found: ${absIn}`);
    process.exit(1);
  }

  const data = JSON.parse(fs.readFileSync(absIn, 'utf-8')) as RunResult;
  const allQa = data.qa;
  const filtered = allQa.filter((q) => !args.onlyCategories || args.onlyCategories.has(q.category));
  const toJudge = args.limit ? filtered.slice(0, args.limit) : filtered;

  console.log(`judging ${toJudge.length}/${allQa.length} predictions from sample ${data.sampleId}`);
  console.log(`model: ${process.env.OPENROUTER_JUDGE_MODEL ?? process.env.OPENROUTER_MODEL ?? 'google/gemini-2.5-flash'}`);
  console.log(`concurrency: ${args.concurrency}`);
  console.log('');

  const t0 = Date.now();
  let done = 0;
  await mapWithConcurrency(toJudge, args.concurrency, async (q) => {
    try {
      const judge = await judgeAnswer({
        question: q.question,
        goldAnswer: q.goldAnswer,
        predicted: q.predicted,
        category: q.category,
      });
      q.judge = judge;
    } catch (err) {
      console.error(`  judge error on Q${q.index}: ${err instanceof Error ? err.message : err}`);
    }
    done++;
    if (done % 25 === 0 || done === toJudge.length) {
      console.log(`  ${done}/${toJudge.length} judged`);
    }
  });

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  const agg = aggregate(allQa);

  console.log('');
  console.log(`=== judged ${data.sampleId} in ${elapsed}s ===`);
  console.log(`overall: n=${agg.n} judge_accuracy=${(agg.judgeAccuracy * 100).toFixed(1)}% (token_F1=${(agg.meanF1 * 100).toFixed(1)}%)`);
  for (const [cat, stats] of Object.entries(agg.byCategory).sort()) {
    const name = CAT_NAMES[Number(cat)] ?? '';
    console.log(
      `  cat ${cat} (${name}): n=${stats.n} judge=${(stats.judgeAccuracy * 100).toFixed(1)}% F1=${(stats.meanF1 * 100).toFixed(1)}%`
    );
  }

  fs.mkdirSync(path.dirname(path.resolve(args.outPath)), { recursive: true });
  const out = { ...data, qa: allQa, judgeAggregates: agg };
  fs.writeFileSync(args.outPath, JSON.stringify(out, null, 2));
  console.log('');
  console.log(`wrote ${args.outPath}`);
  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
