import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { openDb } from '../src/engine';
import { createLLMProvider } from '../src/llm';
import { parseSample, type LocomoSample } from '../src/eval/locomo/parse';
import { runEvalForSample } from '../src/eval/locomo/run';

interface Args {
  dataPath: string;
  sampleIndex: number;
  dbPath: string;
  limit: number | null;
  skipIngest: boolean;
  skipProcess: boolean;
  outPath: string | null;
  skipCategories: Set<number>;
  onlyCategories: Set<number> | null;
  qaConcurrency: number;
  agentic: boolean;
  agentMaxSteps: number;
}

function parseArgs(): Args {
  const args = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    if (i < 0 || i === args.length - 1) return undefined;
    return args[i + 1];
  };
  const has = (flag: string): boolean => args.includes(flag);

  const dataPath = get('--data') ?? './data/locomo/locomo10.json';
  const sampleIndex = Number(get('--sample') ?? '0');
  const limitStr = get('--limit');
  const limit = limitStr ? Number(limitStr) : null;
  const dbPath = get('--db') ?? `./data/locomo-eval-${sampleIndex}.db`;
  const outPath = get('--out') ?? null;
  const skipCatsRaw = get('--skip-categories');
  const skipCategories = new Set<number>(
    skipCatsRaw ? skipCatsRaw.split(',').map((s) => Number(s.trim())) : []
  );
  const onlyCatsRaw = get('--category') ?? get('--only-categories');
  const onlyCategories = onlyCatsRaw
    ? new Set<number>(onlyCatsRaw.split(',').map((s) => Number(s.trim())))
    : null;
  const qaConcurrency = Number(get('--qa-concurrency') ?? '5');
  const agentic = has('--agentic');
  const agentMaxSteps = Number(get('--agent-max-steps') ?? '8');

  if (Number.isNaN(sampleIndex) || sampleIndex < 0) {
    console.error('--sample must be a non-negative integer');
    process.exit(2);
  }
  if (limit !== null && (Number.isNaN(limit) || limit <= 0)) {
    console.error('--limit must be a positive integer');
    process.exit(2);
  }

  return {
    dataPath,
    sampleIndex,
    dbPath,
    limit,
    skipIngest: has('--skip-ingest'),
    skipProcess: has('--skip-process'),
    outPath,
    skipCategories,
    onlyCategories,
    qaConcurrency,
    agentic,
    agentMaxSteps,
  };
}

async function main() {
  const args = parseArgs();

  const absData = path.resolve(args.dataPath);
  if (!fs.existsSync(absData)) {
    console.error(`LOCOMO data not found at ${absData}`);
    console.error(`download with:`);
    console.error(`  curl -fsSL -o ${args.dataPath} https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json`);
    process.exit(1);
  }

  const samples = JSON.parse(fs.readFileSync(absData, 'utf-8')) as LocomoSample[];
  if (args.sampleIndex >= samples.length) {
    console.error(`--sample ${args.sampleIndex} out of range (have ${samples.length})`);
    process.exit(1);
  }
  const raw = samples[args.sampleIndex];
  const sample = parseSample(raw);
  console.log(
    `loaded sample ${args.sampleIndex}: id=${sample.sampleId} A=${sample.speakerA} B=${sample.speakerB} ` +
      `sessions=${sample.sessions.length} qa=${raw.qa.length}`
  );

  const dbExists = fs.existsSync(args.dbPath);
  if (!args.skipIngest && dbExists) {
    console.warn(`[warn] DB ${args.dbPath} already exists; pass --skip-ingest --skip-process to reuse, or delete it first`);
  }

  const db = openDb(args.dbPath);
  const provider = createLLMProvider();

  const t0 = Date.now();
  const result = await runEvalForSample(db, provider, sample, raw.qa, {
    skipIngest: args.skipIngest,
    skipProcess: args.skipProcess,
    limit: args.limit ?? undefined,
    skipCategories: args.skipCategories.size > 0 ? args.skipCategories : undefined,
    onlyCategories: args.onlyCategories ?? undefined,
    qaConcurrency: args.qaConcurrency,
    agentic: args.agentic,
    agentMaxSteps: args.agentMaxSteps,
  });
  const elapsedSec = ((Date.now() - t0) / 1000).toFixed(1);

  console.log('');
  console.log(`=== sample ${result.sampleId} ===`);
  console.log(`elapsed: ${elapsedSec}s`);
  console.log(
    `aggregate: n=${result.aggregates.n} EM=${result.aggregates.meanExactMatch.toFixed(3)} F1=${result.aggregates.meanTokenF1.toFixed(3)}`
  );
  for (const [cat, agg] of Object.entries(result.aggregates.byCategory)) {
    console.log(
      `  cat ${cat}: n=${agg.n} EM=${agg.meanExactMatch.toFixed(3)} F1=${agg.meanTokenF1.toFixed(3)}`
    );
  }

  if (args.outPath) {
    fs.mkdirSync(path.dirname(path.resolve(args.outPath)), { recursive: true });
    fs.writeFileSync(args.outPath, JSON.stringify(result, null, 2));
    console.log(`wrote per-question results to ${args.outPath}`);
  }

  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
