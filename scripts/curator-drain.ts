import 'dotenv/config';
import { openDb, drainPlannedAgentRuns, listAgentActionsForRun } from '../src/engine';
import { createLLMProvider } from '../src/llm';

async function main() {
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const limitArg = Number(process.env.LIMIT ?? '');
  const limit = Number.isFinite(limitArg) && limitArg > 0 ? Math.floor(limitArg) : 5;

  const db = openDb(dbPath);
  const provider = createLLMProvider();

  console.log(`draining up to ${limit} planned curator runs...\n`);

  const stats = await drainPlannedAgentRuns(db, provider, {
    limit,
    log: (line) => console.log(line),
  });

  console.log(
    `\ndone — drained=${stats.drained}, proposed_total=${stats.proposed_total}, errors=${stats.errors.length}`
  );
  if (stats.errors.length > 0) {
    for (const e of stats.errors) console.log(`  ERROR: ${e}`);
  }

  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
