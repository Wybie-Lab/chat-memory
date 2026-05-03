import 'dotenv/config';
import { openDb, processUntilDrained } from '../src/engine';
import { createLLMProvider } from '../src/llm/claude';

async function main() {
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const db = openDb(dbPath);
  const provider = createLLMProvider();

  console.log('processing all unprocessed bursts...\n');

  const stats = await processUntilDrained(db, provider, {
    batchSize: 5,
    log: (line) => console.log(line),
  });

  console.log(
    `\ndone — bursts=${stats.bursts_scanned} kept=${stats.bursts_kept}/dropped=${stats.bursts_dropped} | facts: +${stats.facts_added} updated=${stats.facts_updated} deleted=${stats.facts_deleted} dup=${stats.facts_dropped} guarded=${stats.facts_guarded} | errors=${stats.errors}`
  );

  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
