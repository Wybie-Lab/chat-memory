import 'dotenv/config';
import { openDb } from '../src/memory/db';
import { createLLMProvider } from '../src/llm/claude';
import { processUntilDrained } from '../src/pipeline/process';

async function main() {
  const dbPath = process.env.DB_PATH ?? './data/memory.db';
  const db = openDb(dbPath);
  const provider = createLLMProvider();

  console.log('processing all unprocessed messages...\n');

  const stats = await processUntilDrained(db, provider, {
    batchSize: 10,
    log: (line) => console.log(line),
  });

  console.log(
    `\ndone — scanned=${stats.scanned} kept=${stats.kept} dropped=${stats.dropped} facts=${stats.facts} errors=${stats.errors}`
  );

  process.exit(0);
}

main().catch((err) => {
  console.error('fatal:', err);
  process.exit(1);
});
