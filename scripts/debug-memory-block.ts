import 'dotenv/config';
import { openDb, composeMemoryBlock } from '../src/engine';
import { createLLMProvider } from '../src/llm';

async function main() {
  const dbPath = process.argv[2] ?? './data/locomo-eval-0.db';
  const question = process.argv[3] ?? 'When did Caroline go to the LGBTQ support group?';
  const db = openDb(dbPath);
  const provider = createLLMProvider();
  const composed = await composeMemoryBlock(db, provider, question);
  console.log('Q:', question);
  console.log('matched_subjects:', composed.matched_subjects);
  console.log('threads matched:', composed.threads.map((t) => t.thread.name));
  console.log('budget:', JSON.stringify(composed.budget, null, 2));
  console.log('---');
  console.log(composed.block);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
