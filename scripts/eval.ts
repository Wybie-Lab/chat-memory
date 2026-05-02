import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { createLLMProvider } from '../src/llm/claude';
import { runAll } from '../src/eval/runner';
import { renderMarkdown } from '../src/eval/report';
import type { TestCase } from '../src/eval/types';

const DEFAULT_DIR = path.resolve('./eval');
const DEFAULT_WORKDIR = path.resolve('./data/eval-tmp');

function parseArgs(argv: string[]): {
  dir: string;
  workdir: string;
  out: string | null;
  init: boolean;
  filter: string | null;
} {
  let dir = DEFAULT_DIR;
  let workdir = DEFAULT_WORKDIR;
  let out: string | null = null;
  let init = false;
  let filter: string | null = null;

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dir') dir = path.resolve(argv[++i]);
    else if (a === '--workdir') workdir = path.resolve(argv[++i]);
    else if (a === '--out') out = path.resolve(argv[++i]);
    else if (a === '--init') init = true;
    else if (a === '--only') filter = argv[++i];
    else if (a === '--help' || a === '-h') {
      console.log(`usage: npm run eval -- [--dir DIR] [--workdir DIR] [--out FILE] [--only ID] [--init]

  --dir DIR       Directory of *.json test cases (default: ./eval)
  --workdir DIR   Where temp sqlite DBs are written (default: ./data/eval-tmp)
  --out FILE      Write the markdown report to FILE (default: stdout)
  --only ID       Run only the test with matching id
  --init          Write an example test fixture to DIR/example.json and exit`);
      process.exit(0);
    }
  }

  return { dir, workdir, out, init, filter };
}

const EXAMPLE_TEST: TestCase = {
  id: 'anna-relocates',
  description: 'Anna mentions she moved from Berlin to Lisbon — last fact should win.',
  contact_name: 'Anna',
  contact_wa_id: '15555550100@c.us',
  me: 'Me',
  transcript: [
    {
      ts: '2025-09-10T19:30:00Z',
      direction: 'in',
      body: "hey! quick update — I'm officially employed at Stripe now, started last week",
    },
    {
      ts: '2025-09-10T19:31:30Z',
      direction: 'out',
      body: 'no way, congrats! still in Berlin?',
    },
    {
      ts: '2025-09-10T19:33:10Z',
      direction: 'in',
      body: 'yep still in Berlin, the office here is great',
    },
    {
      ts: '2026-02-04T10:12:00Z',
      direction: 'in',
      body: "btw I'm moving to Lisbon next month, the team is opening a hub there",
    },
    {
      ts: '2026-02-04T10:13:00Z',
      direction: 'out',
      body: 'oh wow, exciting!',
    },
    {
      ts: '2026-02-04T10:14:00Z',
      direction: 'in',
      body: 'yeah I love jazz btw, want to hit Umbria Jazz with you in July',
    },
  ],
  expected_facts: [
    { subject: 'anna', category: 'fact', content_contains: ['Stripe'] },
    { subject: 'anna', category: 'fact', content_contains: ['Lisbon'] },
    { subject: 'anna', category: 'preference', content_contains: ['jazz'] },
  ],
  queries: [
    {
      q: 'where does Anna live?',
      expected_answer_substrings: ['Lisbon'],
      expected_matched_subjects: ['anna'],
    },
    {
      q: 'what does Anna like?',
      expected_answer_substrings: ['jazz'],
      expected_matched_subjects: ['anna'],
    },
  ],
};

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.init) {
    fs.mkdirSync(args.dir, { recursive: true });
    const target = path.join(args.dir, 'example.json');
    if (fs.existsSync(target)) {
      console.error(`refusing to overwrite ${target}`);
      process.exit(1);
    }
    fs.writeFileSync(target, JSON.stringify(EXAMPLE_TEST, null, 2));
    console.log(`wrote ${target}`);
    return;
  }

  if (!fs.existsSync(args.dir)) {
    console.error(`eval dir ${args.dir} does not exist. Try: npm run eval -- --init`);
    process.exit(1);
  }

  const files = fs
    .readdirSync(args.dir)
    .filter((f) => f.endsWith('.json'))
    .sort();
  if (files.length === 0) {
    console.error(`no *.json test files in ${args.dir}. Try: npm run eval -- --init`);
    process.exit(1);
  }

  let tests: TestCase[] = [];
  for (const f of files) {
    const raw = fs.readFileSync(path.join(args.dir, f), 'utf-8');
    const t = JSON.parse(raw) as TestCase;
    tests.push(t);
  }
  if (args.filter) {
    tests = tests.filter((t) => t.id === args.filter);
    if (tests.length === 0) {
      console.error(`no test matched --only ${args.filter}`);
      process.exit(1);
    }
  }

  const provider = createLLMProvider();
  console.log(`running ${tests.length} test(s)...`);

  const results = await runAll(tests, provider, {
    workdir: args.workdir,
    log: (line) => console.log(line),
  });

  const md = renderMarkdown(results);
  if (args.out) {
    fs.writeFileSync(args.out, md);
    console.log(`\nreport: ${args.out}`);
  } else {
    console.log('\n' + md);
  }
}

main().catch((err) => {
  console.error('eval fatal:', err);
  process.exit(1);
});
