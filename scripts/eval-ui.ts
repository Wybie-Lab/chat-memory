import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import fs from 'fs';
import path from 'path';

const port = Number(process.env.EVAL_UI_PORT ?? 3002);
const dataDir = path.resolve(process.env.EVAL_DATA_DIR ?? './data');

const app = express();

app.get('/api/results', (_req, res) => {
  if (!fs.existsSync(dataDir)) return res.json({ files: [] });
  const files = fs
    .readdirSync(dataDir)
    .filter((f) => f.endsWith('.json') && (f.includes('judged') || f.includes('result')))
    .sort();
  res.json({ files });
});

app.get('/api/results/:filename', (req: Request, res: Response) => {
  const raw = req.params.filename;
  if (typeof raw !== 'string') return res.status(400).json({ error: 'bad filename' });
  const safe = path.basename(raw);
  const full = path.join(dataDir, safe);
  if (!fs.existsSync(full)) return res.status(404).json({ error: 'not found' });
  res.sendFile(full);
});

app.get('/', (_req, res) => {
  res.type('html').send(HTML);
});

app.listen(port, () => {
  console.log(`eval UI: http://localhost:${port}  (data dir: ${dataDir})`);
});

const HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LOCOMO eval viewer</title>
<style>
  :root {
    --bg: #0f1117;
    --panel: #1a1d27;
    --panel-2: #252834;
    --border: #2f333f;
    --fg: #e6e6ea;
    --muted: #8b94a8;
    --accent: #7aa2f7;
    --good: #9ece6a;
    --bad: #f7768e;
    --warn: #e0af68;
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, sans-serif; font-size:14px; line-height:1.4; }
  header { display:flex; align-items:center; gap:1rem; padding:0.75rem 1rem; background:var(--panel); border-bottom:1px solid var(--border); position:sticky; top:0; z-index:10; }
  header h1 { font-size:1rem; margin:0; font-weight:600; }
  select, button, input { background:var(--panel-2); color:var(--fg); border:1px solid var(--border); padding:0.4rem 0.6rem; border-radius:4px; font-family:inherit; font-size:inherit; }
  button { cursor:pointer; }
  button.active { background:var(--accent); color:#fff; border-color:var(--accent); }
  main { padding: 1rem; max-width: 100%; }
  .stats { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap:0.75rem; margin-bottom:1rem; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:6px; padding:0.75rem 1rem; }
  .stat .label { color:var(--muted); font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; }
  .stat .value { font-size:1.5rem; font-weight:600; margin-top:0.25rem; }
  .stat .sub { color:var(--muted); font-size:0.75rem; margin-top:0.1rem; }
  .filters { display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.75rem; align-items:center; }
  .filters input[type="text"] { flex:1; min-width:200px; }
  table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--border); border-radius:6px; overflow:hidden; }
  th, td { text-align:left; padding:0.6rem 0.75rem; vertical-align:top; border-bottom:1px solid var(--border); }
  th { background:var(--panel-2); color:var(--muted); font-weight:600; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; position:sticky; top:54px; z-index:5; }
  td.cat { white-space:nowrap; color:var(--muted); font-size:0.8rem; }
  td.q { font-weight:500; }
  td.a { color:var(--fg); font-weight:500; }
  td.pred { color:var(--fg); }
  td.pred.unknown { color:var(--muted); font-style:italic; }
  td.judge { white-space:nowrap; }
  td.judge .reason { display:block; color:var(--muted); font-size:0.75rem; font-style:italic; margin-top:0.35rem; white-space:normal; max-width: 320px; line-height:1.35; }
  .pill { display:inline-block; padding:0.15rem 0.5rem; border-radius:99px; font-size:0.75rem; font-weight:600; }
  .pill.good { background: rgba(158, 206, 106, 0.15); color: var(--good); }
  .pill.bad { background: rgba(247, 118, 142, 0.15); color: var(--bad); }
  .pill.warn { background: rgba(224, 175, 104, 0.15); color: var(--warn); }
  .pill.muted { background: rgba(139, 148, 168, 0.15); color: var(--muted); }
  tr.row-correct td.pred { border-left: 3px solid var(--good); padding-left: 0.5rem; }
  tr.row-incorrect td.pred { border-left: 3px solid var(--bad); padding-left: 0.5rem; }
  tr.row-unjudged td.pred { border-left: 3px solid var(--muted); padding-left: 0.5rem; }
  .empty { padding:3rem; text-align:center; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>LOCOMO eval viewer</h1>
  <select id="file"></select>
  <span id="meta" style="color:var(--muted); font-size:0.85rem;"></span>
</header>
<main>
  <div class="stats" id="stats"></div>
  <div class="filters">
    <button data-filter-cat="all" class="active">all</button>
    <button data-filter-cat="1">single-hop</button>
    <button data-filter-cat="2">temporal</button>
    <button data-filter-cat="3">multi-hop</button>
    <button data-filter-cat="4">open-domain</button>
    <button data-filter-cat="5">adversarial</button>
    <span style="width:1rem"></span>
    <button data-filter-result="all" class="active">all</button>
    <button data-filter-result="correct">correct</button>
    <button data-filter-result="incorrect">incorrect</button>
    <button data-filter-result="unjudged">unjudged</button>
    <span style="width:1rem"></span>
    <input type="text" id="search" placeholder="search question / answer / pred…">
  </div>
  <table>
    <thead>
      <tr>
        <th style="width:90px">cat</th>
        <th style="width:22%">question</th>
        <th style="width:14%">gold</th>
        <th style="width:24%">predicted</th>
        <th style="width:70px">F1</th>
        <th style="width:340px">judge</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
  <div class="empty" id="empty" style="display:none;">no rows match the current filters</div>
</main>

<script>
const CATS = { 1:'single-hop', 2:'temporal', 3:'multi-hop', 4:'open-domain', 5:'adversarial' };
const state = { filterCat: 'all', filterResult: 'all', search: '', data: null };

async function loadFiles() {
  const res = await fetch('/api/results');
  const { files } = await res.json();
  const sel = document.getElementById('file');
  sel.innerHTML = '';
  if (files.length === 0) {
    sel.innerHTML = '<option>no result files in data/</option>';
    return;
  }
  for (const f of files) {
    const opt = document.createElement('option');
    opt.value = f;
    opt.textContent = f;
    if (f.includes('judged')) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => loadFile(sel.value));
  loadFile(sel.value);
}

async function loadFile(name) {
  const res = await fetch('/api/results/' + encodeURIComponent(name));
  state.data = await res.json();
  document.getElementById('meta').textContent =
    'sample=' + (state.data.sampleId ?? '?') + ' · n=' + (state.data.qa?.length ?? 0);
  render();
}

function fmtPct(n) { return (n * 100).toFixed(1) + '%'; }

function aggregate(qa) {
  const judged = qa.filter(q => q.judge);
  const overall = {
    n_total: qa.length,
    n_judged: judged.length,
    judge: judged.length === 0 ? 0 : judged.filter(q => q.judge.correct).length / judged.length,
    f1: qa.length === 0 ? 0 : qa.reduce((s, q) => s + (q.score?.tokenF1 ?? 0), 0) / qa.length,
  };
  const byCat = {};
  for (let cat = 1; cat <= 5; cat++) {
    const sub = qa.filter(q => q.category === cat);
    const subJ = sub.filter(q => q.judge);
    if (sub.length === 0) continue;
    byCat[cat] = {
      n: sub.length,
      judge: subJ.length === 0 ? null : subJ.filter(q => q.judge.correct).length / subJ.length,
      f1: sub.reduce((s, q) => s + (q.score?.tokenF1 ?? 0), 0) / sub.length,
    };
  }
  return { overall, byCat };
}

function renderStats() {
  const stats = document.getElementById('stats');
  if (!state.data?.qa) { stats.innerHTML = ''; return; }
  const agg = aggregate(state.data.qa);
  const cards = [];
  cards.push(\`<div class="stat"><div class="label">overall judge</div><div class="value">\${fmtPct(agg.overall.judge)}</div><div class="sub">\${agg.overall.n_judged}/\${agg.overall.n_total} judged · F1 \${fmtPct(agg.overall.f1)}</div></div>\`);
  for (const [cat, s] of Object.entries(agg.byCat)) {
    const judgeStr = s.judge === null ? '—' : fmtPct(s.judge);
    cards.push(\`<div class="stat"><div class="label">\${CATS[cat]} (\${cat})</div><div class="value">\${judgeStr}</div><div class="sub">n=\${s.n} · F1 \${fmtPct(s.f1)}</div></div>\`);
  }
  stats.innerHTML = cards.join('');
}

function pillFor(q) {
  if (!q.judge) return '<span class="pill muted">—</span>';
  const pill = q.judge.correct
    ? '<span class="pill good">correct</span>'
    : '<span class="pill bad">wrong</span>';
  const reason = q.judge.reasoning ? \`<span class="reason">\${escapeHtml(q.judge.reasoning)}</span>\` : '';
  return pill + reason;
}

function rowClass(q) {
  if (!q.judge) return 'row-unjudged';
  return q.judge.correct ? 'row-correct' : 'row-incorrect';
}

function renderRows() {
  const tbody = document.getElementById('rows');
  const empty = document.getElementById('empty');
  if (!state.data?.qa) { tbody.innerHTML = ''; empty.style.display = 'block'; return; }
  const q = state.search.toLowerCase();
  const filtered = state.data.qa.filter(item => {
    if (state.filterCat !== 'all' && item.category !== Number(state.filterCat)) return false;
    if (state.filterResult === 'correct' && !(item.judge && item.judge.correct)) return false;
    if (state.filterResult === 'incorrect' && !(item.judge && !item.judge.correct)) return false;
    if (state.filterResult === 'unjudged' && item.judge) return false;
    if (q) {
      const blob = (item.question + ' ' + item.goldAnswer + ' ' + item.predicted).toLowerCase();
      if (!blob.includes(q)) return false;
    }
    return true;
  });
  empty.style.display = filtered.length === 0 ? 'block' : 'none';
  tbody.innerHTML = filtered.map(item => {
    const f1 = item.score?.tokenF1 ?? 0;
    const f1Class = f1 > 0.5 ? 'good' : f1 > 0.2 ? 'warn' : 'muted';
    const predClass = String(item.predicted ?? '').trim().toLowerCase() === 'unknown' ? 'pred unknown' : 'pred';
    return \`<tr class="\${rowClass(item)}">
      <td class="cat">\${item.category} · \${CATS[item.category] ?? ''}</td>
      <td class="q">\${escapeHtml(item.question)}</td>
      <td class="a">\${escapeHtml(item.goldAnswer)}</td>
      <td class="\${predClass}">\${escapeHtml(item.predicted)}</td>
      <td><span class="pill \${f1Class}">\${(f1 * 100).toFixed(0)}%</span></td>
      <td class="judge">\${pillFor(item)}</td>
    </tr>\`;
  }).join('');
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function render() {
  renderStats();
  renderRows();
}

document.querySelectorAll('[data-filter-cat]').forEach(b => b.addEventListener('click', () => {
  state.filterCat = b.getAttribute('data-filter-cat');
  document.querySelectorAll('[data-filter-cat]').forEach(x => x.classList.toggle('active', x === b));
  render();
}));
document.querySelectorAll('[data-filter-result]').forEach(b => b.addEventListener('click', () => {
  state.filterResult = b.getAttribute('data-filter-result');
  document.querySelectorAll('[data-filter-result]').forEach(x => x.classList.toggle('active', x === b));
  render();
}));
document.getElementById('search').addEventListener('input', e => {
  state.search = e.target.value;
  render();
});

loadFiles();
</script>
</body>
</html>`;
