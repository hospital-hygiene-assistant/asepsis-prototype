'use strict';

// ── State ────────────────────────────────────────────────────
const state = {
  tests: [],
  currentTestId: null,
  currentResults: null,
  currentDoc: null,
};

// ── API ──────────────────────────────────────────────────────
async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
async function apiPost(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// ── Init ─────────────────────────────────────────────────────
async function init() {
  try {
    const [tests, modules] = await Promise.all([
      apiGet('/api/tests'),
      apiGet('/api/modules'),
    ]);
    state.tests = tests;
    renderModuleSelectors(modules);
    renderTestList(tests);
  } catch (e) {
    console.error('init error:', e);
  }
  document.getElementById('run-custom-btn').addEventListener('click', runCustomQuery);
  document.getElementById('custom-query').addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) runCustomQuery();
  });
}

// ── Module selectors ─────────────────────────────────────────
function renderModuleSelectors(modulesData) {
  const stageMap = { ingest: 'sel-ingest', index: 'sel-index', query: 'sel-query' };
  for (const [stage, { default: def, modules }] of Object.entries(modulesData.stages || {})) {
    const sel = document.getElementById(stageMap[stage]);
    if (!sel) continue;
    sel.innerHTML = '';
    for (const mod of modules) {
      const opt = document.createElement('option');
      opt.value = mod.name;
      opt.textContent = mod.label || mod.name;
      opt.title = mod.description || '';
      if (mod.name === def) opt.selected = true;
      sel.appendChild(opt);
    }
  }
}

function selectedModules() {
  return {
    ingest_module: document.getElementById('sel-ingest')?.value || null,
    index_module:  document.getElementById('sel-index')?.value  || null,
    query_module:  document.getElementById('sel-query')?.value  || null,
  };
}

// ── Sidebar ──────────────────────────────────────────────────
function renderTestList(tests) {
  const container = document.getElementById('test-list');
  const cats = [
    { key: 'SINGLE', label: 'Single Document' },
    { key: 'MULTI',  label: 'Multi-Section' },
    { key: 'CROSS',  label: 'Cross-Document' },
  ];
  for (const { key, label } of cats) {
    const group = tests.filter(t => t.category === key);
    if (!group.length) continue;
    const sec = el('div', 'sidebar-section');
    sec.appendChild(el('div', 'section-label', label));
    for (const t of group) {
      const btn = el('button', 'test-btn');
      btn.dataset.testId = t.id;
      btn.appendChild(el('span', 'test-title', t.description));
      btn.appendChild(el('span', 'test-query', t.query));
      btn.addEventListener('click', () => runTest(t.id));
      sec.appendChild(btn);
    }
    container.appendChild(sec);
  }
}

// ── Run ──────────────────────────────────────────────────────
async function runTest(testId) {
  state.currentTestId = testId;
  document.querySelectorAll('.test-btn').forEach(b => b.classList.remove('active'));
  const btn = document.querySelector(`[data-test-id="${testId}"]`);
  if (btn) btn.classList.add('active');

  const desc = state.tests.find(t => t.id === testId)?.description || testId;
  showLoading(`Running "${desc}" across all documents…`);
  try {
    const data = await apiPost('/api/run', { test_id: testId, ...selectedModules() });
    state.currentResults = data;
    showResults(data);
  } catch (e) {
    showError(e.message);
  }
}

async function runCustomQuery() {
  const query = document.getElementById('custom-query').value.trim();
  if (!query) return;
  state.currentTestId = null;
  document.querySelectorAll('.test-btn').forEach(b => b.classList.remove('active'));
  showLoading('Running custom query…');
  try {
    const data = await apiPost('/api/run', { query, ...selectedModules() });
    state.currentResults = data;
    showResults(data);
  } catch (e) {
    showError(e.message);
  }
}

// ── UI state ─────────────────────────────────────────────────
function showLoading(msg) {
  document.getElementById('welcome').style.display  = 'none';
  document.getElementById('loading').style.display  = 'flex';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('loading-msg').textContent = msg;
}
function showResults(data) {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('results').style.display = 'flex';
  renderResults(data);
}
function showError(msg) {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('welcome').style.display = 'flex';
  document.querySelector('#welcome p').textContent  = `Error: ${msg}`;
}

// ── Render results ───────────────────────────────────────────
function renderResults(data) {
  renderStatusBar(data);
  renderStatsBar(data);

  const docs = Object.keys(data.results);
  renderDocTabs(docs, data);
  if (docs.length) {
    state.currentDoc = docs[0];
    showDocTree(docs[0], data);
  }
  renderSnippets(data);
}

function renderStatusBar(data) {
  document.getElementById('query-display').textContent = data.query;

  const el2 = document.getElementById('test-status');
  el2.innerHTML = '';

  // Pipeline chips
  if (data.pipeline) {
    const wrap = el('div', 'pipeline-badge');
    const stageLabels = { ingest: '①', index: '②', query: '③' };
    for (const [stage, name] of Object.entries(data.pipeline)) {
      const chip = el('span', 'pipeline-chip');
      chip.innerHTML = `${stageLabels[stage] || stage} <span>${name}</span>`;
      chip.title = `${stage}: ${name}`;
      wrap.appendChild(chip);
    }
    el2.appendChild(wrap);
  }

  // Pass/fail badge
  if (!data.test_result) {
    el2.appendChild(el('span', 'status-badge neutral', 'Custom Query'));
    return;
  }
  const r = data.test_result;
  if (r.passed) {
    el2.appendChild(el('span', 'status-badge pass', '✓ PASS'));
  } else {
    const issues = [];
    if (Object.keys(r.missing).length)     issues.push('required missing');
    if (Object.keys(r.any_missing).length) issues.push('any-of missing');
    if (Object.keys(r.spurious).length)    issues.push('forbidden retrieved');
    el2.appendChild(el('span', 'status-badge fail', `✗ FAIL — ${issues.join('; ')}`));
  }
}

function renderStatsBar(data) {
  const bar = document.getElementById('stats-bar');
  bar.innerHTML = '';
  for (const [docName, docData] of Object.entries(data.results)) {
    const total     = countLeaves(docData.tree);
    const retrieved = docData.retrieved_ids.length;
    const pct       = total ? Math.round((retrieved / total) * 100) : 0;

    const card = el('div', 'stat-card');
    card.innerHTML = `
      <div class="stat-doc-name">${docName.replace(/_/g, ' ')}</div>
      <div class="stat-bar-wrap"><div class="stat-bar-fill" style="width:${pct}%"></div></div>
      <div class="stat-count">${retrieved} / ${total} leaves (${pct}%)</div>`;
    bar.appendChild(card);
  }
}

function countLeaves(tree) {
  let n = 0;
  function walk(nodes) { for (const nd of nodes) { nd.isLeaf ? n++ : walk(nd.children || []); } }
  walk(tree);
  return n;
}

function renderDocTabs(docs, data) {
  const tabs = document.getElementById('doc-tabs');
  tabs.innerHTML = '';
  for (const docName of docs) {
    const cnt  = data.results[docName].retrieved_ids.length;
    const tab  = el('button', 'doc-tab');
    tab.dataset.doc = docName;
    tab.innerHTML   = `${docName.replace(/_/g, ' ')} <span class="tab-count${cnt ? '' : ' zero'}">${cnt}</span>`;
    tab.addEventListener('click', () => {
      document.querySelectorAll('.doc-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      state.currentDoc = docName;
      showDocTree(docName, data);
    });
    tabs.appendChild(tab);
  }
  if (tabs.firstChild) tabs.firstChild.classList.add('active');
}

function showDocTree(docName, data) {
  const docData = data.results[docName];
  const tr      = data.test_result || {};
  renderTree(
    docData.tree,
    new Set(docData.retrieved_ids),
    new Set((tr.expected     || {})[docName] || []),
    new Set((tr.expected_any || {})[docName] || []),
    new Set((tr.forbidden    || {})[docName] || []),
  );
}

// ── D3 Tree ──────────────────────────────────────────────────
const STATUS_COLOR = {
  internal:        '#3b82f6',
  'expected-hit':  '#22c55e',
  retrieved:       '#86efac',
  'expected-miss': '#ef4444',
  forbidden:       '#f97316',
  neutral:         '#374151',
};
const STATUS_STROKE = {
  internal:        '#60a5fa',
  'expected-hit':  '#16a34a',
  retrieved:       '#22c55e',
  'expected-miss': '#dc2626',
  forbidden:       '#ea580c',
  neutral:         '#4b5563',
};
const STATUS_ICON = {
  'expected-hit':  '✓',
  'expected-miss': '✗',
  forbidden:       '!',
  retrieved:       '↓',
};

function nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, forbiddenSet) {
  const id = d.data.nodeId;
  if (!d.data.isLeaf) return 'internal';
  const got  = retrievedSet.has(id);
  const exp  = expectedSet.has(id) || expectedAnySet.has(id);
  const forb = forbiddenSet.has(id);
  if (got && forb)  return 'forbidden';
  if (got && exp)   return 'expected-hit';
  if (got)          return 'retrieved';
  if (exp)          return 'expected-miss';
  return 'neutral';
}

function renderTree(rawTree, retrievedSet, expectedSet, expectedAnySet, forbiddenSet) {
  const container = document.getElementById('tree-container');
  container.innerHTML = '';

  const treeData = rawTree.length === 1
    ? rawTree[0]
    : { nodeId: '_root', title: 'root', isLeaf: false, synthetic: false, children: rawTree };

  // d3 hierarchy — treat empty children arrays as leaves
  const root = d3.hierarchy(treeData, d =>
    (d.children && d.children.length) ? d.children : null
  );

  const spacingV = 36;   // px per row
  const spacingH = 240;  // px per depth level
  const mT = 24, mR = 200, mB = 24, mL = 12;

  const treeLayout = d3.tree().nodeSize([spacingV, spacingH]);
  treeLayout(root);

  // Bounding box
  let minX = Infinity, maxX = -Infinity, maxY = -Infinity;
  root.each(d => {
    if (d.x < minX) minX = d.x;
    if (d.x > maxX) maxX = d.x;
    if (d.y > maxY) maxY = d.y;
  });

  const svgW = maxY + spacingH + mL + mR;
  const svgH = (maxX - minX) + mT + mB;

  const svg = d3.select(container).append('svg')
    .attr('width', svgW)
    .attr('height', Math.max(svgH, 120))
    .style('display', 'block');

  const g = svg.append('g')
    .attr('transform', `translate(${mL},${mT - minX})`);

  // Zoom/pan
  svg.call(
    d3.zoom().scaleExtent([0.2, 3])
      .on('zoom', e => g.attr('transform', e.transform))
  );

  // Links
  g.append('g')
    .attr('fill', 'none').attr('stroke', '#334155').attr('stroke-width', 1.5)
    .selectAll('path')
    .data(root.links())
    .join('path')
    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));

  // Node groups
  const nodeG = g.append('g')
    .selectAll('g')
    .data(root.descendants())
    .join('g')
    .attr('transform', d => `translate(${d.y},${d.x})`);

  // Circles
  nodeG.append('circle')
    .attr('r', d => d.data.isLeaf ? 7 : 5)
    .attr('fill',   d => { const s = nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, forbiddenSet); return STATUS_COLOR[s]; })
    .attr('stroke', d => { const s = nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, forbiddenSet); return STATUS_STROKE[s]; })
    .attr('stroke-width', 1.5)
    .style('cursor', d => d.data.isLeaf ? 'pointer' : 'default')
    .on('click', (event, d) => { if (d.data.isLeaf) highlightSnippet(d.data.nodeId); });

  // Status icon (inside circle)
  nodeG.filter(d => d.data.isLeaf)
    .append('text')
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
    .attr('font-size', '8px').attr('fill', '#fff').attr('pointer-events', 'none')
    .text(d => {
      const s = nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, forbiddenSet);
      return STATUS_ICON[s] || '';
    });

  // Node label (nodeId in mono)
  nodeG.append('text')
    .attr('x', d => d.data.isLeaf ? 11 : 8)
    .attr('dominant-baseline', 'central')
    .attr('font-size', '11px')
    .attr('font-family', "'SF Mono','Fira Mono','Consolas',monospace")
    .attr('fill', d => d.data.isLeaf ? '#cbd5e1' : '#64748b')
    .attr('pointer-events', 'none')
    .text(d => {
      const id = d.data.nodeId || '';
      return id === '_root' ? '' : (id.length > 32 ? id.slice(0, 32) + '…' : id);
    });

  // [synth] annotation above label
  nodeG.filter(d => d.data.synthetic)
    .append('text')
    .attr('x', 11).attr('y', -10)
    .attr('font-size', '8px').attr('fill', '#a78bfa').attr('pointer-events', 'none')
    .text('[synth]');

  // Heading-level badge (h1, h2 …) below label for internal nodes
  nodeG.filter(d => !d.data.isLeaf && d.data.headingLevel)
    .append('text')
    .attr('x', 8).attr('y', 13)
    .attr('font-size', '8px').attr('fill', '#475569').attr('pointer-events', 'none')
    .text(d => `h${d.data.headingLevel}`);
}

function highlightSnippet(nodeId) {
  document.querySelectorAll('.snippet-card').forEach(c => c.classList.remove('highlight'));
  const card = document.getElementById(`snip-${nodeId}`);
  if (card) {
    card.classList.add('highlight');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

// ── Snippets ─────────────────────────────────────────────────
function renderSnippets(data) {
  const list = document.getElementById('snippets-list');
  list.innerHTML = '';

  let total = 0;
  for (const [docName, docData] of Object.entries(data.results)) {
    if (!docData.nodes.length) continue;

    const tr      = data.test_result || {};
    const expSet  = new Set([...(tr.expected?.[docName] || []), ...(tr.expected_any?.[docName] || [])]);
    const forbSet = new Set(tr.forbidden?.[docName] || []);

    for (const node of docData.nodes) {
      total++;
      const isExp  = expSet.has(node.node_id);
      const isForb = forbSet.has(node.node_id);

      const card = el('div', 'snippet-card');
      card.id = `snip-${node.node_id}`;

      // Tags
      const tags = el('div', 'snippet-tags');
      tags.appendChild(badge('doc-tag',   docName.replace(/_/g, ' ')));
      tags.appendChild(badge('id-tag',    node.node_id));
      tags.appendChild(badge('level-tag', `h${node.heading_level}`));
      if (node.synthetic) tags.appendChild(badge('synth-tag', 'synthetic'));
      if (isExp && !isForb) tags.appendChild(badge('expected-tag', 'expected'));
      if (isForb)           tags.appendChild(badge('forbidden-tag', 'forbidden'));

      const title   = el('div', 'snippet-title',   node.title);
      const content = el('div', 'snippet-content', node.content || '(no content)');

      card.appendChild(tags);
      card.appendChild(title);
      card.appendChild(content);
      list.appendChild(card);
    }
  }

  if (!total) {
    list.appendChild(el('div', 'empty-msg', 'No nodes retrieved.'));
  }
}

// ── Helpers ──────────────────────────────────────────────────
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls)  e.className   = cls;
  if (text) e.textContent = text;
  return e;
}
function badge(cls, text) { return el('span', `tag ${cls}`, text); }

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
