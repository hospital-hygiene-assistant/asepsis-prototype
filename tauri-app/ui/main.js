'use strict';

// ── State ────────────────────────────────────────────────────
const state = {
  tests: [],
  currentTestId: null,
  currentResults: null,
  currentDoc: null,
  currentQuery: '',
  nodeSpacing: 36,
};

// ── Tooltip (created after DOM is ready — see init()) ────────
let _tooltip = null;
function getTooltip() {
  if (!_tooltip) {
    _tooltip = document.createElement('div');
    _tooltip.id = 'tree-tooltip';
    document.body.appendChild(_tooltip);
  }
  return _tooltip;
}

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
    const [tests, modules, config, status, docs] = await Promise.all([
      apiGet('/api/tests'),
      apiGet('/api/modules'),
      apiGet('/api/config'),
      apiGet('/api/status'),
      getDocs(),
    ]);
    state.tests = tests;
    renderModuleSelectors(modules);
    renderTestList(tests);
    initInstanceStepper(config);
    renderInstanceDots(status);
    renderTreemap(docs);
  } catch (e) {
    console.error('init error:', e);
  }
  document.getElementById('main-run-btn').addEventListener('click', runCustomQuery);
  document.getElementById('main-query-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      runCustomQuery();
    }
  });
  makeDraggable(document.getElementById('chat-input-container'));
}

// ── Ollama instance stepper + live activity indicators ────────
let _statusPoller = null;

function initInstanceStepper(config) {
  let count = config.ollama_instances || 1;
  const countEl  = document.getElementById('inst-count');
  const statusEl = document.getElementById('inst-status');
  const decBtn   = document.getElementById('inst-dec');
  const incBtn   = document.getElementById('inst-inc');

  function renderStepper(n, status) {
    count = n;
    countEl.textContent = n;
    decBtn.disabled = n <= 1;
    if (status) {
      const ok = status.live === status.requested;
      statusEl.textContent = ok
        ? `${status.live} instance${status.live > 1 ? 's' : ''} ready`
        : `${status.live}/${status.requested} ready${status.errors?.length ? ` — ${status.errors[0]}` : ''}`;
      statusEl.className = `inst-status ${ok ? 'inst-ok' : 'inst-warn'}`;
    }
    renderInstanceDots(null); // reset to idle
  }

  async function setCount(n) {
    countEl.textContent = '…';
    decBtn.disabled = incBtn.disabled = true;
    try {
      const result = await apiPost('/api/config', { ollama_instances: n });
      renderStepper(result.live || n, result);
    } catch (e) {
      statusEl.textContent = `Error: ${e.message}`;
      statusEl.className = 'inst-status inst-warn';
      renderStepper(count, null);
    } finally {
      incBtn.disabled = false;
      decBtn.disabled = count <= 1;
    }
  }

  decBtn.addEventListener('click', () => { if (count > 1) setCount(count - 1); });
  incBtn.addEventListener('click', () => setCount(count + 1));
  renderStepper(count, null);
}

function renderInstanceDots(statusData) {
  const container = document.getElementById('inst-indicators');
  if (!container) return;
  const instances = statusData?.instances;
  if (!instances) {
    // keep existing dots but mark all idle
    container.querySelectorAll('.inst-dot').forEach(d => {
      d.className = 'inst-dot idle';
      d.title = 'idle';
    });
    return;
  }
  // Rebuild if count changed
  if (container.children.length !== instances.length) {
    container.innerHTML = '';
    for (const inst of instances) {
      const dot = document.createElement('div');
      dot.className = 'inst-dot idle';
      container.appendChild(dot);
    }
  }
  // Update states
  const dots = container.querySelectorAll('.inst-dot');
  instances.forEach((inst, i) => {
    const busy = inst.active > 0;
    dots[i].className = `inst-dot ${busy ? 'busy' : 'idle'}`;
    dots[i].title = busy ? `Instance ${i + 1} — ${inst.active} leaf${inst.active > 1 ? 's' : ''} active` : `Instance ${i + 1} — idle`;
  });

  // Progress counter
  const prog = statusData?.progress;
  const statusEl = document.getElementById('inst-status');
  if (statusEl && prog && prog.total > 0) {
    const anyBusy = statusData?.any_busy;
    if (anyBusy) {
      statusEl.textContent = `Evaluating ${prog.done} / ${prog.total} leaves`;
      statusEl.className = 'inst-status inst-ok';
    } else if (prog.done > 0) {
      const pruned = prog.total - prog.done;
      statusEl.textContent = pruned > 0
        ? `Done — ${prog.done} evaluated, ${pruned} pruned`
        : `Done — ${prog.total} leaves evaluated`;
      statusEl.className = 'inst-status inst-ok';
    }
  }
}

function startStatusPolling() {
  if (_statusPoller) return;
  _statusPoller = setInterval(async () => {
    try {
      const status = await apiGet('/api/status');
      renderInstanceDots(status);
      applyTreemapEvents(status);
    } catch { /* ignore transient errors */ }
  }, 250);
}

async function stopStatusPolling() {
  if (_statusPoller) { clearInterval(_statusPoller); _statusPoller = null; }
  try {
    const status = await apiGet('/api/status');
    renderInstanceDots(status);
    applyTreemapEvents(status);
  } catch {
    renderInstanceDots(null);
  }
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

  const test = state.tests.find(t => t.id === testId);
  const desc = test?.description || testId;
  state.currentQuery = test?.query || '';
  
  document.getElementById('main-query-input').value = state.currentQuery;

  showLoading(`Running "${desc}" across all documents…`);
  startStatusPolling();
  try {
    const data = await apiPost('/api/run', { test_id: testId, ...selectedModules() });
    state.currentResults = data;
    showResults(data);
  } catch (e) {
    showError(e.message);
  } finally {
    await stopStatusPolling();
  }
}

async function runCustomQuery() {
  const query = document.getElementById('main-query-input').value.trim();
  if (!query) return;
  state.currentTestId = null;
  state.currentQuery = query;
  document.querySelectorAll('.test-btn').forEach(b => b.classList.remove('active'));
  showLoading('Running custom query…');
  startStatusPolling();
  try {
    const data = await apiPost('/api/run', { query, ...selectedModules() });
    state.currentResults = data;
    showResults(data);
  } catch (e) {
    showError(e.message);
  } finally {
    await stopStatusPolling();
  }
}

// ── UI state ─────────────────────────────────────────────────
let _docsCache = null;  // cached document trees for circle-pack viz

async function getDocs() {
  if (!_docsCache) _docsCache = await apiGet('/api/documents');
  return _docsCache;
}

async function showLoading(msg) {
  document.getElementById('welcome-viewport').style.display  = 'flex';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('welcome-status').textContent = msg;
  
  document.getElementById('main-query-input').disabled = true;
  document.getElementById('main-run-btn').disabled = true;

  try {
    const docs = await getDocs();
    renderTreemap(docs);
  } catch (e) {
    console.error('treemap render failed:', e);
  }
}
function showResults(data) {
  document.getElementById('welcome-viewport').style.display = 'none';
  document.getElementById('results').style.display = 'flex';
  
  document.getElementById('main-query-input').disabled = false;
  document.getElementById('main-run-btn').disabled = false;

  renderResults(data);
}
function showError(msg) {
  document.getElementById('welcome-viewport').style.display = 'flex';
  document.getElementById('results').style.display = 'none';
  document.getElementById('welcome-status').textContent  = `Error: ${msg}`;
  
  document.getElementById('main-query-input').disabled = false;
  document.getElementById('main-run-btn').disabled = false;
}

// ── Treemap visualization ────────────────────────────────────
// `_tmNodeById` maps nodeId → SVG <rect> for fast state updates from polling.
let _tmNodeById = {};

function renderTreemap(docs) {
  const container = document.getElementById('treemap-pack');
  container.innerHTML = '';
  _tmNodeById = {};

  const rect = container.getBoundingClientRect();
  const width = rect.width || 800;
  const height = rect.height || 500;

  // Build a single hierarchy: virtual root → docs → sections → leaves
  const rootData = {
    name: '_root',
    children: docs.map(d => ({
      name: d.name,
      nodeId: `__doc__${d.name}`,
      isDoc: true,
      children: d.tree,
    })),
  };

  const root = d3.hierarchy(rootData, n => n.children)
    .sum(n => n.isLeaf ? 1 : 0)
    .sort((a, b) => b.value - a.value);

  d3.treemap()
    .size([width, height])
    .paddingOuter(6)
    .paddingTop(20)
    .paddingInner(3)
    .round(true)(root);

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`)
    .style('background', 'transparent');

  const nodesToRender = root.descendants().filter(d => d.depth > 0);

  const leafG = svg.selectAll('g')
    .data(nodesToRender)
    .join('g')
    .attr('transform', d => `translate(${d.x0},${d.y0})`);

  leafG.append('rect')
    .attr('width', d => d.x1 - d.x0)
    .attr('height', d => d.y1 - d.y0)
    .attr('class', d => `tm-rect tm-pending ${d.data.isLeaf ? 'tm-leaf' : ''}`)
    .each(function(d) {
      this.__d = d;
      const id = d.data.nodeId;
      if (id) _tmNodeById[id] = this;
    })
    .on('mouseover', (event, d) => {
      const id = d.data.nodeId;
      if (!id || id === '_root') return;

      const el = _tmNodeById[id];
      let statusLabel = 'Pending evaluation';
      let statusClass = 'tt-muted';

      if (el) {
        if (el.classList.contains('tm-retrieved')) {
          statusLabel = '✓ Retrieved';
          statusClass = 'tt-selected';
        } else if (el.classList.contains('tm-rejected')) {
          statusLabel = '✗ Evaluated & Rejected';
          statusClass = 'tt-rejected';
        } else if (el.classList.contains('tm-pruned')) {
          statusLabel = '⊘ Pruned';
          statusClass = 'tt-pruned-label';
        }
      }

      let html = `<div class="tt-title">${escHtml(d.data.title || id)}</div>`;
      html += `<div class="tt-label ${statusClass}">${statusLabel}</div>`;

      if (d.data.isLeaf) {
        if (d.data.summary) {
          html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
        }
      } else {
        const depth = d.depth;
        const label = d.data.isDoc ? 'Document' : `Section · h${d.data.headingLevel || depth}`;
        html += `<div class="tt-label tt-section" style="margin-top:4px;">${label}</div>`;
        if (d.data.summary) {
          html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
        }
      }

      const tip = getTooltip();
      tip.innerHTML = html;
      tip.style.display = 'block';
      positionTooltip(event);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', () => { getTooltip().style.display = 'none'; });

  // Document name labels on top-level doc circles/rectangles
  leafG.filter(d => d.data.isDoc)
    .append('text')
    .attr('class', 'tm-doc-label')
    .attr('x', 6)
    .attr('y', 14)
    .text(d => d.data.name.replace(/_/g, ' '));

  // Leaf node title labels if they fit
  leafG.filter(d => d.data.isLeaf)
    .append('text')
    .attr('class', 'tm-label-text')
    .attr('x', 4)
    .attr('y', 14)
    .text(d => {
      const label = d.data.title || d.data.nodeId || '';
      const rectW = d.x1 - d.x0;
      const rectH = d.y1 - d.y0;
      const maxChars = Math.floor((rectW - 8) / 6.5);
      if (maxChars < 4 || rectH < 18) return '';
      return label.length > maxChars ? label.slice(0, maxChars - 1) + '…' : label;
    });
}

function applyTreemapEvents(status) {
  if (!status || !status.live) return;
  const live = status.live;
  const prunedSet = new Set(live.pruned || []);
  const rejectedSet = new Set(live.rejected || []);
  const retrievedSet = new Set(live.retrieved || []);

  for (const [id, el] of Object.entries(_tmNodeById)) {
    if (!el || !el.__d) continue;
    const d = el.__d;

    let isPruned = false;
    let isRejected = false;
    let isRetrieved = false;

    // Walk up the tree from this node to the root
    let curr = d;
    while (curr) {
      const currId = curr.data.nodeId;
      if (currId) {
        if (retrievedSet.has(currId)) {
          isRetrieved = true;
        }
        if (rejectedSet.has(currId)) {
          isRejected = true;
        }
        if (prunedSet.has(currId)) {
          isPruned = true;
        }
      }
      curr = curr.parent;
    }

    const isLeaf = el.classList.contains('tm-leaf');
    const baseClass = `tm-rect ${isLeaf ? 'tm-leaf' : ''}`;
    
    if (isRetrieved) {
      el.setAttribute('class', `${baseClass} tm-retrieved`);
    } else if (isRejected) {
      el.setAttribute('class', `${baseClass} tm-rejected`);
    } else if (isPruned) {
      el.setAttribute('class', `${baseClass} tm-pruned`);
    } else {
      el.setAttribute('class', `${baseClass} tm-pending`);
    }
  }

  // Update header progress text with current evaluation details
  const welcomeStatusEl = document.getElementById('welcome-status');
  if (welcomeStatusEl) {
    const prog = status.progress;
    if (prog && prog.total > 0) {
      welcomeStatusEl.textContent = `Evaluating ${prog.done}/${prog.total}`;
    }
  }
}

// ── Render results ───────────────────────────────────────────
function renderResults(data) {
  renderStatusBar(data);

  const docs = Object.keys(data.results);
  if (docs.length) {
    state.currentDoc = docs[0];
  }
  
  renderStatsBar(data);
  initGraphControls();

  if (docs.length) {
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
    el2.appendChild(el('span', 'status-badge fail', `✗ FAIL — ${issues.join('; ')}`));
  }
}

function renderStatsBar(data) {
  const bar = document.getElementById('stats-bar');
  bar.innerHTML = '';
  const docs = Object.keys(data.results);

  docs.forEach((docName) => {
    const docData   = data.results[docName];
    const total     = countLeaves(docData.tree);
    const retrieved = docData.retrieved_ids.length;
    const pct       = total ? Math.round((retrieved / total) * 100) : 0;

    const card = el('div', 'stat-card doc-tab-card');
    card.dataset.doc = docName;
    card.innerHTML = `
      <div class="stat-doc-name">${docName.replace(/_/g, ' ')}</div>
      <div class="stat-bar-wrap"><div class="stat-bar-fill" style="width:${pct}%"></div></div>
      <div class="stat-count">${retrieved} / ${total} leaves (${pct}%)</div>`;
    
    const tabSizeInput = document.getElementById('ctrl-tab-size');
    if (tabSizeInput) {
      card.style.minWidth = `${tabSizeInput.value}px`;
    }

    if (state.currentDoc === docName) {
      card.classList.add('active');
    }

    card.addEventListener('click', () => {
      document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      state.currentDoc = docName;
      showDocTree(docName, data);
    });

    bar.appendChild(card);
  });
}

function countLeaves(tree) {
  let n = 0;
  function walk(nodes) { for (const nd of nodes) { nd.isLeaf ? n++ : walk(nd.children || []); } }
  walk(tree);
  return n;
}

function showDocTree(docName, data) {
  const docData = data.results[docName];
  const tr      = data.test_result || {};
  renderTree(
    docData.tree,
    new Set(docData.retrieved_ids),
    new Set((tr.expected     || {})[docName] || []),
    new Set((tr.expected_any || {})[docName] || []),
    docData.node_meta || {},
  );
}

// ── D3 Tree ──────────────────────────────────────────────────
const STATUS_COLOR = {
  internal:        '#3b82f6', // Section (unevaluated)
  'expected-hit':  '#22c55e', // Expected hit
  retrieved:       '#86efac', // Retrieved
  'expected-miss': '#ef4444', // Expected missing
  kept:            '#0ea5e9', // Kept (passed pruning)
  pruned:          '#64748b', // Pruned
  rejected:        '#f97316', // Evaluated & Rejected
  neutral:         '#374151', // Leaf (unevaluated)
};
const STATUS_STROKE = {
  internal:        '#60a5fa',
  'expected-hit':  '#16a34a',
  retrieved:       '#22c55e',
  'expected-miss': '#dc2626',
  kept:            '#38bdf8',
  pruned:          '#475569',
  rejected:        '#ea580c',
  neutral:         '#4b5563',
};
const STATUS_ICON = {
  'expected-hit':  '✓',
  'expected-miss': '✗',
  retrieved:       '↓',
  rejected:        '×',
};

function nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons = {}) {
  const id = d.data.nodeId;
  const meta = nodeReasons[id] || {};

  // Check expected status first (ground truth comparison)
  const got = retrievedSet.has(id);
  const exp = expectedSet.has(id) || expectedAnySet.has(id);
  if (got && exp) return 'expected-hit';
  if (exp) return 'expected-miss';
  if (got) return 'retrieved';

  // Check node metadata status from pruning/evaluation
  if (meta.status === 'kept') return 'kept';
  if (meta.status === 'pruned') return 'pruned';
  if (meta.status === 'rejected') return 'rejected';

  // Fallback for unevaluated nodes
  if (!d.data.isLeaf) return 'internal';
  return 'neutral';
}

function renderTree(rawTree, retrievedSet, expectedSet, expectedAnySet, nodeReasons = {}) {
  const container = document.getElementById('tree-container');
  container.innerHTML = '';

  const treeData = rawTree.length === 1
    ? rawTree[0]
    : { nodeId: '_root', title: 'root', isLeaf: false, synthetic: false, children: rawTree };

  // d3 hierarchy — treat empty children arrays as leaves
  const root = d3.hierarchy(treeData, d =>
    (d.children && d.children.length) ? d.children : null
  );

    const spacingV = state.nodeSpacing || 36;   // px per row
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

  // Check if target or any of its descendants is retrieved
  function leadsToRetrieved(node, retrievedSet) {
    let found = false;
    node.each(d => {
      if (retrievedSet.has(d.data.nodeId)) {
        found = true;
      }
    });
    return found;
  }

  // Links
  const linkGroup = g.append('g')
    .selectAll('g.link-group')
    .data(root.links())
    .join('g')
    .attr('class', 'link-group');

  // The visible link
  linkGroup.append('path')
    .attr('fill', 'none')
    .attr('stroke', '#334155')
    .attr('stroke-width', 1.5)
    .attr('class', d => leadsToRetrieved(d.target, retrievedSet) ? 'link-highlight' : '')
    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));

  // The hover-catcher path (thick, invisible, hoverable)
  linkGroup.append('path')
    .attr('class', 'link-hover-catcher')
    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x))
    .on('mouseover', (event, d) => {
      const pathNodes = d.target.ancestors().reverse(); // from root to target
      let html = `<div class="tt-label tt-section" style="margin-bottom:8px; border-bottom:1px solid var(--border); padding-bottom:4px; font-weight:700;">Path Reasoning</div>`;
      html += `<div style="display:flex; flex-direction:column; gap:8px;">`;

      pathNodes.forEach((nd, index) => {
        const id = nd.data.nodeId;
        const meta = nodeReasons[id] || {};
        const status = meta.status;
        const title = nd.data.title || id || 'Document';

        let statusBadge = '';
        let statusStyle = '';
        if (status === 'retrieved' || retrievedSet.has(id)) {
          statusBadge = 'Retrieved';
          statusStyle = 'color: #86efac;';
        } else if (status === 'kept') {
          statusBadge = 'Kept';
          statusStyle = 'color: #38bdf8;';
        } else if (status === 'pruned') {
          statusBadge = 'Pruned';
          statusStyle = 'color: #64748b;';
        } else if (status === 'rejected') {
          statusBadge = 'Rejected';
          statusStyle = 'color: #f97316;';
        } else {
          statusBadge = nd.data.isLeaf ? 'Leaf' : 'Section';
          statusStyle = 'color: #94a3b8;';
        }

        html += `<div style="font-size:11px;">`;
        html += `<div style="font-weight:600; color:var(--text); display:flex; justify-content:space-between; gap:10px;">`;
        html += `<span>${index + 1}. ${escHtml(title.length > 24 ? title.slice(0, 24) + '…' : title)}</span>`;
        html += `<span style="font-size:9px; text-transform:uppercase; font-weight:700; ${statusStyle}">${statusBadge}</span>`;
        html += `</div>`;

        if (meta.reason) {
          html += `<div style="color:var(--muted); font-size:10px; margin-top:2px; line-height:1.4; padding-left:8px; border-left:1.5px solid var(--border);">${escHtml(meta.reason)}</div>`;
        } else if (nd.data.summary) {
          html += `<div style="color:var(--muted); font-size:10px; margin-top:2px; line-height:1.4; padding-left:8px; border-left:1.5px solid var(--border);">${escHtml(nd.data.summary)}</div>`;
        }
        html += `</div>`;
      });

      html += `</div>`;

      const tip = getTooltip();
      tip.innerHTML = html;
      tip.style.display = 'block';
      positionTooltip(event);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', () => { getTooltip().style.display = 'none'; });

  // Node groups
  const nodeG = g.append('g')
    .selectAll('g')
    .data(root.descendants())
    .join('g')
    .attr('transform', d => `translate(${d.y},${d.x})`);

  // Circles
  nodeG.append('circle')
    .attr('r', d => d.data.isLeaf ? 7 : 5)
    .attr('fill',   d => { const s = nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons); return STATUS_COLOR[s]; })
    .attr('stroke', d => { const s = nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons); return STATUS_STROKE[s]; })
    .attr('stroke-width', 1.5)
    .style('cursor', 'pointer')
    .on('click', (event, d) => { if (d.data.isLeaf) highlightSnippet(d.data.nodeId); })
    .on('mouseover', (event, d) => {
      const id   = d.data.nodeId;
      const meta = nodeReasons[id] || {};
      const status = meta.status;
      const isLeaf = d.data.isLeaf;

      let html = `<div class="tt-title">${escHtml(d.data.title || id || 'Document')}</div>`;

      if (status === 'retrieved') {
        html += `<div class="tt-label tt-selected">✓ Selected by model</div>`;
        if (meta.reason) html += `<div class="tt-reason">${escHtml(meta.reason)}</div>`;
        if (meta.quote)  html += `<div class="tt-quote" style="margin-top:6px; font-style:italic; border-left:2px solid var(--green); padding-left:6px; color:#cbd5e1;">"${escHtml(meta.quote)}"</div>`;
      } else if (status === 'rejected') {
        html += `<div class="tt-label tt-rejected" style="color:var(--orange)">✗ Evaluated & Rejected</div>`;
        if (meta.reason) html += `<div class="tt-reason">${escHtml(meta.reason)}</div>`;
        else if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      } else if (status === 'pruned') {
        html += `<div class="tt-label tt-pruned-label" style="color:var(--gray)">⊘ Pruned (Not evaluated)</div>`;
        if (meta.reason) html += `<div class="tt-reason">${escHtml(meta.reason)}</div>`;
        else if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      } else if (status === 'kept') {
        html += `<div class="tt-label tt-kept-label" style="color:var(--accent)">☉ Section Kept (Passed pruning)</div>`;
        if (meta.reason) html += `<div class="tt-reason">${escHtml(meta.reason)}</div>`;
        else if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      } else {
        if (isLeaf) {
          html += `<div class="tt-label tt-not-selected">Not retrieved</div>`;
          if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
        } else {
          const depth = d.depth;
          const label = depth === 0 ? 'Document' : `Section · h${d.data.headingLevel || depth}`;
          html += `<div class="tt-label tt-section">${label}</div>`;
          if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
        }
      }

      const tip = getTooltip();
      tip.innerHTML = html;
      tip.style.display = 'block';
      positionTooltip(event);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', () => { getTooltip().style.display = 'none'; });

  // Status icon (inside circle)
  nodeG.filter(d => d.data.isLeaf)
    .append('text')
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
    .attr('font-size', '8px').attr('fill', '#fff').attr('pointer-events', 'none')
    .text(d => {
      const s = nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons);
      return STATUS_ICON[s] || '';
    });

  // Node label (nodeId in mono)
  nodeG.append('text')
    .attr('x', d => d.data.isLeaf ? 11 : -11)
    .attr('text-anchor', d => d.data.isLeaf ? 'start' : 'end')
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
    .attr('x', -11).attr('y', 13)
    .attr('text-anchor', 'end')
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

    for (const node of docData.nodes) {
      total++;
      const isExp  = expSet.has(node.node_id);

      const card = el('div', 'snippet-card');
      card.id = `snip-${node.node_id}`;

      // Tags
      const tags = el('div', 'snippet-tags');
      tags.appendChild(badge('doc-tag',   docName.replace(/_/g, ' ')));
      tags.appendChild(badge('id-tag',    node.node_id));
      tags.appendChild(badge('level-tag', `h${node.heading_level}`));
      if (node.synthetic) tags.appendChild(badge('synth-tag', 'synthetic'));
      if (isExp) tags.appendChild(badge('expected-tag', 'expected'));

      const title   = el('div', 'snippet-title', node.title);
      const content = el('div', 'snippet-content');
      content.innerHTML = highlightRelevantContent(node.content || '(no content)', node.quote || '');

      card.appendChild(tags);
      card.appendChild(title);
      if (node.reason) {
        const reasonEl = el('div', 'snippet-reason');
        reasonEl.innerHTML = `<span class="reason-label">Model:</span> ${escHtml(node.reason)}`;
        card.appendChild(reasonEl);
      }
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

function escHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function positionTooltip(event) {
  const tip    = getTooltip();
  const margin = 14;
  const tw     = tip.offsetWidth  || 280;
  const th     = tip.offsetHeight || 80;
  let x = event.clientX + margin;
  let y = event.clientY + margin;
  if (x + tw > window.innerWidth)  x = event.clientX - tw - margin;
  if (y + th > window.innerHeight) y = event.clientY - th - margin;
  tip.style.left = `${x}px`;
  tip.style.top  = `${y}px`;
}

/**
 * Highlight the quote inside content using the model's verbatim quote.
 * Strategy:
 *   1. Try exact case-insensitive substring match of the full quote.
 *   2. If not found, try each sentence of the quote separately.
 *   3. If still nothing, return plain escaped content (no false highlights).
 */
function highlightRelevantContent(content, quote) {
  if (!quote || !content) return escHtml(content || '');

  // Build list of candidate spans to highlight, from longest to shortest
  const candidates = [quote];
  // Also try individual sentences from the quote
  quote.split(/[.!?]+/).forEach(s => { const t = s.trim(); if (t.length > 20) candidates.push(t); });

  for (const candidate of candidates) {
    const idx = content.toLowerCase().indexOf(candidate.toLowerCase());
    if (idx !== -1) {
      // Found — split content at this span and wrap
      const before = escHtml(content.slice(0, idx));
      const match  = escHtml(content.slice(idx, idx + candidate.length));
      const after  = escHtml(content.slice(idx + candidate.length));
      return `${before}<mark>${match}</mark>${after}`;
    }
  }

  // Nothing matched verbatim — return plain content without false highlights
  return escHtml(content);
}

function makeDraggable(el) {
  if (!el) return;
  let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  
  el.onmousedown = dragMouseDown;

  function dragMouseDown(e) {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'BUTTON' || e.target.closest('button')) {
      return;
    }
    e = e || window.event;
    e.preventDefault();
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    
    let newTop = el.offsetTop - pos2;
    let newLeft = el.offsetLeft - pos1;
    
    newTop = Math.max(10, Math.min(window.innerHeight - el.offsetHeight - 10, newTop));
    newLeft = Math.max(10, Math.min(window.innerWidth - el.offsetWidth - 10, newLeft));
    
    el.style.top = newTop + "px";
    el.style.left = newLeft + "px";
    el.style.bottom = "auto";
    el.style.right = "auto";
  }

  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

function initGraphControls() {
  const treeArea = document.getElementById('tree-area');
  if (!treeArea) return;
  treeArea.style.position = 'relative';
  
  if (document.getElementById('graph-controls')) return;
  
  const controls = el('div');
  controls.id = 'graph-controls';
  controls.style.cssText = 'position:absolute; top:12px; right:16px; z-index:100; display:flex; gap:12px; background:var(--surface); border:1px solid var(--border); padding:6px 12px; border-radius:16px; align-items:center; box-shadow:0 2px 10px rgba(0,0,0,0.3);';
  
  controls.innerHTML = `
    <div style="display:flex; align-items:center; gap:6px; font-size:10px; color:var(--muted);">
      <span>Tab Size</span>
      <input type="range" id="ctrl-tab-size" min="80" max="220" value="160" style="width:60px; accent-color:var(--accent); cursor:pointer;">
    </div>
    <div style="display:flex; align-items:center; gap:6px; font-size:10px; color:var(--muted);">
      <span>Node Spacing</span>
      <input type="range" id="ctrl-node-spacing" min="20" max="60" value="${state.nodeSpacing}" style="width:60px; accent-color:var(--accent); cursor:pointer;">
    </div>
    <button id="ctrl-toggle-snippets" style="background:var(--surface2); border:1px solid var(--border); border-radius:4px; padding:2px 8px; font-size:10px; color:var(--text); cursor:pointer;">Hide Snippets</button>
  `;
  
  treeArea.appendChild(controls);
  
  document.getElementById('ctrl-tab-size').addEventListener('input', e => {
    const val = e.target.value;
    document.querySelectorAll('.stat-card').forEach(card => {
      card.style.minWidth = `${val}px`;
    });
  });
  
  document.getElementById('ctrl-node-spacing').addEventListener('input', e => {
    state.nodeSpacing = parseInt(e.target.value);
    if (state.currentDoc && state.currentResults) {
      showDocTree(state.currentDoc, state.currentResults);
    }
  });
  
  document.getElementById('ctrl-toggle-snippets').addEventListener('click', () => {
    const panel = document.getElementById('snippets-panel');
    const btn = document.getElementById('ctrl-toggle-snippets');
    if (panel.style.display === 'none') {
      panel.style.display = 'flex';
      btn.textContent = 'Hide Snippets';
    } else {
      panel.style.display = 'none';
      btn.textContent = 'Show Snippets';
    }
  });
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
