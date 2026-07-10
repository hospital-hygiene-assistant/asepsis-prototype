'use strict';

// ── State ────────────────────────────────────────────────────
const state = {
  tests: [],
  currentTestId: null,
  currentResults: null,
  currentDoc: null,
  currentQuery: '',
  nodeSpacing: 36,
  resultView: 'graph',   // 'graph' (D3 tree) | 'boxes' (treemap)
  docViewerStem: null,   // stem currently open in the doc-viewer modal
  liveStatus: null,      // latest /api/status snapshot (for live verdicts)
  explainCache: {},      // {stem: {node_id: {topic, reason, addresses_query}}}
  ingestModules: {},     // {name: MODULE_INFO} for the ingest stage
  showPins: false,       // doc-viewer provenance-pin visibility toggle
};

// ── Verdict helpers (shared by doc-viewer bands, TOC, and result boxes) ──
// Prefers the final run results; falls back to live /api/status events so the
// doc viewer can color + explain decisions in real time mid-run.
function getDocVerdicts(stem) {
  const res = state.currentResults?.results?.[stem];
  if (res) return { retrieved: new Set(res.retrieved_ids || []), meta: res.node_meta || {}, live: false };

  const live = state.liveStatus?.live;
  if (live && (live.meta || live.retrieved)) {
    const meta = { ...(live.meta || {}) };
    const ensure = (ids, status) => (ids || []).forEach(id => { if (!meta[id]) meta[id] = { status }; });
    ensure(live.retrieved, 'retrieved'); ensure(live.rejected, 'rejected');
    ensure(live.kept, 'kept');           ensure(live.pruned, 'pruned');
    return { retrieved: new Set(live.retrieved || []), meta, live: true };
  }
  return null;
}
function verdictFor(baseId, v) {
  if (!v) return null;
  const m = v.meta[baseId] || null;
  if (v.retrieved.has(baseId) || m?.status === 'retrieved') return 'accepted';
  if (m?.status === 'rejected') return 'rejected';
  if (m?.status === 'pruned')   return 'pruned';
  if (m?.status === 'kept')     return 'kept';
  return null;
}

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
  let docs = [];
  try {
    const [tests, modules, config, status, fetchedDocs] = await Promise.all([
      apiGet('/api/tests'),
      apiGet('/api/modules'),
      apiGet('/api/config'),
      apiGet('/api/status'),
      getDocs(),
    ]);
    state.tests = tests;
    docs = fetchedDocs;
    renderModuleSelectors(modules);
    renderTestList(tests);
    initInstanceStepper(config);
    renderInstanceDots(status);
    renderTreemap(docs);
  } catch (e) {
    console.error('init error:', e);
  }
  
  // Welcome setup card visibility based on documents
  const setupCard = document.getElementById('welcome-setup-card');
  const treemapPack = document.getElementById('treemap-pack');
  const treemapLegend = document.getElementById('treemap-pack-legend');
  const toggleBtn = document.getElementById('welcome-toggle-setup-btn');
  const welcomeStatus = document.getElementById('welcome-status');
  const welcomeSourcePath = document.getElementById('welcome-source-path');

  // Pre-populate welcome path if already configured
  try {
    const src = await apiGet('/api/ingest/source?module=betteringest_pdf');
    if (src.source_dir) {
      if (welcomeSourcePath) welcomeSourcePath.value = src.source_dir;
      const runBtn = document.getElementById('welcome-run-btn');
      if (runBtn) runBtn.disabled = false;
      const modalInput = document.getElementById('ingest-source-path');
      if (modalInput) modalInput.value = src.source_dir;
      const modalRunBtn = document.getElementById('ingest-run-btn');
      if (modalRunBtn) modalRunBtn.disabled = false;
    }
  } catch (e) {}

  if (docs.length === 0) {
    if (setupCard) setupCard.style.display = 'flex';
    if (treemapPack) treemapPack.style.display = 'none';
    if (treemapLegend) treemapLegend.style.display = 'none';
    if (toggleBtn) toggleBtn.style.display = 'none';
    if (welcomeStatus) welcomeStatus.textContent = "No documents found in the database. Please select a folder of PDFs to begin.";
  } else {
    if (setupCard) setupCard.style.display = 'none';
    if (treemapPack) treemapPack.style.display = 'block';
    if (treemapLegend) treemapLegend.style.display = 'flex';
    if (toggleBtn) {
      toggleBtn.style.display = 'inline-block';
      toggleBtn.textContent = '📁 Configure Source Folder';
    }
    if (welcomeStatus) welcomeStatus.textContent = "Select a test case from the sidebar or enter a custom query below to start.";
  }

  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      if (setupCard.style.display === 'none') {
        setupCard.style.display = 'flex';
        treemapPack.style.display = 'none';
        treemapLegend.style.display = 'none';
        toggleBtn.textContent = '✕ Close Setup';
      } else {
        setupCard.style.display = 'none';
        treemapPack.style.display = 'block';
        treemapLegend.style.display = 'flex';
        toggleBtn.textContent = '📁 Configure Source Folder';
      }
    });
  }

  initChat();
  makeDraggable(document.getElementById('chat-input-container'));
  initDocViewer();
  initIngestSetup();
}

// ── Chat window: collapse/expand, hotkey, auto-grow ──────────
function setChatExpanded(expanded) {
  const container = document.getElementById('chat-input-container');
  const input = document.getElementById('main-query-input');
  container.classList.toggle('collapsed', !expanded);
  if (expanded) {
    requestAnimationFrame(() => { input.focus(); autoGrowTextarea(input); });
  } else {
    input.blur();
  }
}

// Grow the textarea with content, up to ~10 lines, then scroll.
function autoGrowTextarea(ta) {
  ta.style.height = 'auto';
  const cs = getComputedStyle(ta);
  const line = parseFloat(cs.lineHeight) || 19;
  const padY = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
  const maxH = line * 10 + padY;
  ta.style.height = Math.min(ta.scrollHeight, maxH) + 'px';
  ta.style.overflowY = ta.scrollHeight > maxH ? 'auto' : 'hidden';
}

function initChat() {
  const container = document.getElementById('chat-input-container');
  const launcher  = document.getElementById('chat-launcher');
  const collapse  = document.getElementById('chat-collapse');
  const input     = document.getElementById('main-query-input');

  document.getElementById('main-run-btn').addEventListener('click', runCustomQuery);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runCustomQuery(); }
    else if (e.key === 'Escape')          { e.preventDefault(); setChatExpanded(false); }
  });
  input.addEventListener('input', () => autoGrowTextarea(input));

  launcher.addEventListener('click', () => setChatExpanded(true));
  collapse.addEventListener('click', () => setChatExpanded(false));

  // Global hotkey: ⌘K / Ctrl+K toggles the chat; "/" opens it when not typing.
  document.addEventListener('keydown', (e) => {
    const ae = document.activeElement;
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(ae?.tagName || '') || ae?.isContentEditable;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      setChatExpanded(container.classList.contains('collapsed'));
    } else if (e.key === '/' && !typing && container.classList.contains('collapsed')) {
      e.preventDefault();
      setChatExpanded(true);
    }
  });
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
      state.liveStatus = status;
      renderInstanceDots(status);
      applyTreemapEvents(status);
      refreshOpenDocViewer();
    } catch { /* ignore transient errors */ }
  }, 250);
}

async function stopStatusPolling() {
  if (_statusPoller) { clearInterval(_statusPoller); _statusPoller = null; }
  try {
    const status = await apiGet('/api/status');
    state.liveStatus = status;
    renderInstanceDots(status);
    applyTreemapEvents(status);
    refreshOpenDocViewer();
  } catch {
    renderInstanceDots(null);
  }
}

// Re-decorate the doc viewer if it's open (called on every status poll).
function refreshOpenDocViewer() {
  if (!state.docViewerStem) return;
  const modal = document.getElementById('doc-modal');
  if (!modal || modal.style.display === 'none') return;
  decorateDocVerdicts(document.getElementById('doc-modal-content'), state.docViewerStem);
}

// ── Module selectors ─────────────────────────────────────────
function renderModuleSelectors(modulesData) {
  const stageMap = { ingest: 'sel-ingest', index: 'sel-index', query: 'sel-query' };
  for (const mod of modulesData.stages?.ingest?.modules || []) {
    state.ingestModules[mod.name] = mod;
  }
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
  
  const qInput = document.getElementById('main-query-input');
  qInput.value = state.currentQuery;
  autoGrowTextarea(qInput);

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
    .style('cursor', 'pointer')
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
    .on('mouseout', () => { getTooltip().style.display = 'none'; })
    .on('click', (event, d) => {
      // Find the enclosing document and open the viewer.
      let curr = d;
      while (curr && !curr.data.isDoc) curr = curr.parent;
      if (!curr) return;
      const stem = curr.data.name;
      // Scroll to a specific heading when a section/leaf was clicked.
      const targetTitle = d.data.isDoc ? null : (d.data.title || null);
      getTooltip().style.display = 'none';
      openDocViewer(stem, targetTitle);
    });

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

// ── Document viewer modal ────────────────────────────────────
function slugify(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')   // drop punctuation
    .trim()
    .replace(/\s+/g, '-');
}

async function openDocViewer(stem, targetTitle = null) {
  const modal   = document.getElementById('doc-modal');
  const titleEl = document.getElementById('doc-modal-title');
  const tocEl   = document.getElementById('doc-modal-toc');
  const bodyEl  = document.getElementById('doc-modal-content');

  titleEl.textContent = (stem || 'Document').replace(/_/g, ' ');
  tocEl.innerHTML = '';
  bodyEl.innerHTML = '<div class="empty-msg">Loading…</div>';
  modal.style.display = 'flex';

  let doc;
  try {
    doc = await apiGet(`/api/document/${encodeURIComponent(stem)}/full`);
  } catch (e) {
    bodyEl.innerHTML = `<div class="empty-msg">Could not load document: ${escHtml(e.message)}</div>`;
    return;
  }

  // Render markdown (marked is loaded globally as `marked`).
  const html = (typeof marked !== 'undefined')
    ? (marked.parse ? marked.parse(doc.markdown) : marked(doc.markdown))
    : `<pre>${escHtml(doc.markdown)}</pre>`;
  bodyEl.innerHTML = html;

  // Provenance pins: upgrade ```pin code blocks into chips; one checkbox
  // (header) shows/hides them all — same rendered DOM either way.
  const pinCount = decoratePinBlocks(bodyEl, stem);
  initPinToggle(bodyEl, pinCount);

  // Assign anchor IDs to headings and build the TOC from them.
  const docV = getDocVerdicts(stem);
  const headings = bodyEl.querySelectorAll('h1, h2, h3, h4');
  const seen = {};
  headings.forEach(h => {
    let id = slugify(h.textContent);
    if (!id) return;
    if (seen[id] != null) { seen[id]++; id = `${id}-${seen[id]}`; }
    else { seen[id] = 0; }
    h.id = id;

    const link = document.createElement('a');
    link.href = `#${id}`;
    link.className = `toc-${h.tagName.toLowerCase()}`;
    // Share the verdict color scheme on the TOC entry.
    const vd = verdictFor(slugify(h.textContent), docV);
    if (vd) link.classList.add(`toc-v-${vd}`);
    link.textContent = h.textContent;
    link.addEventListener('click', (ev) => {
      ev.preventDefault();
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    tocEl.appendChild(link);
  });

  // Overlay verdicts (live during a run, or final results). Track the open
  // stem so the status poller can refresh this view in real time.
  state.docViewerStem = stem;
  bodyEl.dataset.vsig = '';            // force a fresh decoration
  decorateDocVerdicts(bodyEl, stem);

  // Scroll to a requested heading (when a section/leaf box was clicked).
  if (targetTitle) {
    const want = slugify(targetTitle);
    let match = bodyEl.querySelector(`#${CSS.escape(want)}`);
    if (!match) {
      // Fuzzy fallback: first heading whose slug starts with the target slug.
      match = Array.from(headings).find(h => slugify(h.textContent).startsWith(want));
    }
    if (match) {
      // Defer so layout is ready before scrolling.
      requestAnimationFrame(() => match.scrollIntoView({ behavior: 'auto', block: 'start' }));
    } else {
      bodyEl.scrollTop = 0;
    }
  } else {
    bodyEl.scrollTop = 0;
  }
}

const VERDICT_CLASSES = ['accepted', 'rejected', 'kept', 'pruned'];

/**
 * Annotate the rendered markdown with the model's per-node verdicts. Each
 * heading and its following content get a verdict class (sage = retrieved,
 * rose = rejected, blue = kept, greyed = pruned); accepted leaves get the
 * deciding quote highlighted in amber. Every annotated element carries
 * data-vid so hover can surface the decision.
 *
 * Idempotent and safe to re-run on every status poll: it clears prior
 * annotations first and skips work when the verdict signature is unchanged.
 * Reads final results when available, else live /api/status events.
 */
function decorateDocVerdicts(bodyEl, stem) {
  if (!bodyEl) return;
  const legend = document.getElementById('doc-verdict-legend');
  const v = getDocVerdicts(stem);

  // Signature guard — avoid churning the DOM when nothing changed.
  const sig = v ? JSON.stringify(Object.keys(v.meta).sort().map(k => k + ':' + (v.meta[k].status || '')))
                : '';
  if (bodyEl.dataset.vsig === sig) return;
  bodyEl.dataset.vsig = sig;

  // Clear previous annotations (classes, data-vid, quote marks).
  bodyEl.querySelectorAll('[data-vid]').forEach(el => {
    VERDICT_CLASSES.forEach(c => el.classList.remove(`md-h-${c}`, `md-c-${c}`));
    delete el.dataset.vid;
  });
  bodyEl.querySelectorAll('mark.md-quote').forEach(mk => {
    mk.replaceWith(document.createTextNode(mk.textContent));
  });
  bodyEl.normalize();

  if (legend) legend.style.display = 'none';
  if (!v) return;

  const headings = Array.from(bodyEl.querySelectorAll('h1, h2, h3, h4'));
  let annotated = 0;

  headings.forEach(h => {
    const baseId = slugify(h.textContent);     // matches pageindex node_id
    const verdict = verdictFor(baseId, v);
    if (!verdict) return;
    const m = v.meta[baseId] || null;

    h.classList.add(`md-h-${verdict}`);
    h.dataset.vid = baseId;

    // Tag the content siblings up to (not including) the next heading.
    const contentEls = [];
    let sib = h.nextElementSibling;
    while (sib && !/^H[1-4]$/.test(sib.tagName)) {
      sib.classList.add(`md-c-${verdict}`);
      sib.dataset.vid = baseId;
      contentEls.push(sib);
      sib = sib.nextElementSibling;
    }

    if (verdict === 'accepted' && m?.quote) {
      highlightQuoteInEls(contentEls.length ? contentEls : [h], m.quote);
    }
    annotated++;
  });

  if (legend && annotated > 0) legend.style.display = 'flex';
}

/** Highlight the model's deciding quote (verbatim, else longest matching
 *  sentence) in a <mark> across the given elements. Single-text-node match. */
function highlightQuoteInEls(els, quote) {
  const candidates = [quote];
  quote.split(/[.!?]+/).forEach(s => { const t = s.trim(); if (t.length > 20) candidates.push(t); });

  for (const cand of candidates) {
    for (const root of els) {
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
      let node;
      while ((node = walker.nextNode())) {
        const idx = node.nodeValue.toLowerCase().indexOf(cand.toLowerCase());
        if (idx !== -1) {
          try {
            const range = document.createRange();
            range.setStart(node, idx);
            range.setEnd(node, idx + cand.length);
            const mark = document.createElement('mark');
            mark.className = 'md-quote';
            range.surroundContents(mark);
            return true;
          } catch { /* boundary-spanning match — try next candidate */ }
        }
      }
    }
  }
  return false;
}

function closeDocViewer() {
  document.getElementById('doc-modal').style.display = 'none';
  state.docViewerStem = null;
}

// Tooltip describing the model's decision for the hovered paragraph/heading.
function showDecisionTooltip(event, vid) {
  const stem = state.docViewerStem;
  const v = stem ? getDocVerdicts(stem) : null;
  const m = v?.meta?.[vid];
  const verdict = v ? verdictFor(vid, v) : null;
  if (!verdict) { getTooltip().style.display = 'none'; return; }

  const LABEL = {
    accepted: ['✓ Retrieved', 'tt-selected'],
    rejected: ['✗ Evaluated & not selected', 'tt-rejected'],
    kept:     ['☉ Section kept (passed pruning)', ''],
    pruned:   ['⊘ Pruned (branch skipped)', 'tt-pruned-label'],
  }[verdict];

  let html = `<div class="tt-label ${LABEL[1]}">${LABEL[0]}${v.live ? ' · live' : ''}</div>`;
  if (m?.reason) html += `<div class="tt-reason">${escHtml(m.reason)}</div>`;
  if (m?.quote)  html += `<div class="tt-quote" style="margin-top:6px;font-style:italic;border-left:2px solid var(--green);padding-left:6px;color:#cbd5e1;">"${escHtml(m.quote)}"</div>`;

  // For negatives, surface any cached on-demand explanation, else invite one.
  if (verdict === 'rejected' || verdict === 'pruned') {
    const ex = state.explainCache?.[stem]?.[vid];
    if (ex && (ex.topic || ex.reason)) {
      if (ex.topic)  html += `<div class="tt-reason" style="margin-top:6px;"><span style="color:var(--muted)">Topic:</span> ${escHtml(ex.topic)}</div>`;
      if (ex.reason) html += `<div class="tt-reason tt-muted">${escHtml(ex.reason)}</div>`;
    } else if (!m?.reason) {
      html += `<div class="tt-reason tt-muted">The model judged this does not directly answer the query.</div>`;
    }
    html += `<div class="tt-hint" style="margin-top:6px;color:var(--accent);font-size:10px;">▸ click to ask the model why (grounded re-read)</div>`;
  }

  const tip = getTooltip();
  tip.innerHTML = html;
  tip.style.display = 'block';
  positionTooltip(event);
}

function initDocViewer() {
  const modal = document.getElementById('doc-modal');
  if (!modal) return;
  document.getElementById('doc-modal-close').addEventListener('click', closeDocViewer);
  modal.querySelector('.doc-modal-backdrop').addEventListener('click', closeDocViewer);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.style.display !== 'none') closeDocViewer();
  });

  // Hover any annotated paragraph/heading → show the model's decision.
  const content = document.getElementById('doc-modal-content');
  content.addEventListener('mouseover', (e) => {
    const el = e.target.closest('[data-vid]');
    if (el) showDecisionTooltip(e, el.dataset.vid);
    else getTooltip().style.display = 'none';
  });
  content.addEventListener('mousemove', (e) => {
    if (getTooltip().style.display === 'block') positionTooltip(e);
  });
  content.addEventListener('mouseleave', () => { getTooltip().style.display = 'none'; });

  // Click a rejected/pruned element → on-demand grounded "why not" explanation.
  // Click a retrieved element with a provenance pin → jump to its page/bbox
  // in the source PDF.
  content.addEventListener('click', (e) => {
    if (e.target.closest('.pin-chip')) return;   // chip has its own handler
    const el = e.target.closest('[data-vid]');
    if (!el) return;
    const stem = state.docViewerStem; if (!stem) return;
    const v = getDocVerdicts(stem);
    const verdict = verdictFor(el.dataset.vid, v);
    if (verdict === 'rejected' || verdict === 'pruned') {
      requestExplanation(stem, el.dataset.vid, el);
    } else if (verdict === 'accepted') {
      const pin = findNodePin(stem, el.dataset.vid);
      if (pin?.page) openSourceView(stem, pin);
    }
  });
}

// ── On-demand "why not selected" explanation ─────────────────
function getExplainPopover() {
  let p = document.getElementById('doc-explain-popover');
  if (!p) {
    p = document.createElement('div');
    p.id = 'doc-explain-popover';
    p.style.display = 'none';
    document.body.appendChild(p);
    // Dismiss on outside click / Esc.
    document.addEventListener('mousedown', (e) => {
      if (p.style.display !== 'none' && !p.contains(e.target) && !e.target.closest('[data-vid]')) {
        p.style.display = 'none';
      }
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') p.style.display = 'none'; });
  }
  return p;
}

function positionPopover(anchorEl) {
  const p = getExplainPopover();
  const r = anchorEl.getBoundingClientRect();
  const pw = p.offsetWidth || 320;
  let x = Math.min(r.left, window.innerWidth - pw - 16);
  let y = r.bottom + 8;
  if (y + (p.offsetHeight || 120) > window.innerHeight) y = Math.max(8, r.top - (p.offsetHeight || 120) - 8);
  p.style.left = `${Math.max(8, x)}px`;
  p.style.top  = `${y}px`;
}

function renderExplainPopover(anchorEl, data) {
  const p = getExplainPopover();
  let html = `<div class="ep-head">Why not selected</div>`;
  if (data.loading) {
    html += `<div class="ep-loading"><span class="ep-spinner"></span> Asking the model (dedicated instance)…</div>`;
  } else if (data.error) {
    html += `<div class="ep-reason" style="color:var(--orange)">Could not get an explanation: ${escHtml(data.error)}</div>`;
  } else {
    if (data.topic)  html += `<div class="ep-topic"><span>Topic</span> ${escHtml(data.topic)}</div>`;
    if (data.reason) html += `<div class="ep-reason">${escHtml(data.reason)}</div>`;
    if (data.addresses_query) {
      html += `<div class="ep-flag">⚠ On re-read the model now thinks this <b>does</b> address the query — worth a manual check.</div>`;
    }
    if (!data.topic && !data.reason) html += `<div class="ep-reason tt-muted">No explanation returned.</div>`;
    if (data.pin?.page) {
      html += `<button class="ep-pin-link">📍 View in source PDF — page ${escHtml(String(data.pin.page))}</button>`;
    }
  }
  p.innerHTML = html;
  const pinLink = p.querySelector('.ep-pin-link');
  if (pinLink) pinLink.addEventListener('click', () => openSourceView(state.docViewerStem, data.pin));
  p.style.display = 'block';
  positionPopover(anchorEl);
}

async function requestExplanation(stem, nodeId, anchorEl) {
  const cache = (state.explainCache[stem] ||= {});
  if (cache[nodeId]) { renderExplainPopover(anchorEl, cache[nodeId]); return; }

  renderExplainPopover(anchorEl, { loading: true });
  try {
    const res = await apiPost('/api/explain', { stem, node_id: nodeId, query: state.currentQuery || '' });
    cache[nodeId] = res;
    renderExplainPopover(anchorEl, res);
  } catch (e) {
    renderExplainPopover(anchorEl, { error: e.message });
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
  initViewTabs();

  if (docs.length) {
    renderCurrentResultView();
  }
  renderSnippets(data);
}

// ── Result view switching: graph (tree) ↔ boxes (treemap) ─────
function initViewTabs() {
  const g = document.getElementById('tab-graph');
  const b = document.getElementById('tab-boxes');
  if (!g || g.dataset.wired) return;
  g.dataset.wired = '1';
  g.addEventListener('click', () => { state.resultView = 'graph'; renderCurrentResultView(); });
  b.addEventListener('click', () => { state.resultView = 'boxes'; renderCurrentResultView(); });
}

function renderCurrentResultView() {
  const graphOn = state.resultView !== 'boxes';
  const treeC = document.getElementById('tree-container');
  const boxC  = document.getElementById('result-treemap');
  if (treeC) treeC.style.display = graphOn ? '' : 'none';
  if (boxC)  boxC.style.display  = graphOn ? 'none' : '';
  document.getElementById('tab-graph')?.classList.toggle('active', graphOn);
  document.getElementById('tab-boxes')?.classList.toggle('active', !graphOn);

  if (!state.currentDoc || !state.currentResults) return;
  if (graphOn) showDocTree(state.currentDoc, state.currentResults);
  else         renderResultBoxes(state.currentDoc, state.currentResults);
}

// Treemap of a single result document, colored by the final verdicts.
function renderResultBoxes(docName, data) {
  const container = document.getElementById('result-treemap');
  if (!container) return;
  container.innerHTML = '';
  const docData = data?.results?.[docName];
  if (!docData) return;

  const retrieved = new Set(docData.retrieved_ids || []);
  const meta = docData.node_meta || {};

  const rect = container.getBoundingClientRect();
  const width  = rect.width  || 800;
  const height = rect.height || 480;

  const rootData = {
    name: docName, nodeId: `__doc__${docName}`, isDoc: true,
    title: docName.replace(/_/g, ' '), children: docData.tree,
  };
  const root = d3.hierarchy(rootData, n => n.children)
    .sum(n => n.isLeaf ? 1 : 0)
    .sort((a, b) => b.value - a.value);
  d3.treemap().size([width, height]).paddingOuter(6).paddingTop(20).paddingInner(3).round(true)(root);

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`);

  const nodes = root.descendants().filter(d => d.depth > 0);
  const g = svg.selectAll('g').data(nodes).join('g')
    .attr('transform', d => `translate(${d.x0},${d.y0})`);

  g.append('rect')
    .attr('width',  d => Math.max(0, d.x1 - d.x0))
    .attr('height', d => Math.max(0, d.y1 - d.y0))
    .attr('class', d => `tm-rect ${d.data.isLeaf ? 'tm-leaf' : ''} ${resultBoxClass(d.data, retrieved, meta)}`)
    .style('cursor', 'pointer')
    .on('mouseover', (event, d) => {
      const id = d.data.nodeId; const m = meta[id] || {};
      let label = d.data.isLeaf ? 'Pending' : 'Section', cls = 'tt-muted';
      if (retrieved.has(id) || m.status === 'retrieved') { label = '✓ Retrieved'; cls = 'tt-selected'; }
      else if (m.status === 'rejected') { label = '✗ Evaluated & Rejected'; cls = 'tt-rejected'; }
      else if (m.status === 'pruned')   { label = '⊘ Pruned'; cls = 'tt-pruned-label'; }
      else if (m.status === 'kept')     { label = '☉ Section kept'; cls = ''; }
      let html = `<div class="tt-title">${escHtml(d.data.title || id)}</div>`;
      html += `<div class="tt-label ${cls}">${label}</div>`;
      const txt = m.reason || d.data.summary;
      if (txt) html += `<div class="tt-reason tt-muted">${escHtml(txt)}</div>`;
      if (m.quote) html += `<div class="tt-quote" style="margin-top:6px;font-style:italic;border-left:2px solid var(--green);padding-left:6px;color:#cbd5e1;">"${escHtml(m.quote)}"</div>`;
      const tip = getTooltip(); tip.innerHTML = html; tip.style.display = 'block'; positionTooltip(event);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', () => { getTooltip().style.display = 'none'; })
    .on('click', (event, d) => {
      getTooltip().style.display = 'none';
      openDocViewer(docName, d.data.isDoc ? null : (d.data.title || null));
    });

  g.filter(d => d.data.isDoc).append('text')
    .attr('class', 'tm-doc-label').attr('x', 6).attr('y', 14)
    .text(d => d.data.title);

  g.filter(d => d.data.isLeaf).append('text')
    .attr('class', 'tm-label-text').attr('x', 4).attr('y', 14)
    .text(d => {
      const w = d.x1 - d.x0, h = d.y1 - d.y0;
      const maxChars = Math.floor((w - 8) / 6.5);
      if (maxChars < 4 || h < 18) return '';
      const l = d.data.title || d.data.nodeId || '';
      return l.length > maxChars ? l.slice(0, maxChars - 1) + '…' : l;
    });
}

function resultBoxClass(nodeData, retrieved, meta) {
  const id = nodeData.nodeId; const m = meta[id] || {};
  if (retrieved.has(id) || m.status === 'retrieved') return 'tm-retrieved';
  if (m.status === 'rejected') return 'tm-rejected';
  if (m.status === 'pruned')   return 'tm-pruned';
  return 'tm-pending';
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

    // "Read" affordance — opens the rendered markdown with verdict highlights.
    const readBtn = el('button', 'doc-read-btn', '⤢ Read');
    readBtn.title = 'Open the rendered document with the model\'s verdict highlights';
    readBtn.addEventListener('click', (e) => {
      e.stopPropagation();              // don't trigger the card's tree switch
      openDocViewer(docName);
    });
    card.appendChild(readBtn);

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
      renderCurrentResultView();
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

      // Provenance pin badge — click jumps to the chunk's page/bbox in the
      // source PDF (pins exist for PDF-ingested documents).
      if (node.pin?.page) {
        const pinBadge = badge('pin-tag', `📍 p.${node.pin.page}`);
        pinBadge.title = 'View this passage in the source PDF';
        pinBadge.style.cursor = 'pointer';
        pinBadge.addEventListener('click', () => openSourceView(docName, node.pin));
        tags.appendChild(pinBadge);
      }

      const title   = el('div', 'snippet-title', node.title);
      const content = el('div', 'snippet-content');
      content.innerHTML = highlightRelevantContent(node.content || '(no content)', node.quote || '');

      card.appendChild(tags);
      card.appendChild(title);

      // Asset chunks: the retrieval decision read the caption; show the crop
      // alongside it so the evidence is visible.
      if (node.pin?.kind === 'asset' && node.pin.image) {
        const fig = el('div', 'snippet-asset');
        const img = document.createElement('img');
        img.src = node.pin.image;
        img.alt = node.title;
        img.loading = 'lazy';
        img.addEventListener('click', () => openSourceView(docName, node.pin));
        fig.appendChild(img);
        card.appendChild(fig);
      }
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

// ── Ingest setup (PDF folder picker for source-based ingest modules) ──
// A module whose MODULE_INFO carries source:"pdf_folder" (betteringest_pdf)
// needs a flat folder of PDFs before it can run. Selecting it opens this
// modal: pick a folder (native Tauri dialog when available), then run
// ingest + re-index in the backend with live progress.
function ingestModuleNeedsSource(name) {
  return state.ingestModules[name]?.source === 'pdf_folder';
}

function initIngestSetup() {
  const sel = document.getElementById('sel-ingest');
  if (!sel) return;
  sel.addEventListener('change', () => {
    if (ingestModuleNeedsSource(sel.value)) openIngestSetup(sel.value);
  });
  // First popup: if the app starts with a source-based module already active,
  // ask for the folder right away (unless one is configured and valid).
  if (ingestModuleNeedsSource(sel.value)) {
    apiGet(`/api/ingest/source?module=${encodeURIComponent(sel.value)}`)
      .then(src => { if (!src.source_ok) openIngestSetup(sel.value); })
      .catch(() => openIngestSetup(sel.value));
  }

  document.getElementById('ingest-modal-close').addEventListener('click', closeIngestSetup);
  document.getElementById('ingest-cancel-btn').addEventListener('click', closeIngestSetup);
  document.querySelector('#ingest-modal .doc-modal-backdrop')
    .addEventListener('click', closeIngestSetup);
  document.addEventListener('keydown', (e) => {
    const modal = document.getElementById('ingest-modal');
    if (e.key === 'Escape' && modal.style.display !== 'none') closeIngestSetup();
  });

  document.getElementById('ingest-browse-btn').addEventListener('click', () => browseForFolder('ingest-source-path'));
  document.getElementById('ingest-source-path').addEventListener('input', () => {
    const val = document.getElementById('ingest-source-path').value;
    const other = document.getElementById('welcome-source-path');
    if (other) other.value = val;
    document.getElementById('ingest-run-btn').disabled = !val.trim();
    const otherBtn = document.getElementById('welcome-run-btn');
    if (otherBtn) otherBtn.disabled = !val.trim();
  });
  document.getElementById('ingest-run-btn').addEventListener('click', () => runIngest('ingest'));

  // Welcome card event listeners
  const welcomeBrowseBtn = document.getElementById('welcome-browse-btn');
  if (welcomeBrowseBtn) {
    welcomeBrowseBtn.addEventListener('click', () => browseForFolder('welcome-source-path'));
  }
  const welcomeSourcePath = document.getElementById('welcome-source-path');
  if (welcomeSourcePath) {
    welcomeSourcePath.addEventListener('input', () => {
      const val = welcomeSourcePath.value;
      const other = document.getElementById('ingest-source-path');
      if (other) other.value = val;
      document.getElementById('welcome-run-btn').disabled = !val.trim();
      const otherBtn = document.getElementById('ingest-run-btn');
      if (otherBtn) otherBtn.disabled = !val.trim();
    });
  }
  const welcomeRunBtn = document.getElementById('welcome-run-btn');
  if (welcomeRunBtn) {
    welcomeRunBtn.addEventListener('click', () => runIngest('welcome'));
  }
}

async function openIngestSetup(moduleName) {
  const modal = document.getElementById('ingest-modal');
  modal.dataset.module = moduleName || document.getElementById('sel-ingest').value;
  document.getElementById('ingest-file-list').innerHTML = '';
  document.getElementById('ingest-warnings').innerHTML = '';
  document.getElementById('ingest-progress').style.display = 'none';
  document.getElementById('ingest-run-btn').textContent = 'Ingest & Index';
  modal.style.display = 'flex';
  try {
    const src = await apiGet(`/api/ingest/source?module=${encodeURIComponent(modal.dataset.module)}`);
    const input = document.getElementById('ingest-source-path');
    if (src.source_dir && !input.value) input.value = src.source_dir;
    document.getElementById('ingest-run-btn').disabled = !input.value.trim();
  } catch { /* fresh setup */ }
}

function closeIngestSetup() {
  document.getElementById('ingest-modal').style.display = 'none';
}

async function browseForFolder(inputId) {
  const input = document.getElementById(inputId || 'ingest-source-path');
  const dialog = window.__TAURI__?.dialog;
  if (dialog?.open) {
    // Native folder dialog (Tauri dialog plugin).
    const picked = await dialog.open({ directory: true, multiple: false,
                                       title: 'Choose a folder of PDFs' });
    if (picked) {
      input.value = Array.isArray(picked) ? picked[0] : picked;
      input.dispatchEvent(new Event('input'));
    }
  } else {
    // Browser fallback: no native dialog — focus the text input.
    input.placeholder = 'No native dialog in browser mode — paste the folder path here';
    input.focus();
  }
}

async function runIngest(prefix) {
  const isWelcome = prefix === 'welcome';
  const sourcePathId = `${prefix}-source-path`;
  const runBtnId = `${prefix}-run-btn`;
  const fileListId = `${prefix}-file-list`;
  const warningsId = `${prefix}-warnings`;
  const progressId = `${prefix}-progress`;
  const progressFillId = `${prefix}-progress-fill`;
  const progressMsgId = `${prefix}-progress-msg`;

  const path    = document.getElementById(sourcePathId).value.trim();
  const runBtn  = document.getElementById(runBtnId);
  const listEl  = document.getElementById(fileListId);
  const warnEl  = document.getElementById(warningsId);
  if (!path) return;

  runBtn.disabled = true;
  warnEl.innerHTML = '';
  try {
    const activeModule = document.getElementById('sel-ingest').value || 'betteringest_pdf';
    const src = await apiPost('/api/ingest/source', { module: activeModule, path });
    listEl.style.display = 'block';
    listEl.innerHTML = `<div class="ingest-file-count">${src.pdf_count} PDF(s):</div>` +
      src.pdfs.map(p => `<div class="ingest-file">${escHtml(p)}</div>`).join('');
    await apiPost('/api/ingest/run', {
      ingest_module: activeModule,
      index_module: document.getElementById('sel-index')?.value || null,
    });
  } catch (e) {
    warnEl.innerHTML = `<div class="ingest-warn">✗ ${escHtml(e.message)}</div>`;
    runBtn.disabled = false;
    return;
  }

  const progWrap = document.getElementById(progressId);
  const fill = document.getElementById(progressFillId);
  const msg  = document.getElementById(progressMsgId);
  progWrap.style.display = 'block';

  const poll = setInterval(async () => {
    let p;
    try { p = await apiGet('/api/ingest/progress'); } catch { return; }
    const pct = p.total ? Math.round((p.done / p.total) * 100) : 0;
    fill.style.width = `${p.phase === 'index' || p.state === 'done' ? 100 : pct}%`;
    msg.textContent = p.message || p.phase || '';
    if (p.state === 'done' || p.state === 'error') {
      clearInterval(poll);
      runBtn.disabled = false;
      if (p.state === 'error') {
        warnEl.innerHTML = `<div class="ingest-warn">✗ ${escHtml(p.message)}</div>`;
        return;
      }
      for (const w of p.warnings || []) {
        warnEl.innerHTML += `<div class="ingest-warn">⚠ ${escHtml(w)}</div>`;
      }
      msg.textContent = `Done — ${p.docs?.length ?? 0} document(s) ingested and indexed.`;
      runBtn.textContent = 'Re-ingest';
      // Refresh the corpus view with the new documents.
      _docsCache = null;
      try {
        const docs = await getDocs();
        renderTreemap(docs);
        if (docs.length > 0) {
          // If we are on the welcome screen and documents were successfully loaded, hide the setup card and show the treemap!
          document.getElementById('treemap-pack').style.display = 'block';
          document.getElementById('treemap-pack-legend').style.display = 'flex';
          document.getElementById('welcome-setup-card').style.display = 'none';
          const toggleBtn = document.getElementById('welcome-toggle-setup-btn');
          if (toggleBtn) {
            toggleBtn.style.display = 'inline-block';
            toggleBtn.textContent = '📁 Configure Source Folder';
          }
          document.getElementById('welcome-status').textContent = "Select a test case from the sidebar or enter a custom query below to start.";
        }
      } catch (e) {
        console.error("refresh error:", e);
      }
      if (!(p.warnings || []).length) {
        if (!isWelcome) setTimeout(closeIngestSetup, 1200);
      }
    }
  }, 500);
}

// ── Provenance pins (page/bbox metadata from PDF ingest) ─────
// Pin blocks are ```pin fenced code in the knowledge_base markdown; marked
// renders them as <pre><code>. decoratePinBlocks() upgrades each into a
// wrapper with a clickable chip — one render path; the header checkbox is
// the single show/hide switch for all of them.
// Look up a node's pin from the cached document trees (loaded at startup /
// after ingest) — used when only a node_id is at hand (doc-viewer clicks).
function findNodePin(stem, nodeId) {
  const doc = (_docsCache || []).find(d => d.name === stem);
  if (!doc) return null;
  let found = null;
  (function walk(nodes) {
    for (const n of nodes || []) {
      if (found) return;
      if (n.nodeId === nodeId) { found = n.pin || null; return; }
      walk(n.children);
    }
  })(doc.tree);
  return found;
}

function parsePinText(text) {
  const pin = {};
  for (const line of (text || '').split('\n')) {
    const i = line.indexOf(':');
    if (i < 0) continue;
    const key = line.slice(0, i).trim();
    let val = line.slice(i + 1).trim();
    if (val.startsWith('[')) { try { val = JSON.parse(val); } catch { /* keep raw */ } }
    pin[key] = val;
  }
  return pin;
}

function decoratePinBlocks(bodyEl, stem) {
  let count = 0;
  bodyEl.querySelectorAll('pre > code').forEach(code => {
    const isPin = /language-pin/.test(code.className) ||
      (/^id: /m.test(code.textContent) && /^kind: (section|asset)$/m.test(code.textContent));
    if (!isPin) return;
    const pin = parsePinText(code.textContent);
    const pre = code.parentElement;
    const wrap = document.createElement('div');
    wrap.className = 'pin-block';
    const chip = document.createElement('button');
    chip.className = 'pin-chip';
    chip.innerHTML = `📍 <b>${escHtml(pin.id || 'pin')}</b> · ${escHtml(pin.kind || '')}` +
      (pin.page ? ` · p.${escHtml(String(pin.page))}` : '') +
      (pin.page ? ` <span class="pin-chip-hint">view in PDF ↗</span>` : '');
    chip.addEventListener('click', () => { if (pin.page) openSourceView(stem, pin); });
    pre.replaceWith(wrap);
    wrap.appendChild(chip);
    wrap.appendChild(pre);
    count++;
  });
  return count;
}

function initPinToggle(bodyEl, pinCount) {
  const toggle = document.getElementById('doc-pins-toggle');
  const box = document.getElementById('doc-pins-checkbox');
  if (!toggle || !box) return;
  toggle.style.display = pinCount > 0 ? '' : 'none';
  box.checked = state.showPins;
  bodyEl.classList.toggle('pins-visible', state.showPins);
  box.onchange = () => {
    state.showPins = box.checked;
    bodyEl.classList.toggle('pins-visible', state.showPins);
  };
}

// ── Source view: the pin's page rendered from the source PDF ─
function getSourcePopover() {
  let p = document.getElementById('source-popover');
  if (!p) {
    p = document.createElement('div');
    p.id = 'source-popover';
    p.style.display = 'none';
    p.innerHTML = `
      <div class="sp-head">
        <span id="sp-title">Source</span>
        <button id="sp-close" title="Close">✕</button>
      </div>
      <div class="sp-body"><img id="sp-img" alt="source page"></div>`;
    document.body.appendChild(p);
    p.querySelector('#sp-close').addEventListener('click', () => { p.style.display = 'none'; });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') p.style.display = 'none';
    });
  }
  return p;
}

// Open the source-PDF page for a pin, with its bbox highlighted — the
// "jump to its exact spot in the PDF" affordance.
function openSourceView(stem, pin) {
  if (!pin || !pin.page) return;
  const p = getSourcePopover();
  const img = p.querySelector('#sp-img');
  const params = new URLSearchParams();
  if (Array.isArray(pin.bbox)) params.set('bbox', pin.bbox.join(','));
  if (Array.isArray(pin.regions)) params.set('regions', JSON.stringify(pin.regions));
  p.querySelector('#sp-title').textContent =
    `${(pin.doc || stem).replace(/_/g, ' ')} — page ${pin.page}` +
    (pin.kind === 'asset' ? ` · ${pin.asset || 'asset'}` : '');
  img.src = '';
  p.classList.add('sp-loading');
  img.onload = () => p.classList.remove('sp-loading');
  img.onerror = () => {
    p.classList.remove('sp-loading');
    p.querySelector('#sp-title').textContent += ' — source PDF unavailable';
  };
  img.src = `/api/document/${encodeURIComponent(pin.doc || stem)}/page/${pin.page}?${params}`;
  p.style.display = 'flex';
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
