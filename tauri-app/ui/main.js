'use strict';

/* ═══════════════════════════════════════════════════════════════════
   PageIndex Explorer — frontend
   One sequential workflow:  ① Library  →  ② Ask  →  ③ Review.

   Sections
     1. state + persistence          6. library (treemap, filter, live)
     2. api + toasts                 7. results (cards, graph, boxes)
     3. init                         8. evidence (snippets)
     4. workflow (views, topbar)     9. reader (doc viewer, explain, pins)
     5. command bar (ask)           10. corpus setup (ingest) + source view
   ═══════════════════════════════════════════════════════════════════ */

/* ── 1 · State ─────────────────────────────────────────────── */

const prefs = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(`pix.${key}`);
      return v === null ? fallback : JSON.parse(v);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`pix.${key}`, JSON.stringify(value)); } catch { /* private mode */ }
  },
};

const state = {
  tests: [],
  currentTestId: null,
  currentResults: null,
  currentDoc: null,
  currentQuery: '',
  running: false,
  view: 'library',                       // 'library' | 'results'
  resultView: 'graph',                   // 'graph' | 'boxes'
  nodeSpacing: prefs.get('nodeSpacing', 36),
  showPins: prefs.get('showPins', false),
  docViewerStem: null,
  liveStatus: null,
  explainCache: {},                      // {stem: {node_id: explanation}}
  ingestModules: {},                     // {name: MODULE_INFO}
  moduleDescs: {},                       // {'stage/name': description}
};

let _docsCache = null;
async function getDocs() {
  if (!_docsCache) _docsCache = await apiGet('/api/documents');
  return _docsCache;
}

/* ── 2 · API + toasts ──────────────────────────────────────── */

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
  if (!r.ok) {
    let detail = '';
    try { detail = (await r.json()).error || ''; } catch { /* not json */ }
    throw new Error(detail || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

function toast(message, kind = 'ok', ms = 4200) {
  const t = el('div', `toast toast-${kind}`, message);
  document.getElementById('toasts').appendChild(t);
  setTimeout(() => { t.classList.add('toast-out'); setTimeout(() => t.remove(), 350); }, ms);
}

/* ── 3 · Init ──────────────────────────────────────────────── */

async function init() {
  let docs = [];
  try {
    const [tests, modules, config, status, fetchedDocs] = await Promise.all([
      apiGet('/api/tests'), apiGet('/api/modules'), apiGet('/api/config'),
      apiGet('/api/status'), getDocs(),
    ]);
    state.tests = tests;
    docs = fetchedDocs;
    initSettings(modules, config);
    renderInstanceDots(status);
  } catch (e) {
    console.error('init error:', e);
    toast(`Could not reach the backend: ${e.message}`, 'err', 8000);
  }

  initWorkflow();
  initCommandBar();
  initLibrary(docs);
  initResultsChrome();
  initDocViewer();
  initSetupCard(docs);
  setView('library');
}

/* ── 4 · Workflow: views + topbar ──────────────────────────── */

function setView(name) {
  state.view = name;
  document.getElementById('view-library').hidden = name !== 'library';
  document.getElementById('view-results').hidden = name !== 'results';
  if (name === 'library' && !state.running) resetLibraryStatus();
  updateFlowSteps();
}

// The three steps light up with where you are; each is a real button.
function updateFlowSteps() {
  const map = {
    library: state.view === 'library' && !state.running,
    ask: state.running || !document.getElementById('command-bar').classList.contains('collapsed'),
    review: state.view === 'results',
  };
  document.querySelectorAll('.flow-step').forEach(btn => {
    const step = btn.dataset.step;
    btn.classList.toggle('active', !!map[step]);
    btn.classList.toggle('done',
      (step === 'library' && (state.view === 'results' || state.running)) ||
      (step === 'ask' && state.view === 'results' && !state.running));
  });
}

function initWorkflow() {
  document.querySelectorAll('.flow-step').forEach(btn => {
    btn.addEventListener('click', () => {
      const step = btn.dataset.step;
      if (step === 'library') setView('library');
      else if (step === 'ask') setChatExpanded(true);
      else if (step === 'review') {
        if (state.currentResults) setView('results');
        else toast('No results yet — ask a question first.', 'warn');
      }
    });
  });
  document.getElementById('back-to-library').addEventListener('click', () => setView('library'));
  document.getElementById('rerun-btn').addEventListener('click', () => {
    if (state.currentTestId) runTest(state.currentTestId);
    else if (state.currentQuery) runQuery(state.currentQuery);
  });

  // Esc walks backwards through the workflow when nothing else consumes it.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (anyOverlayOpen()) return;                        // overlays handle their own Esc
    if (!document.getElementById('command-bar').classList.contains('collapsed')) return;
    if (state.view === 'results') setView('library');
  });
}

function anyOverlayOpen() {
  return !document.getElementById('doc-modal').hidden ||
    document.getElementById('source-popover')?.style.display === 'flex' ||
    document.getElementById('doc-explain-popover')?.style.display === 'block' ||
    !document.getElementById('settings-popover').hidden ||
    !document.getElementById('display-popover').hidden;
}

function setRunPill(text) {
  const pill = document.getElementById('run-pill');
  pill.hidden = !text;
  if (text) document.getElementById('run-pill-text').textContent = text;
}

/* ── Popover plumbing (settings ⚙ / display Aa) ─────────────── */

function attachPopover(btnId, popId, place) {
  const btn = document.getElementById(btnId);
  const pop = document.getElementById(popId);
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const open = pop.hidden;
    closeAllPopovers();
    if (!open) return;
    pop.hidden = false;
    const r = btn.getBoundingClientRect();
    place(pop, r);
  });
  document.addEventListener('mousedown', (e) => {
    if (!pop.hidden && !pop.contains(e.target) && e.target !== btn && !btn.contains(e.target)) {
      pop.hidden = true;
    }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !pop.hidden) pop.hidden = true;
  });
}

function closeAllPopovers() {
  document.getElementById('settings-popover').hidden = true;
  document.getElementById('display-popover').hidden = true;
}

/* ── Settings: pipeline modules · corpus source · engine ────── */

function initSettings(modulesData, config) {
  const stageMap = { ingest: 'sel-ingest', index: 'sel-index', query: 'sel-query' };
  for (const [stage, { default: def, modules }] of Object.entries(modulesData.stages || {})) {
    const sel = document.getElementById(stageMap[stage]);
    if (!sel) continue;
    sel.innerHTML = '';
    for (const mod of modules) {
      state.moduleDescs[`${stage}/${mod.name}`] = mod.description || '';
      if (stage === 'ingest') state.ingestModules[mod.name] = mod;
      const opt = document.createElement('option');
      opt.value = mod.name;
      opt.textContent = mod.label || mod.name;
      opt.title = mod.description || '';
      if (mod.name === def) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', () => updateModuleDesc(stage, sel.value));
  }

  attachPopover('settings-btn', 'settings-popover', (pop, r) => {
    pop.style.top = `${r.bottom + 8}px`;
    pop.style.right = '10px';
    pop.style.left = 'auto';
    refreshSourceSection();
  });
  attachPopover('display-btn', 'display-popover', (pop, r) => {
    pop.style.top = `${r.bottom + 8}px`;
    pop.style.right = '14px';
    pop.style.left = 'auto';
  });

  // ⌘, opens settings — the platform-native place for it.
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === ',') {
      e.preventDefault();
      document.getElementById('settings-btn').click();
    }
  });

  // Ingest module choice drives the corpus-source affordances.
  document.getElementById('sel-ingest').addEventListener('change', async () => {
    updateSourceAffordances();
    refreshSourceSection();
    const name = document.getElementById('sel-ingest').value;
    if (ingestModuleNeedsSource(name)) {
      try {
        const src = await apiGet(`/api/ingest/source?module=${encodeURIComponent(name)}`);
        if (!src.source_ok) { closeAllPopovers(); openSetupCard(); }
      } catch { closeAllPopovers(); openSetupCard(); }
    }
  });
  document.getElementById('settings-source-btn').addEventListener('click', () => {
    closeAllPopovers();
    setView('library');
    openSetupCard();
  });

  initInstanceStepper(config);
  updateSourceAffordances();
  updateModuleDesc('ingest', document.getElementById('sel-ingest').value);
}

function updateModuleDesc(stage, name) {
  const p = document.getElementById('module-desc');
  p.textContent = state.moduleDescs[`${stage}/${name}`] || '';
}

function selectedModules() {
  return {
    ingest_module: document.getElementById('sel-ingest')?.value || null,
    index_module:  document.getElementById('sel-index')?.value  || null,
    query_module:  document.getElementById('sel-query')?.value  || null,
  };
}

function ingestModuleNeedsSource(name) {
  return state.ingestModules[name]?.source === 'pdf_folder';
}

function updateSourceAffordances() {
  const needs = ingestModuleNeedsSource(document.getElementById('sel-ingest').value);
  document.getElementById('settings-source-section').hidden = !needs;
  document.getElementById('library-source-btn').hidden = !needs;
}

async function refreshSourceSection() {
  const pathEl = document.getElementById('settings-source-path');
  try {
    const name = document.getElementById('sel-ingest').value;
    if (!ingestModuleNeedsSource(name)) return;
    const src = await apiGet(`/api/ingest/source?module=${encodeURIComponent(name)}`);
    pathEl.textContent = src.source_dir || 'No folder selected';
    pathEl.title = src.source_dir || '';
  } catch { pathEl.textContent = 'No folder selected'; }
}

/* ── Engine: Ollama instance stepper + live dots ────────────── */

let _statusPoller = null;

function initInstanceStepper(config) {
  let count = config?.ollama_instances || 1;
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
    renderInstanceDots(null);
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
    container.querySelectorAll('.inst-dot').forEach(d => {
      d.className = 'inst-dot'; d.title = 'idle';
    });
    return;
  }
  if (container.children.length !== instances.length) {
    container.innerHTML = '';
    instances.forEach(() => container.appendChild(el('div', 'inst-dot')));
  }
  const dots = container.querySelectorAll('.inst-dot');
  instances.forEach((inst, i) => {
    const busy = inst.active > 0;
    dots[i].className = `inst-dot ${busy ? 'busy' : ''}`;
    dots[i].title = busy
      ? `Instance ${i + 1} — ${inst.active} leaf${inst.active > 1 ? 's' : ''} active`
      : `Instance ${i + 1} — idle`;
  });

  const prog = statusData?.progress;
  if (state.running && prog?.total > 0) {
    setRunPill(`Evaluating ${prog.done} / ${prog.total}`);
    document.getElementById('library-status').textContent =
      `Reading your documents — ${prog.done} of ${prog.total} passages evaluated…`;
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
    } catch { /* transient */ }
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
  } catch { renderInstanceDots(null); }
}

/* ── 5 · Command bar (Ask) ─────────────────────────────────── */

function setChatExpanded(expanded) {
  const container = document.getElementById('command-bar');
  const input = document.getElementById('main-query-input');
  container.classList.toggle('collapsed', !expanded);
  if (expanded) {
    requestAnimationFrame(() => { input.focus(); autoGrowTextarea(input); renderSuggestions(); });
  } else {
    input.blur();
  }
  updateFlowSteps();
}

function autoGrowTextarea(ta) {
  ta.style.height = 'auto';
  const cs = getComputedStyle(ta);
  const line = parseFloat(cs.lineHeight) || 19;
  const padY = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
  const maxH = line * 10 + padY;
  ta.style.height = Math.min(ta.scrollHeight, maxH) + 'px';
  ta.style.overflowY = ta.scrollHeight > maxH ? 'auto' : 'hidden';
}

// Example queries live inside the command bar: visible when it's empty,
// filtered as you type, arrow-key navigable. (They replace the old sidebar.)
let _sugFocus = -1;

function renderSuggestions() {
  const box = document.getElementById('suggestions');
  const input = document.getElementById('main-query-input');
  const needle = input.value.trim().toLowerCase();
  const cats = [
    { key: 'SINGLE', label: 'Single document' },
    { key: 'MULTI',  label: 'Multi-section' },
    { key: 'CROSS',  label: 'Cross-document' },
  ];
  box.innerHTML = '';
  _sugFocus = -1;
  let total = 0;
  for (const { key, label } of cats) {
    const group = state.tests.filter(t => t.category === key &&
      (!needle || t.description.toLowerCase().includes(needle) ||
       t.query.toLowerCase().includes(needle)));
    if (!group.length) continue;
    box.appendChild(el('div', 'sug-group-label', label));
    for (const t of group) {
      const b = el('button', 'sug-item');
      b.dataset.testId = t.id;
      b.appendChild(el('span', 'sug-title', t.description));
      b.appendChild(el('span', 'sug-query', t.query));
      b.addEventListener('click', () => { setChatExpanded(false); runTest(t.id); });
      box.appendChild(b);
      total++;
    }
  }
  box.hidden = total === 0 || (!!needle && input.value.length > 60);
}

function moveSuggestionFocus(delta) {
  const items = Array.from(document.querySelectorAll('.sug-item'));
  if (!items.length) return false;
  _sugFocus = Math.max(-1, Math.min(items.length - 1, _sugFocus + delta));
  items.forEach((it, i) => it.classList.toggle('sug-focus', i === _sugFocus));
  if (_sugFocus >= 0) items[_sugFocus].scrollIntoView({ block: 'nearest' });
  return _sugFocus >= 0;
}

function initCommandBar() {
  const container = document.getElementById('command-bar');
  const input     = document.getElementById('main-query-input');

  input.value = prefs.get('lastQuery', '');

  document.getElementById('main-run-btn').addEventListener('click', submitCommandBar);
  document.getElementById('chat-launcher').addEventListener('click', () => setChatExpanded(true));
  document.getElementById('chat-collapse').addEventListener('click', () => setChatExpanded(false));

  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && !document.getElementById('suggestions').hidden) {
      e.preventDefault(); moveSuggestionFocus(1);
    } else if (e.key === 'ArrowUp' && _sugFocus >= 0) {
      e.preventDefault(); moveSuggestionFocus(-1);
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const focused = document.querySelector('.sug-item.sug-focus');
      if (focused) focused.click();
      else submitCommandBar();
    } else if (e.key === 'Escape') {
      e.preventDefault(); setChatExpanded(false);
    }
  });
  input.addEventListener('input', () => {
    autoGrowTextarea(input);
    renderSuggestions();
    prefs.set('lastQuery', input.value);
  });

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

  makeDraggable(container);
}

function submitCommandBar() {
  const query = document.getElementById('main-query-input').value.trim();
  if (!query || state.running) return;
  runQuery(query);
}

/* ── Runs (tests + custom queries) ─────────────────────────── */

async function runTest(testId) {
  const test = state.tests.find(t => t.id === testId);
  state.currentTestId = testId;
  state.currentQuery = test?.query || '';
  const input = document.getElementById('main-query-input');
  input.value = state.currentQuery;
  autoGrowTextarea(input);
  await executeRun({ test_id: testId }, `Running “${test?.description || testId}”…`);
}

async function runQuery(query) {
  state.currentTestId = null;
  state.currentQuery = query;
  await executeRun({ query }, 'Running your question across the library…');
}

async function executeRun(body, message) {
  if (state.running) return;
  state.running = true;
  setChatExpanded(false);
  setView('library');                                  // watch the corpus light up
  setRunPill('Starting…');
  document.getElementById('library-status').textContent = message;
  document.getElementById('main-query-input').disabled = true;
  document.getElementById('main-run-btn').disabled = true;
  try { renderTreemap(await getDocs()); } catch { /* keep old canvas */ }
  startStatusPolling();
  updateFlowSteps();
  try {
    const data = await apiPost('/api/run', { ...body, ...selectedModules() });
    state.currentResults = data;
    renderResults(data);
    setView('results');
  } catch (e) {
    toast(`Run failed: ${e.message}`, 'err', 7000);
    document.getElementById('library-status').textContent = `Something went wrong: ${e.message}`;
  } finally {
    state.running = false;
    setRunPill(null);
    document.getElementById('main-query-input').disabled = false;
    document.getElementById('main-run-btn').disabled = false;
    await stopStatusPolling();
    if (state.view === 'library') resetLibraryStatus();
    updateFlowSteps();
  }
}

/* ── 6 · Library: corpus treemap + filter + live coloring ──── */

let _tmNodeById = {};

function initLibrary(docs) {
  renderTreemap(docs);
  updateLibraryCount(docs);
  resetLibraryStatus();

  document.getElementById('library-source-btn').addEventListener('click', openSetupCard);

  const filter = document.getElementById('library-filter');
  filter.addEventListener('input', () => applyLibraryFilter(filter.value.trim().toLowerCase()));

  window.addEventListener('resize', debounce(async () => {
    if (state.view === 'library' && _docsCache) renderTreemap(_docsCache);
  }, 200));
}

function updateLibraryCount(docs) {
  const leaves = (docs || []).reduce((n, d) => n + (d.leaf_count || 0), 0);
  document.getElementById('library-count').textContent =
    docs?.length ? `${docs.length} document${docs.length > 1 ? 's' : ''} · ${leaves} sections` : '';
}

function resetLibraryStatus() {
  document.getElementById('library-status').textContent = (_docsCache || []).length
    ? 'Click any section to read it — or press ⌘K and ask a question.'
    : 'Your library is empty.';
}

function applyLibraryFilter(needle) {
  for (const [, rect] of Object.entries(_tmNodeById)) {
    if (!rect?.__d) continue;
    const d = rect.__d;
    const hay = `${d.data.title || ''} ${d.data.nodeId || ''}`.toLowerCase();
    rect.classList.toggle('tm-filter-dim', !!needle && !hay.includes(needle));
  }
}

function renderTreemap(docs) {
  const container = document.getElementById('treemap-pack');
  container.innerHTML = '';
  _tmNodeById = {};
  if (!docs?.length) { document.getElementById('treemap-legend').hidden = true; return; }
  document.getElementById('treemap-legend').hidden = false;

  const rect = container.getBoundingClientRect();
  const width = rect.width || 800;
  const height = rect.height || 500;

  const rootData = {
    name: '_root',
    children: docs.map(d => ({
      name: d.name, nodeId: `__doc__${d.name}`, isDoc: true, children: d.tree,
    })),
  };
  const root = d3.hierarchy(rootData, n => n.children)
    .sum(n => n.isLeaf ? 1 : 0)
    .sort((a, b) => b.value - a.value);

  d3.treemap().size([width, height]).paddingOuter(6).paddingTop(20)
    .paddingInner(3).round(true)(root);

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`)
    .style('background', 'transparent');

  const leafG = svg.selectAll('g')
    .data(root.descendants().filter(d => d.depth > 0))
    .join('g')
    .attr('transform', d => `translate(${d.x0},${d.y0})`);

  leafG.append('rect')
    .attr('width', d => d.x1 - d.x0)
    .attr('height', d => d.y1 - d.y0)
    .attr('class', d => `tm-rect tm-pending ${d.data.isLeaf ? 'tm-leaf' : ''}`)
    .style('cursor', 'pointer')
    .each(function (d) {
      this.__d = d;
      if (d.data.nodeId) _tmNodeById[d.data.nodeId] = this;
    })
    .on('mouseover', (event, d) => {
      const id = d.data.nodeId;
      if (!id || id === '_root') return;
      const elx = _tmNodeById[id];
      let statusLabel = 'Pending evaluation', statusClass = 'tt-muted';
      if (elx) {
        if (elx.classList.contains('tm-retrieved')) { statusLabel = '✓ Retrieved'; statusClass = 'tt-selected'; }
        else if (elx.classList.contains('tm-rejected')) { statusLabel = '✗ Evaluated & rejected'; statusClass = 'tt-rejected'; }
        else if (elx.classList.contains('tm-pruned')) { statusLabel = '⊘ Pruned'; statusClass = 'tt-pruned-label'; }
      }
      let html = `<div class="tt-title">${escHtml(d.data.title || id)}</div>`;
      html += `<div class="tt-label ${statusClass}">${statusLabel}</div>`;
      if (!d.data.isLeaf) {
        const label = d.data.isDoc ? 'Document' : `Section · h${d.data.headingLevel || d.depth}`;
        html += `<div class="tt-label tt-section" style="margin-top:4px;">${label}</div>`;
      }
      if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      showTooltip(event, html);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', hideTooltip)
    .on('click', (event, d) => {
      let curr = d;
      while (curr && !curr.data.isDoc) curr = curr.parent;
      if (!curr) return;
      hideTooltip();
      openDocViewer(curr.data.name, d.data.isDoc ? null : (d.data.title || null));
    });

  leafG.filter(d => d.data.isDoc)
    .append('text').attr('class', 'tm-doc-label').attr('x', 6).attr('y', 14)
    .text(d => d.data.name.replace(/_/g, ' '));

  leafG.filter(d => d.data.isLeaf)
    .append('text').attr('class', 'tm-label-text').attr('x', 4).attr('y', 14)
    .text(d => fitLabel(d.data.title || d.data.nodeId || '', d.x1 - d.x0, d.y1 - d.y0));

  // re-apply an active filter across re-renders
  const needle = document.getElementById('library-filter').value.trim().toLowerCase();
  if (needle) applyLibraryFilter(needle);
}

function fitLabel(label, w, h) {
  const maxChars = Math.floor((w - 8) / 6.5);
  if (maxChars < 4 || h < 18) return '';
  return label.length > maxChars ? label.slice(0, maxChars - 1) + '…' : label;
}

function applyTreemapEvents(status) {
  if (!status || !status.live) return;
  const live = status.live;
  const prunedSet = new Set(live.pruned || []);
  const rejectedSet = new Set(live.rejected || []);
  const retrievedSet = new Set(live.retrieved || []);

  for (const [, elx] of Object.entries(_tmNodeById)) {
    if (!elx || !elx.__d) continue;
    let cls = 'tm-pending';
    for (let curr = elx.__d; curr; curr = curr.parent) {
      const id = curr.data.nodeId;
      if (!id) continue;
      if (retrievedSet.has(id)) { cls = 'tm-retrieved'; break; }
      if (rejectedSet.has(id)) cls = 'tm-rejected';
      else if (prunedSet.has(id) && cls === 'tm-pending') cls = 'tm-pruned';
    }
    const isLeaf = elx.classList.contains('tm-leaf');
    elx.setAttribute('class', `tm-rect ${isLeaf ? 'tm-leaf' : ''} ${cls}`);
  }
}

/* ── 7 · Results: query bar, doc cards, graph & boxes ──────── */

function initResultsChrome() {
  document.getElementById('tab-graph').addEventListener('click', () => {
    state.resultView = 'graph'; renderCurrentResultView();
  });
  document.getElementById('tab-boxes').addEventListener('click', () => {
    state.resultView = 'boxes'; renderCurrentResultView();
  });

  // Display popover controls
  document.getElementById('ctrl-tab-size').addEventListener('input', e => {
    document.querySelectorAll('.stat-card').forEach(c => { c.style.minWidth = `${e.target.value}px`; });
  });
  const spacing = document.getElementById('ctrl-node-spacing');
  spacing.value = state.nodeSpacing;
  spacing.addEventListener('input', e => {
    state.nodeSpacing = parseInt(e.target.value);
    prefs.set('nodeSpacing', state.nodeSpacing);
    if (state.currentDoc && state.currentResults) showDocTree(state.currentDoc, state.currentResults);
  });
  const snip = document.getElementById('ctrl-toggle-snippets');
  snip.checked = prefs.get('snippetsVisible', true);
  document.getElementById('snippets-panel').hidden = !snip.checked;
  snip.addEventListener('change', () => {
    prefs.set('snippetsVisible', snip.checked);
    document.getElementById('snippets-panel').hidden = !snip.checked;
  });
}

function renderResults(data) {
  renderStatusBar(data);
  const docs = Object.keys(data.results);
  if (docs.length) state.currentDoc = docs[0];
  renderStatsBar(data);
  renderCurrentResultView();
  renderSnippets(data);
}

function renderStatusBar(data) {
  const q = document.getElementById('query-display');
  q.textContent = data.query;
  q.title = data.query;

  const elx = document.getElementById('test-status');
  elx.innerHTML = '';

  if (data.pipeline) {
    const wrap = el('div', 'pipeline-badge');
    const stageLabels = { ingest: '①', index: '②', query: '③' };
    for (const [stage, name] of Object.entries(data.pipeline)) {
      const chip = el('span', 'pipeline-chip');
      chip.innerHTML = `${stageLabels[stage] || stage} <span>${escHtml(name)}</span>`;
      chip.title = `${stage}: ${name}`;
      wrap.appendChild(chip);
    }
    elx.appendChild(wrap);
  }

  if (!data.test_result) {
    elx.appendChild(el('span', 'status-badge neutral', 'Custom query'));
    return;
  }
  const r = data.test_result;
  if (r.passed) {
    elx.appendChild(el('span', 'status-badge pass', '✓ PASS'));
  } else {
    const issues = [];
    if (Object.keys(r.missing).length)     issues.push('required missing');
    if (Object.keys(r.any_missing).length) issues.push('any-of missing');
    elx.appendChild(el('span', 'status-badge fail', `✗ FAIL — ${issues.join('; ')}`));
  }
}

function renderStatsBar(data) {
  const bar = document.getElementById('stats-bar');
  bar.innerHTML = '';
  const tabSize = document.getElementById('ctrl-tab-size')?.value;

  for (const docName of Object.keys(data.results)) {
    const docData   = data.results[docName];
    const total     = countLeaves(docData.tree);
    const retrieved = docData.retrieved_ids.length;
    const pct       = total ? Math.round((retrieved / total) * 100) : 0;

    const card = el('div', 'stat-card');
    card.dataset.doc = docName;
    card.innerHTML = `
      <div class="stat-doc-name">${escHtml(docName.replace(/_/g, ' '))}</div>
      <div class="stat-bar-wrap"><div class="stat-bar-fill" style="width:${pct}%"></div></div>
      <div class="stat-count">${retrieved} / ${total} sections (${pct}%)</div>`;

    const readBtn = el('button', 'doc-read-btn', '⤢ Read');
    readBtn.title = 'Open the document with the model’s verdicts overlaid';
    readBtn.addEventListener('click', (e) => { e.stopPropagation(); openDocViewer(docName); });
    card.appendChild(readBtn);

    if (tabSize) card.style.minWidth = `${tabSize}px`;
    if (state.currentDoc === docName) card.classList.add('active');
    card.addEventListener('click', () => {
      document.querySelectorAll('.stat-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      state.currentDoc = docName;
      renderCurrentResultView();
    });
    bar.appendChild(card);
  }
}

function countLeaves(tree) {
  let n = 0;
  (function walk(nodes) { for (const nd of nodes) { nd.isLeaf ? n++ : walk(nd.children || []); } })(tree);
  return n;
}

function renderCurrentResultView() {
  const graphOn = state.resultView !== 'boxes';
  document.getElementById('tree-container').style.display = graphOn ? '' : 'none';
  document.getElementById('result-treemap').hidden = graphOn;
  document.getElementById('tab-graph').classList.toggle('active', graphOn);
  document.getElementById('tab-boxes').classList.toggle('active', !graphOn);
  if (!state.currentDoc || !state.currentResults) return;
  if (graphOn) showDocTree(state.currentDoc, state.currentResults);
  else         renderResultBoxes(state.currentDoc, state.currentResults);
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

/* ── D3 tree (Graph tab) — verdict colors shared with style.css ── */

const STATUS_COLOR = {
  internal:        '#3b82f6',
  'expected-hit':  '#22c55e',
  retrieved:       '#86efac',
  'expected-miss': '#ef4444',
  kept:            '#0ea5e9',
  pruned:          '#64748b',
  rejected:        '#f97316',
  neutral:         '#374151',
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
  const got = retrievedSet.has(id);
  const exp = expectedSet.has(id) || expectedAnySet.has(id);
  if (got && exp) return 'expected-hit';
  if (exp) return 'expected-miss';
  if (got) return 'retrieved';
  if (meta.status === 'kept') return 'kept';
  if (meta.status === 'pruned') return 'pruned';
  if (meta.status === 'rejected') return 'rejected';
  if (!d.data.isLeaf) return 'internal';
  return 'neutral';
}

function renderTree(rawTree, retrievedSet, expectedSet, expectedAnySet, nodeReasons = {}) {
  const container = document.getElementById('tree-container');
  container.innerHTML = '';

  const treeData = rawTree.length === 1
    ? rawTree[0]
    : { nodeId: '_root', title: 'root', isLeaf: false, synthetic: false, children: rawTree };

  const root = d3.hierarchy(treeData, d =>
    (d.children && d.children.length) ? d.children : null);

  const spacingV = state.nodeSpacing || 36;
  const spacingH = 240;
  const mT = 24, mR = 200, mB = 24, mL = 12;

  d3.tree().nodeSize([spacingV, spacingH])(root);

  let minX = Infinity, maxX = -Infinity, maxY = -Infinity;
  root.each(d => {
    if (d.x < minX) minX = d.x;
    if (d.x > maxX) maxX = d.x;
    if (d.y > maxY) maxY = d.y;
  });

  const svg = d3.select(container).append('svg')
    .attr('width', maxY + spacingH + mL + mR)
    .attr('height', Math.max((maxX - minX) + mT + mB, 120))
    .style('display', 'block');

  const g = svg.append('g').attr('transform', `translate(${mL},${mT - minX})`);
  svg.call(d3.zoom().scaleExtent([0.2, 3]).on('zoom', e => g.attr('transform', e.transform)));

  function leadsToRetrieved(node) {
    let found = false;
    node.each(d => { if (retrievedSet.has(d.data.nodeId)) found = true; });
    return found;
  }

  const linkGroup = g.append('g').selectAll('g.link-group')
    .data(root.links()).join('g').attr('class', 'link-group');

  linkGroup.append('path')
    .attr('fill', 'none')
    .attr('stroke', '#334155')
    .attr('stroke-width', 1.5)
    .attr('class', d => leadsToRetrieved(d.target) ? 'link-highlight' : '')
    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));

  // Hovering an edge shows the full root→node reasoning chain (the audit trail).
  linkGroup.append('path')
    .attr('class', 'link-hover-catcher')
    .attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x))
    .on('mouseover', (event, d) => {
      const pathNodes = d.target.ancestors().reverse();
      let html = `<div class="tt-label tt-section" style="margin-bottom:8px;border-bottom:1px solid rgba(148,163,184,0.2);padding-bottom:4px;font-weight:700;">Path reasoning</div>`;
      html += `<div style="display:flex;flex-direction:column;gap:8px;">`;
      pathNodes.forEach((nd, index) => {
        const id = nd.data.nodeId;
        const meta = nodeReasons[id] || {};
        const title = nd.data.title || id || 'Document';
        let badge = nd.data.isLeaf ? 'Leaf' : 'Section', color = '#94a3b8';
        if (meta.status === 'retrieved' || retrievedSet.has(id)) { badge = 'Retrieved'; color = '#86efac'; }
        else if (meta.status === 'kept') { badge = 'Kept'; color = '#38bdf8'; }
        else if (meta.status === 'pruned') { badge = 'Pruned'; color = '#64748b'; }
        else if (meta.status === 'rejected') { badge = 'Rejected'; color = '#f97316'; }
        html += `<div style="font-size:11px;">
          <div style="font-weight:600;display:flex;justify-content:space-between;gap:10px;">
            <span>${index + 1}. ${escHtml(title.length > 24 ? title.slice(0, 24) + '…' : title)}</span>
            <span style="font-size:9px;text-transform:uppercase;font-weight:700;color:${color}">${badge}</span>
          </div>`;
        const note = meta.reason || nd.data.summary;
        if (note) html += `<div style="color:#8494ab;font-size:10px;margin-top:2px;line-height:1.4;padding-left:8px;border-left:1.5px solid rgba(148,163,184,0.2);">${escHtml(note)}</div>`;
        html += `</div>`;
      });
      html += `</div>`;
      showTooltip(event, html);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', hideTooltip);

  const nodeG = g.append('g').selectAll('g')
    .data(root.descendants()).join('g')
    .attr('transform', d => `translate(${d.y},${d.x})`);

  nodeG.append('circle')
    .attr('r', d => d.data.isLeaf ? 7 : 5)
    .attr('fill',   d => STATUS_COLOR[nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons)])
    .attr('stroke', d => STATUS_STROKE[nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons)])
    .attr('stroke-width', 1.5)
    .style('cursor', 'pointer')
    .on('click', (event, d) => { if (d.data.isLeaf) highlightSnippet(d.data.nodeId); })
    .on('mouseover', (event, d) => {
      const id   = d.data.nodeId;
      const meta = nodeReasons[id] || {};
      let html = `<div class="tt-title">${escHtml(d.data.title || id || 'Document')}</div>`;
      const withReason = (label, cls) => {
        html += `<div class="tt-label ${cls}">${label}</div>`;
        if (meta.reason) html += `<div class="tt-reason">${escHtml(meta.reason)}</div>`;
        else if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      };
      if (meta.status === 'retrieved') {
        html += `<div class="tt-label tt-selected">✓ Selected by the model</div>`;
        if (meta.reason) html += `<div class="tt-reason">${escHtml(meta.reason)}</div>`;
        if (meta.quote)  html += `<div class="tt-quote">“${escHtml(meta.quote)}”</div>`;
      } else if (meta.status === 'rejected') withReason('✗ Evaluated & rejected', 'tt-rejected');
      else if (meta.status === 'pruned')     withReason('⊘ Pruned (not evaluated)', 'tt-pruned-label');
      else if (meta.status === 'kept')       withReason('☉ Section kept (passed pruning)', 'tt-kept-label');
      else if (d.data.isLeaf) {
        html += `<div class="tt-label tt-not-selected">Not retrieved</div>`;
        if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      } else {
        const label = d.depth === 0 ? 'Document' : `Section · h${d.data.headingLevel || d.depth}`;
        html += `<div class="tt-label tt-section">${label}</div>`;
        if (d.data.summary) html += `<div class="tt-reason tt-muted">${escHtml(d.data.summary)}</div>`;
      }
      showTooltip(event, html);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', hideTooltip);

  nodeG.filter(d => d.data.isLeaf)
    .append('text')
    .attr('text-anchor', 'middle').attr('dominant-baseline', 'central')
    .attr('font-size', '8px').attr('fill', '#fff').attr('pointer-events', 'none')
    .text(d => STATUS_ICON[nodeStatus(d, retrievedSet, expectedSet, expectedAnySet, nodeReasons)] || '');

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

  nodeG.filter(d => d.data.synthetic)
    .append('text')
    .attr('x', 11).attr('y', -10)
    .attr('font-size', '8px').attr('fill', '#a78bfa').attr('pointer-events', 'none')
    .text('[synth]');

  nodeG.filter(d => !d.data.isLeaf && d.data.headingLevel)
    .append('text')
    .attr('x', -11).attr('y', 13)
    .attr('text-anchor', 'end')
    .attr('font-size', '8px').attr('fill', '#475569').attr('pointer-events', 'none')
    .text(d => `h${d.data.headingLevel}`);
}

/* ── Boxes tab: single-document verdict treemap ─────────────── */

function renderResultBoxes(docName, data) {
  const container = document.getElementById('result-treemap');
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
  d3.treemap().size([width, height]).paddingOuter(6).paddingTop(20)
    .paddingInner(3).round(true)(root);

  const svg = d3.select(container).append('svg')
    .attr('width', width).attr('height', height)
    .attr('viewBox', `0 0 ${width} ${height}`);

  const g = svg.selectAll('g').data(root.descendants().filter(d => d.depth > 0)).join('g')
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
      else if (m.status === 'rejected') { label = '✗ Evaluated & rejected'; cls = 'tt-rejected'; }
      else if (m.status === 'pruned')   { label = '⊘ Pruned'; cls = 'tt-pruned-label'; }
      else if (m.status === 'kept')     { label = '☉ Section kept'; cls = 'tt-kept-label'; }
      let html = `<div class="tt-title">${escHtml(d.data.title || id)}</div>`;
      html += `<div class="tt-label ${cls}">${label}</div>`;
      const txt = m.reason || d.data.summary;
      if (txt) html += `<div class="tt-reason tt-muted">${escHtml(txt)}</div>`;
      if (m.quote) html += `<div class="tt-quote">“${escHtml(m.quote)}”</div>`;
      showTooltip(event, html);
    })
    .on('mousemove', positionTooltip)
    .on('mouseout', hideTooltip)
    .on('click', (event, d) => {
      hideTooltip();
      openDocViewer(docName, d.data.isDoc ? null : (d.data.title || null));
    });

  g.filter(d => d.data.isDoc).append('text')
    .attr('class', 'tm-doc-label').attr('x', 6).attr('y', 14)
    .text(d => d.data.title);

  g.filter(d => d.data.isLeaf).append('text')
    .attr('class', 'tm-label-text').attr('x', 4).attr('y', 14)
    .text(d => fitLabel(d.data.title || d.data.nodeId || '', d.x1 - d.x0, d.y1 - d.y0));
}

function resultBoxClass(nodeData, retrieved, meta) {
  const id = nodeData.nodeId; const m = meta[id] || {};
  if (retrieved.has(id) || m.status === 'retrieved') return 'tm-retrieved';
  if (m.status === 'rejected') return 'tm-rejected';
  if (m.status === 'pruned')   return 'tm-pruned';
  return 'tm-pending';
}

/* ── 8 · Evidence (snippets) ───────────────────────────────── */

function renderSnippets(data) {
  const list = document.getElementById('snippets-list');
  list.innerHTML = '';

  let total = 0;
  for (const [docName, docData] of Object.entries(data.results)) {
    if (!docData.nodes.length) continue;
    const tr = data.test_result || {};
    const expSet = new Set([...(tr.expected?.[docName] || []),
                            ...(tr.expected_any?.[docName] || [])]);

    for (const node of docData.nodes) {
      total++;
      const card = el('div', 'snippet-card');
      card.id = `snip-${node.node_id}`;

      const tags = el('div', 'snippet-tags');
      tags.appendChild(badge('doc-tag',   docName.replace(/_/g, ' ')));
      tags.appendChild(badge('id-tag',    node.node_id));
      tags.appendChild(badge('level-tag', `h${node.heading_level}`));
      if (node.synthetic) tags.appendChild(badge('synth-tag', 'synthetic'));
      if (expSet.has(node.node_id)) tags.appendChild(badge('expected-tag', 'expected'));

      // Provenance pin — one click to the exact page/bbox in the source PDF.
      if (node.pin?.page) {
        const pinBadge = badge('pin-tag', `📍 p.${node.pin.page}`);
        pinBadge.title = 'View this passage in the source PDF';
        pinBadge.addEventListener('click', () => openSourceView(docName, node.pin));
        tags.appendChild(pinBadge);
      }

      const title = el('div', 'snippet-title', node.title);
      title.title = 'Open in the reader';
      title.addEventListener('click', () => openDocViewer(docName, node.title));

      const content = el('div', 'snippet-content');
      content.innerHTML = highlightRelevantContent(node.content || '(no content)', node.quote || '');

      card.appendChild(tags);
      card.appendChild(title);

      // Asset chunks: the model read the caption — show the actual crop too.
      if (node.pin?.kind === 'asset' && node.pin.image) {
        const fig = el('div', 'snippet-asset');
        const img = document.createElement('img');
        img.src = node.pin.image;
        img.alt = node.title;
        img.loading = 'lazy';
        img.title = 'View in the source PDF';
        img.addEventListener('click', () => openSourceView(docName, node.pin));
        fig.appendChild(img);
        card.appendChild(fig);
      }

      if (node.reason) {
        const reasonEl = el('div', 'snippet-reason');
        const copyBtn = el('button', 'snippet-copy', '⧉ copy');
        copyBtn.title = 'Copy this passage';
        copyBtn.addEventListener('click', () => {
          navigator.clipboard?.writeText(node.content || '').then(
            () => toast('Passage copied.', 'ok', 1800),
            () => toast('Could not copy.', 'warn', 2500));
        });
        reasonEl.appendChild(copyBtn);
        const span = document.createElement('span');
        span.innerHTML = `<span class="reason-label">MODEL · </span>${escHtml(node.reason)}`;
        reasonEl.appendChild(span);
        card.appendChild(reasonEl);
      }
      card.appendChild(content);
      list.appendChild(card);
    }
  }

  document.getElementById('snippets-count').textContent = total;
  if (!total) list.appendChild(el('div', 'empty-msg', 'Nothing retrieved for this question.'));
}

function highlightSnippet(nodeId) {
  document.querySelectorAll('.snippet-card').forEach(c => c.classList.remove('highlight'));
  const card = document.getElementById(`snip-${nodeId}`);
  if (card) {
    card.classList.add('highlight');
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function highlightRelevantContent(content, quote) {
  if (!quote || !content) return escHtml(content || '');
  const candidates = [quote];
  quote.split(/[.!?]+/).forEach(s => { const t = s.trim(); if (t.length > 20) candidates.push(t); });
  for (const candidate of candidates) {
    const idx = content.toLowerCase().indexOf(candidate.toLowerCase());
    if (idx !== -1) {
      return `${escHtml(content.slice(0, idx))}<mark>${escHtml(content.slice(idx, idx + candidate.length))}</mark>${escHtml(content.slice(idx + candidate.length))}`;
    }
  }
  return escHtml(content);
}

/* ── 9 · Reader (document viewer) ──────────────────────────── */

function slugify(text) {
  return (text || '')
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-');
}

// Verdicts for a document: final results if present, else live run events.
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

async function openDocViewer(stem, targetTitle = null) {
  const modal   = document.getElementById('doc-modal');
  const titleEl = document.getElementById('doc-modal-title');
  const tocEl   = document.getElementById('doc-modal-toc');
  const bodyEl  = document.getElementById('doc-modal-content');

  titleEl.textContent = (stem || 'Document').replace(/_/g, ' ');
  tocEl.innerHTML = '';
  bodyEl.innerHTML = '<div class="empty-msg">Loading…</div>';
  modal.hidden = false;

  let doc;
  try {
    doc = await apiGet(`/api/document/${encodeURIComponent(stem)}/full`);
  } catch (e) {
    bodyEl.innerHTML = `<div class="empty-msg">Could not load document: ${escHtml(e.message)}</div>`;
    return;
  }

  const html = (typeof marked !== 'undefined')
    ? (marked.parse ? marked.parse(doc.markdown) : marked(doc.markdown))
    : `<pre>${escHtml(doc.markdown)}</pre>`;
  bodyEl.innerHTML = html;

  // Provenance pins → chips; the header checkbox is the single toggle.
  const pinCount = decoratePinBlocks(bodyEl, stem);
  initPinToggle(bodyEl, pinCount);

  // Anchor ids + TOC (with verdict colors mirrored).
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
    const vd = verdictFor(slugify(h.textContent), docV);
    if (vd) link.classList.add(`toc-v-${vd}`);
    link.textContent = h.textContent;
    link.addEventListener('click', (ev) => {
      ev.preventDefault();
      h.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    tocEl.appendChild(link);
  });

  state.docViewerStem = stem;
  bodyEl.dataset.vsig = '';
  decorateDocVerdicts(bodyEl, stem);

  if (targetTitle) {
    const want = slugify(targetTitle);
    let match = bodyEl.querySelector(`#${CSS.escape(want)}`);
    if (!match) match = Array.from(headings).find(h => slugify(h.textContent).startsWith(want));
    if (match) requestAnimationFrame(() => match.scrollIntoView({ behavior: 'auto', block: 'start' }));
    else bodyEl.scrollTop = 0;
  } else {
    bodyEl.scrollTop = 0;
  }
}

function closeDocViewer() {
  document.getElementById('doc-modal').hidden = true;
  state.docViewerStem = null;
}

function refreshOpenDocViewer() {
  if (!state.docViewerStem) return;
  if (document.getElementById('doc-modal').hidden) return;
  decorateDocVerdicts(document.getElementById('doc-modal-content'), state.docViewerStem);
}

const VERDICT_CLASSES = ['accepted', 'rejected', 'kept', 'pruned'];

/** Overlay per-node verdicts on the rendered markdown. Idempotent; re-run on
 *  every status poll while a run is live. */
function decorateDocVerdicts(bodyEl, stem) {
  if (!bodyEl) return;
  const legend = document.getElementById('doc-verdict-legend');
  const v = getDocVerdicts(stem);

  const sig = v ? JSON.stringify(Object.keys(v.meta).sort().map(k => k + ':' + (v.meta[k].status || '')))
                : '';
  if (bodyEl.dataset.vsig === sig) return;
  bodyEl.dataset.vsig = sig;

  bodyEl.querySelectorAll('[data-vid]').forEach(elx => {
    VERDICT_CLASSES.forEach(c => elx.classList.remove(`md-h-${c}`, `md-c-${c}`));
    delete elx.dataset.vid;
  });
  bodyEl.querySelectorAll('mark.md-quote').forEach(mk => {
    mk.replaceWith(document.createTextNode(mk.textContent));
  });
  bodyEl.normalize();

  if (legend) legend.hidden = true;
  if (!v) return;

  const headings = Array.from(bodyEl.querySelectorAll('h1, h2, h3, h4'));
  let annotated = 0;

  headings.forEach(h => {
    const baseId = slugify(h.textContent);
    const verdict = verdictFor(baseId, v);
    if (!verdict) return;
    const m = v.meta[baseId] || null;

    h.classList.add(`md-h-${verdict}`);
    h.dataset.vid = baseId;

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

  if (legend && annotated > 0) legend.hidden = false;
}

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
          } catch { /* spans element boundary — try next candidate */ }
        }
      }
    }
  }
  return false;
}

function showDecisionTooltip(event, vid) {
  const stem = state.docViewerStem;
  const v = stem ? getDocVerdicts(stem) : null;
  const m = v?.meta?.[vid];
  const verdict = v ? verdictFor(vid, v) : null;
  if (!verdict) { hideTooltip(); return; }

  const LABEL = {
    accepted: ['✓ Retrieved', 'tt-selected'],
    rejected: ['✗ Evaluated & not selected', 'tt-rejected'],
    kept:     ['☉ Section kept (passed pruning)', 'tt-kept-label'],
    pruned:   ['⊘ Pruned (branch skipped)', 'tt-pruned-label'],
  }[verdict];

  let html = `<div class="tt-label ${LABEL[1]}">${LABEL[0]}${v.live ? ' · live' : ''}</div>`;
  if (m?.reason) html += `<div class="tt-reason">${escHtml(m.reason)}</div>`;
  if (m?.quote)  html += `<div class="tt-quote">“${escHtml(m.quote)}”</div>`;

  if (verdict === 'rejected' || verdict === 'pruned') {
    const ex = state.explainCache?.[stem]?.[vid];
    if (ex && (ex.topic || ex.reason)) {
      if (ex.topic)  html += `<div class="tt-reason" style="margin-top:6px;"><span class="tt-muted">Topic:</span> ${escHtml(ex.topic)}</div>`;
      if (ex.reason) html += `<div class="tt-reason tt-muted">${escHtml(ex.reason)}</div>`;
    } else if (!m?.reason) {
      html += `<div class="tt-reason tt-muted">The model judged this does not directly answer the query.</div>`;
    }
    html += `<div class="tt-hint">▸ click to ask why (grounded re-read)</div>`;
  } else if (verdict === 'accepted' && findNodePin(stem, vid)?.page) {
    html += `<div class="tt-hint">▸ click to view in the source PDF</div>`;
  }
  showTooltip(event, html);
}

function initDocViewer() {
  const modal = document.getElementById('doc-modal');
  document.getElementById('doc-modal-close').addEventListener('click', closeDocViewer);
  modal.querySelector('.doc-modal-backdrop').addEventListener('click', closeDocViewer);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.hidden &&
        document.getElementById('source-popover')?.style.display !== 'flex') {
      closeDocViewer();
    }
  });

  const content = document.getElementById('doc-modal-content');
  content.addEventListener('mouseover', (e) => {
    const elx = e.target.closest('[data-vid]');
    if (elx) showDecisionTooltip(e, elx.dataset.vid);
    else hideTooltip();
  });
  content.addEventListener('mousemove', (e) => {
    if (getTooltip().style.display === 'block') positionTooltip(e);
  });
  content.addEventListener('mouseleave', hideTooltip);

  // rejected/pruned → grounded "why not"; retrieved + pin → source PDF.
  content.addEventListener('click', (e) => {
    if (e.target.closest('.pin-chip')) return;
    const elx = e.target.closest('[data-vid]');
    if (!elx) return;
    const stem = state.docViewerStem; if (!stem) return;
    const verdict = verdictFor(elx.dataset.vid, getDocVerdicts(stem));
    if (verdict === 'rejected' || verdict === 'pruned') {
      requestExplanation(stem, elx.dataset.vid, elx);
    } else if (verdict === 'accepted') {
      const pin = findNodePin(stem, elx.dataset.vid);
      if (pin?.page) openSourceView(stem, pin);
    }
  });
}

/* ── explain popover ("why not selected") ───────────────────── */

function getExplainPopover() {
  let p = document.getElementById('doc-explain-popover');
  if (!p) {
    p = document.createElement('div');
    p.id = 'doc-explain-popover';
    p.style.display = 'none';
    document.body.appendChild(p);
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
  const pw = p.offsetWidth || 330;
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

/* ── provenance pins in the reader ──────────────────────────── */

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
    if (val.startsWith('[')) { try { val = JSON.parse(val); } catch { /* raw */ } }
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
  toggle.hidden = pinCount === 0;
  box.checked = state.showPins;
  bodyEl.classList.toggle('pins-visible', state.showPins);
  box.onchange = () => {
    state.showPins = box.checked;
    prefs.set('showPins', state.showPins);
    bodyEl.classList.toggle('pins-visible', state.showPins);
  };
}

/* ── 10 · Corpus setup (ingest) — the ONE surface ───────────── */

function initSetupCard(docs) {
  document.getElementById('setup-browse-btn').addEventListener('click', browseForFolder);
  const input = document.getElementById('setup-source-path');
  input.addEventListener('input', () => {
    document.getElementById('setup-run-btn').disabled = !input.value.trim();
  });
  document.getElementById('setup-run-btn').addEventListener('click', runIngest);
  document.getElementById('setup-dismiss-btn').addEventListener('click', closeSetupCard);

  // First launch of the workflow: an empty library asks for its corpus;
  // a configured PDF module that lost its folder asks again.
  if (!docs?.length) {
    openSetupCard({ dismissable: false });
  } else if (ingestModuleNeedsSource(document.getElementById('sel-ingest').value)) {
    apiGet('/api/ingest/source?module=' +
        encodeURIComponent(document.getElementById('sel-ingest').value))
      .then(src => { if (!src.source_ok) openSetupCard(); })
      .catch(() => { /* backend unreachable — init already toasted */ });
  }
}

async function openSetupCard(opts = {}) {
  const card = document.getElementById('setup-card');
  card.hidden = false;
  document.getElementById('treemap-pack').hidden = true;
  document.getElementById('treemap-legend').hidden = true;
  document.getElementById('setup-dismiss-btn').hidden =
    opts.dismissable === false ? true : !(_docsCache || []).length;
  document.getElementById('setup-warnings').innerHTML = '';
  document.getElementById('setup-progress').hidden = true;
  document.getElementById('library-status').textContent =
    (_docsCache || []).length
      ? 'Point the library at a different folder of PDFs, or re-ingest the current one.'
      : 'No documents yet — choose a folder of PDFs to begin.';
  try {
    const name = document.getElementById('sel-ingest').value;
    const src = await apiGet(`/api/ingest/source?module=${encodeURIComponent(name)}`);
    const input = document.getElementById('setup-source-path');
    if (src.source_dir && !input.value) input.value = src.source_dir;
    document.getElementById('setup-run-btn').disabled = !input.value.trim();
  } catch { /* fresh setup */ }
}

function closeSetupCard() {
  document.getElementById('setup-card').hidden = true;
  if ((_docsCache || []).length) {
    document.getElementById('treemap-pack').hidden = false;
    document.getElementById('treemap-legend').hidden = false;
  }
  resetLibraryStatus();
}

async function browseForFolder() {
  const input = document.getElementById('setup-source-path');
  const dialog = window.__TAURI__?.dialog;
  if (dialog?.open) {
    const picked = await dialog.open({ directory: true, multiple: false,
                                       title: 'Choose a folder of PDFs' });
    if (picked) {
      input.value = Array.isArray(picked) ? picked[0] : picked;
      input.dispatchEvent(new Event('input'));
    }
  } else {
    input.placeholder = 'No native dialog in browser mode — paste the folder path here';
    input.focus();
  }
}

async function runIngest() {
  const path   = document.getElementById('setup-source-path').value.trim();
  const runBtn = document.getElementById('setup-run-btn');
  const listEl = document.getElementById('setup-file-list');
  const warnEl = document.getElementById('setup-warnings');
  if (!path) return;

  runBtn.disabled = true;
  warnEl.innerHTML = '';
  const moduleName = ingestModuleNeedsSource(document.getElementById('sel-ingest').value)
    ? document.getElementById('sel-ingest').value : 'betteringest_pdf';
  try {
    const src = await apiPost('/api/ingest/source', { module: moduleName, path });
    listEl.hidden = false;
    listEl.innerHTML = `<div class="file-count">${src.pdf_count} PDF(s)</div>` +
      src.pdfs.map(p => `<div class="file">${escHtml(p)}</div>`).join('');
    await apiPost('/api/ingest/run', {
      ingest_module: moduleName,
      index_module: document.getElementById('sel-index')?.value || null,
    });
  } catch (e) {
    warnEl.innerHTML = `<div class="warn">✗ ${escHtml(e.message)}</div>`;
    runBtn.disabled = false;
    return;
  }

  const progWrap = document.getElementById('setup-progress');
  const fill = document.getElementById('setup-progress-fill');
  const msg  = document.getElementById('setup-progress-msg');
  progWrap.hidden = false;

  const poll = setInterval(async () => {
    let p;
    try { p = await apiGet('/api/ingest/progress'); } catch { return; }
    const pct = p.total ? Math.round((p.done / p.total) * 100) : 0;
    fill.style.width = `${p.phase === 'index' || p.state === 'done' ? 100 : pct}%`;
    msg.textContent = p.message || p.phase || '';
    if (p.state !== 'done' && p.state !== 'error') return;

    clearInterval(poll);
    runBtn.disabled = false;
    if (p.state === 'error') {
      warnEl.innerHTML = `<div class="warn">✗ ${escHtml(p.message)}</div>`;
      toast(`Ingest failed: ${p.message}`, 'err', 8000);
      return;
    }
    for (const w of p.warnings || []) {
      warnEl.innerHTML += `<div class="warn">⚠ ${escHtml(w)}</div>`;
      toast(w, 'warn', 7000);
    }
    msg.textContent = `Done — ${p.docs?.length ?? 0} document(s) ingested and indexed.`;
    runBtn.textContent = 'Re-ingest';
    toast(`Library updated — ${p.docs?.length ?? 0} document(s) ready.`, 'ok');

    _docsCache = null;
    try {
      const docs = await getDocs();
      renderTreemap(docs);
      updateLibraryCount(docs);
      refreshSourceSection();
      if (docs.length && !(p.warnings || []).length) setTimeout(closeSetupCard, 1000);
    } catch (e) { console.error('refresh error:', e); }
  }, 500);
}

/* ── source popover: the pin's PDF page, bbox highlighted ───── */

function getSourcePopover() {
  let p = document.getElementById('source-popover');
  if (!p) {
    p = document.createElement('div');
    p.id = 'source-popover';
    p.style.display = 'none';
    p.innerHTML = `
      <div class="sp-head">
        <span id="sp-title">Source</span>
        <button id="sp-close" title="Close (Esc)">✕</button>
      </div>
      <div class="sp-body"><img id="sp-img" alt="source page"></div>`;
    document.body.appendChild(p);
    p.querySelector('#sp-close').addEventListener('click', () => { p.style.display = 'none'; });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && p.style.display === 'flex') p.style.display = 'none';
    });
  }
  return p;
}

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

/* ── Utilities ─────────────────────────────────────────────── */

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls)  e.className   = cls;
  if (text) e.textContent = text;
  return e;
}
function badge(cls, text) { return el('span', `tag ${cls}`, text); }

function escHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

let _tooltip = null;
function getTooltip() {
  if (!_tooltip) {
    _tooltip = document.createElement('div');
    _tooltip.id = 'tree-tooltip';
    document.body.appendChild(_tooltip);
  }
  return _tooltip;
}
function showTooltip(event, html) {
  const tip = getTooltip();
  tip.innerHTML = html;
  tip.style.display = 'block';
  positionTooltip(event);
}
function hideTooltip() { getTooltip().style.display = 'none'; }

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

function makeDraggable(elx) {
  if (!elx) return;
  let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  elx.onmousedown = dragMouseDown;

  function dragMouseDown(e) {
    if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'BUTTON' ||
        e.target.closest('button') || e.target.closest('.suggestions')) return;
    e.preventDefault();
    pos3 = e.clientX; pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }
  function elementDrag(e) {
    e.preventDefault();
    pos1 = pos3 - e.clientX; pos2 = pos4 - e.clientY;
    pos3 = e.clientX; pos4 = e.clientY;
    let newTop = elx.offsetTop - pos2;
    let newLeft = elx.offsetLeft - pos1;
    newTop = Math.max(10, Math.min(window.innerHeight - elx.offsetHeight - 10, newTop));
    newLeft = Math.max(10, Math.min(window.innerWidth - elx.offsetWidth - 10, newLeft));
    elx.style.top = newTop + 'px';
    elx.style.left = newLeft + 'px';
    elx.style.bottom = 'auto';
    elx.style.right = 'auto';
  }
  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

/* ── Boot ──────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', init);
