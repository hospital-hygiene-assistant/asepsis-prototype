'use strict';

/* ═══════════════════════════════════════════════════════════════════
   ASEPSIS chatbot tab — loaded after main.js, shares its globals
   (state, apiGet/apiPost, toast, getDocs, renderResults, openDocViewer,
   openSourceView, highlightRelevantContent, start/stopStatusPolling).

   Sections
     1. tab switching                4. live retrieval trace (CoT)
     2. chat state + input           5. answer card + sources
     3. send workflow                6. evidence library (right dock)
   ═══════════════════════════════════════════════════════════════════ */

/* ── 1 · App tabs: Chatbot ⟷ Retrieval ─────────────────────── */

const chatState = {
  tab: 'chat',
  messages: [],            // {role, content, ...; assistant: {id, data}}
  answerCount: 0,
  pending: null,           // live card refs while a question is running
  nodeInfo: {},            // node_id -> {doc, title} (null doc = ambiguous)
  evMode: 'all',
  activeAnswerId: null,
};

function setAppTab(tab) {
  chatState.tab = tab;
  prefs.set('appTab', tab);
  document.body.dataset.tab = tab;
  document.querySelectorAll('.app-tab').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tab));
  const chatView = document.getElementById('view-chat');
  if (tab === 'chat') {
    chatView.hidden = false;
    document.getElementById('view-library').hidden = true;
    document.getElementById('view-results').hidden = true;
    renderChatEmptyState();
    document.getElementById('chat-input').focus();
  } else {
    chatView.hidden = true;
    setView(state.view);               // restore library/results visibility
    // Re-render whatever was drawn while this tab was hidden (0×0 measures).
    if (state.view === 'library' && _docsCache) renderTreemap(_docsCache);
    if (state.view === 'results' && state.currentResults) renderCurrentResultView();
  }
}

// Keep the retrieval views hidden whenever main.js flips them while the
// chat tab is front — e.g. init()'s setView('library') racing tab restore.
const _setViewOrig = setView;
setView = function (name) {
  _setViewOrig(name);
  if (chatState.tab === 'chat') {
    document.getElementById('view-library').hidden = true;
    document.getElementById('view-results').hidden = true;
  }
};

function openInRetrieval(docName, nodeId) {
  if (!state.currentResults) { setAppTab('retrieval'); return; }
  setAppTab('retrieval');
  setView('results');
  if (docName && state.currentResults.results[docName]) {
    state.currentDoc = docName;
    document.querySelectorAll('.stat-card').forEach(c =>
      c.classList.toggle('active', c.dataset.doc === docName));
    renderCurrentResultView();
  }
  if (nodeId) highlightSnippet(nodeId);
}

/* ── 2 · Chat copy, input plumbing ─────────────────────────── */

const chatCopy = {
  answerLabel: 'Answer',
  sections: [
    ['short_answer', 'Short answer'],
    ['recommended_action', 'Recommended action'],
    ['rationale', 'Rationale'],
    ['limitations', 'Limitations'],
  ],
  sourcesLabel: 'Document passages',
  noSources: 'No document passages support this answer.',
  whySelected: 'Why selected',
  emptyLabel: 'Example questions',
  emptyHint: 'Clicking one places it in the input field.',
  loading: 'Preparing the answer…',
  grounding: {
    grounded:              ['Grounded in document passages', 'check'],
    partially_grounded:    ['Partially grounded', 'file'],
    insufficient_evidence: ['No sufficient source', 'alert'],
    search_incomplete:      ['Search incomplete', 'alert'],
    not_connected:         ['Document search not connected', 'info'],
  },
};

const ICONS = {
  check: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
  alert: '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
  file:  '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>',
  info:  '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
  zoom:  '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"></polyline><polyline points="9 21 3 21 3 15"></polyline><line x1="21" y1="3" x2="14" y2="10"></line><line x1="3" y1="21" x2="10" y2="14"></line></svg>',
  book:  '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>',
  map:   '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>',
};

function groundingBadge(status) {
  const [label, icon] = chatCopy.grounding[status] || chatCopy.grounding.not_connected;
  const span = el('span', `grounding-badge ${status}`);
  span.innerHTML = `${ICONS[icon]} ${escHtml(label)}`;
  return span;
}

function prettyDoc(name) { return (name || '').replace(/_/g, ' '); }

// The debug console has its own rendering model. Keep the v3 wire translation
// in one adapter so every view below consumes one local shape and no view can
// accidentally reconstruct immutable evidence links.
function debugAnswerFromWire(data) {
  if (data?.contract_version !== 3 || !data.outcome?.kind || !data.outcome.coverage) {
    throw new Error('The backend returned an invalid chat contract.');
  }
  const outcome = data.outcome;
  if (outcome.kind === 'retrieval_unavailable' || outcome.kind === 'synthesis_unavailable') {
    throw new Error(outcome.kind === 'retrieval_unavailable'
      ? 'Document retrieval was unavailable.'
      : 'Answer synthesis was unavailable.');
  }
  const status = outcome.kind === 'answered'
    ? (outcome.coverage.status === 'complete' ? 'grounded' : 'partially_grounded')
    : outcome.kind;
  const citations = outcome.kind === 'answered' ? outcome.citations : [];
  return {
    query: data.query,
    answer: outcome.kind === 'answered' ? outcome.answer : {},
    grounding: {
      status,
      generation_id: outcome.coverage.generation_id,
      searched_documents: outcome.coverage.searched_documents,
      total_documents: outcome.coverage.total_documents,
      incomplete_checks: outcome.coverage.incomplete_checks,
      sources: citations.map(citation => ({
        id: citation.id,
        number: citation.number,
        document: citation.document_id,
        node_id: citation.node_id,
        title: citation.title,
        breadcrumb: citation.breadcrumb,
        excerpt: citation.excerpt,
        quote: citation.quote,
        reason: citation.selection_reason,
        source_href: citation.visual.pdf_href,
        visual: citation.visual,
      })),
    },
  };
}

function setSendEnabled(enabled) {
  const input = document.getElementById('chat-input');
  const btn = document.getElementById('chat-send');
  input.disabled = !enabled;
  btn.disabled = !enabled || !input.value.trim();
  document.getElementById('chat-send-label').textContent = enabled ? 'Send' : 'Sending';
}

function renderChatEmptyState() {
  const empty = document.getElementById('chat-empty');
  empty.hidden = chatState.messages.length > 0;
  if (empty.hidden || empty.childElementCount > 0 && empty.dataset.filled === '1') return;
  if (!state.tests?.length) { setTimeout(renderChatEmptyState, 600); return; }
  empty.innerHTML = '';
  empty.dataset.filled = '1';
  empty.appendChild(el('p', 'chat-empty-label', chatCopy.emptyLabel));
  empty.appendChild(el('p', 'chat-empty-hint', chatCopy.emptyHint));
  const wrap = el('div', 'chat-empty-prompts');
  const picks = [state.tests[0], state.tests[2], state.tests[5]].filter(Boolean);
  for (const t of picks) {
    const b = el('button', 'chat-prompt-btn', t.query);
    b.type = 'button';
    b.addEventListener('click', () => {
      const input = document.getElementById('chat-input');
      input.value = t.query;
      input.dispatchEvent(new Event('input'));
      input.focus();
    });
    wrap.appendChild(b);
  }
  empty.appendChild(wrap);
}

function scrollChatToBottom(smooth = true) {
  const sc = document.getElementById('chat-scroll');
  sc.scrollTo({ top: sc.scrollHeight, behavior: smooth ? 'smooth' : 'auto' });
}

/* ── 3 · Send workflow ─────────────────────────────────────── */

async function buildNodeInfo() {
  const info = {};
  try {
    const docs = await getDocs();
    for (const d of docs) {
      (function walk(nodes) {
        for (const n of nodes || []) {
          if (n.nodeId) {
            if (info[n.nodeId] && info[n.nodeId].doc !== d.name) info[n.nodeId].doc = null;
            else if (!info[n.nodeId]) info[n.nodeId] = { doc: d.name, title: n.title || n.nodeId };
          }
          walk(n.children);
        }
      })(d.tree);
    }
  } catch { /* trace lines fall back to bare node ids */ }
  return info;
}

async function sendChat(text) {
  if (state.running) { toast('A run is already in progress.', 'warn'); return; }

  chatState.messages.push({ role: 'user', content: text });
  renderChatEmptyState();

  const messagesEl = document.getElementById('chat-messages');
  const userRow = el('div', 'chat-row user');
  userRow.appendChild(el('div', 'chat-user-bubble', text));
  messagesEl.appendChild(userRow);

  const pending = buildPendingCard();
  messagesEl.appendChild(pending.row);
  document.getElementById('view-chat').classList.add('has-conversation');
  scrollChatToBottom();

  state.running = true;
  state.currentTestId = null;
  state.currentQuery = text;
  setRunPill('Evaluating…');
  setSendEnabled(false);
  const input = document.getElementById('chat-input');
  input.value = '';
  input.style.height = '';

  chatState.pending = pending;
  chatState.nodeInfo = await buildNodeInfo();
  try { renderTreemap(await getDocs()); } catch { /* keep old canvas */ }
  state.runId = newRunId();
  startStatusPolling();
  updateFlowSteps();

  try {
    const data = await apiPost('/api/chat', {
      query: text,
      run_id: state.runId,
    });
    finalizeAnswerCard(pending, debugAnswerFromWire(data));
  } catch (e) {
    pending.card.querySelector('.answer-body').innerHTML =
      `<p class="chat-error">The question could not be processed: ${escHtml(e.message)}</p>`;
    pending.progressEl.remove();
    toast(`Chat failed: ${e.message}`, 'err', 7000);
  } finally {
    state.running = false;
    setRunPill(null);
    await stopStatusPolling();     // final status flush → last trace lines land
    chatState.pending = null;
    setSendEnabled(true);
    updateFlowSteps();
    if (chatState.tab === 'chat') document.getElementById('chat-input').focus();
  }
}

/* ── 4 · Live retrieval trace — the model's chain of thought ── */

const TRACE_META = {
  retrieved: ['✓', 'Retrieved'],
  rejected:  ['✕', 'Rejected'],
  kept:      ['○', 'Kept'],
  pruned:    ['⊘', 'Pruned'],
};

function buildPendingCard() {
  const row = el('div', 'chat-row assistant');
  const card = el('article', 'answer-card');
  card.id = `answer-a${chatState.answerCount + 1}`;

  const head = el('div', 'answer-head');
  head.appendChild(el('p', 'answer-label', chatCopy.answerLabel));
  card.appendChild(head);

  const body = el('div', 'answer-body');
  card.appendChild(body);

  const trace = el('div', 'trace-block open');
  const toggle = el('button', 'trace-toggle');
  toggle.type = 'button';
  toggle.innerHTML = `<span class="trace-caret">▶</span> Retrieval reasoning
    <span class="trace-count"></span>
    <span class="trace-open-retrieval">Open retrieval map ↗</span>`;
  toggle.addEventListener('click', (e) => {
    if (e.target.closest('.trace-open-retrieval')) { openInRetrieval(); return; }
    trace.classList.toggle('open');
  });
  const traceBody = el('div', 'trace-body');
  const progressEl = el('div', 'trace-progress');
  progressEl.innerHTML = `<span class="trace-spinner"></span><span class="trace-progress-text">Walking the document trees…</span>`;
  const logEl = el('div', 'trace-log');
  logEl.hidden = true;
  traceBody.appendChild(progressEl);
  traceBody.appendChild(logEl);
  trace.appendChild(toggle);
  trace.appendChild(traceBody);
  card.appendChild(trace);

  row.appendChild(card);
  return { row, card, trace, traceBody, progressEl, logEl,
           countEl: toggle.querySelector('.trace-count'),
           seen: new Set(), stats: { retrieved: 0, total: 0 } };
}

// Called from main.js's 250ms status poller during any run.
function chatOnStatus(status) {
  const p = chatState.pending;
  if (!p || !status) return;

  const prog = status.progress || {};
  const textEl = p.progressEl.querySelector('.trace-progress-text');
  if (textEl) {
    if (status.chat?.phase === 'synthesis') {
      textEl.textContent = status.chat.detail || 'Composing the answer…';
    } else if (prog.total > 0) {
      textEl.textContent = `Evaluating passages — ${prog.done} / ${prog.total}`;
    }
  }

  const audit = status.live?.decisions;
  const entries = Array.isArray(audit)
    ? audit.map(item => [`${item.document_id}::${item.node_id}`, item])
    : Object.entries(status.live?.meta || {});
  let added = false;
  for (const [identity, m] of entries) {
    if (p.seen.has(identity)) continue;
    p.seen.add(identity);
    const nodeId = m.node_id || identity;
    const [ico, word] = TRACE_META[m.status] || ['·', m.status];
    const line = el('div', `trace-line ${m.status}`);
    const info = chatState.nodeInfo[nodeId] || {};
    const label = info.title || nodeId;
    const documentId = m.document_id || info.doc;
    const docTag = documentId ? ` <span class="trace-doc">· ${escHtml(prettyDoc(documentId))}</span>` : '';
    line.innerHTML = `<span class="trace-ico">${ico}</span><span>` +
      `<span class="trace-what">${word} — ${escHtml(label)}</span>${docTag}` +
      (m.reason ? ` <span class="trace-why">${escHtml(m.reason)}</span>` : '') +
      `</span>`;
    p.logEl.appendChild(line);
    p.stats.total++;
    if (m.status === 'retrieved') p.stats.retrieved++;
    added = true;
  }
  if (added) {
    p.logEl.hidden = false;
    p.countEl.textContent = `· ${p.stats.total} decisions · ${p.stats.retrieved} retrieved`;
    p.logEl.scrollTop = p.logEl.scrollHeight;
  }
}

/* ── 5 · Answer card + source cards ────────────────────────── */

// Escape + linebreaks + [n] → clickable citation chips.
function renderRichText(text, sources, answerId) {
  let html = (typeof marked !== 'undefined')
    ? (marked.parse ? marked.parse(text || '') : marked(text || ''))
    : escHtml(text || '').replace(/\n/g, '<br>');

  html = html.replace(/\[(\d+)\]/g, (whole, n) => {
    const num = parseInt(n, 10);
    if (!sources || num < 1 || num > sources.length) return whole;
    return `<button type="button" class="cite-chip" data-cite="${num}" data-answer="${answerId}" title="Jump to source [${num}]">${num}</button>`;
  });
  return html;
}

function finalizeAnswerCard(p, data) {
  const answerId = `a${++chatState.answerCount}`;
  p.card.id = `answer-${answerId}`;
  chatState.activeAnswerId = answerId;

  const sources = data.grounding?.sources || [];
  const status = data.grounding?.status || 'not_connected';

  // Header: grounding badge
  p.card.querySelector('.answer-head').appendChild(groundingBadge(status));

  // Body: structured sections, else the raw answer
  const body = p.card.querySelector('.answer-body');
  body.innerHTML = '';
  const a = data.answer || {};
  const present = chatCopy.sections.filter(([key]) => (a[key] || '').trim());
  if (present.length) {
    for (const [key, label] of present) {
      const sec = el('section', 'answer-section');
      const h = el('h3', '', label);
      const txt = el('p', 'answer-text');
      txt.innerHTML = renderRichText(a[key], sources, answerId);
      sec.appendChild(h); sec.appendChild(txt);
      body.appendChild(sec);
    }
  } else if ((a.content || '').trim()) {
    const div = el('div', 'answer-fallback');
    div.innerHTML = renderRichText(a.content, sources, answerId);
    body.appendChild(div);
  } else {
    body.appendChild(el('p', 'chat-loading', 'The model returned no answer text — see the sources below.'));
  }

  // Trace: freeze into a collapsed audit block
  p.progressEl.remove();
  p.trace.classList.remove('open');
  if (p.stats.total === 0) p.trace.remove();

  // Sources
  const srcWrap = el('div', 'answer-sources');
  const rule = el('div', 'answer-sources-rule');
  rule.appendChild(el('p', 'sources-label', chatCopy.sourcesLabel));
  const grounding = data.grounding || {};
  const coverage = `${grounding.searched_documents ?? 0}/${grounding.total_documents ?? 0} documents searched; ` +
    `${grounding.incomplete_checks ?? 0} incomplete check(s).`;
  rule.appendChild(el('p', 'sources-summary', coverage));
  if (!sources.length) {
    const none = el('p', 'sources-none');
    none.innerHTML = `${ICONS.info} ${escHtml(chatCopy.noSources)}`;
    rule.appendChild(none);
  } else {
    const list = el('div', 'sources-list');
    for (const s of sources) list.appendChild(buildSourceCard(s, answerId));
    rule.appendChild(list);
  }
  srcWrap.appendChild(rule);
  p.card.appendChild(srcWrap);

  // Deterministic AI disclaimer — rendered by the app on every answer,
  // never left to the model's prompt.
  const disclaimer = el('footer', 'answer-disclaimer');
  disclaimer.innerHTML = `${ICONS.info}<span><b>AI-generated answer.</b> ` +
    `Always double-check against the cited document passages before acting on it.</span>`;
  p.card.appendChild(disclaimer);

  // Citation chips → scroll to the source card
  p.card.addEventListener('click', (e) => {
    const chip = e.target.closest('.cite-chip');
    if (chip) focusSourceCard(chip.dataset.answer, parseInt(chip.dataset.cite, 10));
  });

  chatState.messages.push({ role: 'assistant', id: answerId, query: data.query, data });
  updateEvidenceLibrary();
  scrollChatToBottom();
}

function buildSourceCard(s, answerId) {
  const number = s.number;
  const documentName = s.document;
  const exactPage = s.visual?.status === 'exact' ? s.visual.pages?.[0]?.page : null;
  const card = el('article', 'source-card');
  card.id = `src-${answerId}-${number}`;

  const eyebrow = el('p', 'source-eyebrow');
  eyebrow.innerHTML = `[${number}] ${escHtml(prettyDoc(documentName))}` +
    (exactPage ? ` <span class="source-page-pill">p. ${escHtml(String(exactPage))}</span>` : '');
  card.appendChild(eyebrow);

  const title = el('h3', 'source-title');
  if (s.breadcrumb && s.breadcrumb !== s.title) {
    const parts = escHtml(s.breadcrumb).split(' &gt; ');
    title.innerHTML = parts.length > 1
      ? `<span class="source-crumb">${parts.slice(0, -1).join(' › ')} › </span>${parts.at(-1)}`
      : escHtml(s.title);
  } else {
    title.textContent = s.title;
  }
  card.appendChild(title);

  // Excerpt with the deciding quote highlighted
  if ((s.excerpt || '').trim()) {
    const quote = el('blockquote', 'source-excerpt');
    quote.innerHTML = highlightRelevantContent(s.excerpt, s.quote || '');
    card.appendChild(quote);
    if (s.excerpt.length > 420) {
      quote.classList.add('clamped');
      const more = el('button', 'source-expand', 'Show full passage');
      more.type = 'button';
      more.addEventListener('click', () => {
        const expanded = quote.classList.toggle('expanded');
        quote.classList.toggle('clamped', !expanded);
        more.textContent = expanded ? 'Collapse passage' : 'Show full passage';
      });
      card.appendChild(more);
    }
  }

  if ((s.reason || '').trim()) {
    const why = el('div', 'source-why');
    why.appendChild(el('p', 'source-why-label', chatCopy.whySelected));
    why.appendChild(el('p', '', s.reason));
    card.appendChild(why);
  }

  const actions = el('div', 'source-actions');
  if (exactPage && s.source_href) {
    const b = el('button', 'source-action');
    b.type = 'button';
    b.innerHTML = `${ICONS.zoom} Open source page`;
    b.addEventListener('click', () => window.open(
      `${s.source_href}#page=${exactPage}`,
      '_blank',
    ));
    actions.appendChild(b);
  } else {
    // No provenance pin or the source PDF is unavailable — keep the button,
    // but as an explicit warning instead of silently hiding the affordance.
    const b = el('button', 'source-action warn');
    b.type = 'button';
    b.innerHTML = `${ICONS.alert} Open source page`;
    b.title = 'No source PDF is available for this passage';
    b.addEventListener('click', () => toast(
      `No exact source-PDF location is available for “${prettyDoc(documentName)}”. ` +
      `Verify against the cited text.`,
      'warn', 8000));
    actions.appendChild(b);
  }
  const read = el('button', 'source-action');
  read.type = 'button';
  read.innerHTML = `${ICONS.book} Read in document`;
  read.title = 'Open the annotated reader at this section';
  read.addEventListener('click', () => openDocViewer(documentName, s.title));
  actions.appendChild(read);

  card.appendChild(actions);

  return card;
}

function focusSourceCard(answerId, n) {
  const card = document.getElementById(`src-${answerId}-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: 'smooth', block: 'center' });
  card.classList.remove('flash');
  requestAnimationFrame(() => card.classList.add('flash'));
  selectEvidenceSource(answerId, n);
}

/* ── 6 · Evidence library (right dock) ─────────────────────── */

function evidenceGroups() {
  return chatState.messages
    .filter(m => m.role === 'assistant' && (m.data?.grounding?.sources || []).length)
    .map((m, i) => ({
      answerId: m.id,
      label: `Answer ${i + 1}`,
      query: m.query,
      status: m.data.grounding.status,
      summary: `${m.data.grounding.searched_documents}/${m.data.grounding.total_documents} documents searched`,
      sources: m.data.grounding.sources,
    }));
}

function updateEvidenceLibrary() {
  const aside = document.getElementById('evidence-library');
  const groups = evidenceGroups();
  aside.hidden = groups.length === 0;
  if (aside.hidden) return;

  const wrap = document.getElementById('evidence-groups');
  wrap.innerHTML = '';
  const timeline = el('div', 'ev-timeline');

  const shown = chatState.evMode === 'current'
    ? groups.filter(g => g.answerId === chatState.activeAnswerId)
    : groups;

  for (const g of shown) {
    const group = el('section', `ev-group${g.answerId === chatState.activeAnswerId ? ' active' : ''}`);
    group.appendChild(el('span', 'ev-dot'));

    const head = el('div', 'ev-group-head');
    const titleBtn = el('button', 'ev-group-title');
    titleBtn.type = 'button';
    titleBtn.innerHTML = `${escHtml(g.label)} <span class="ev-count">· ${g.sources.length} source${g.sources.length > 1 ? 's' : ''}</span>`;
    titleBtn.addEventListener('click', () => {
      document.getElementById(`answer-${g.answerId}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    head.appendChild(titleBtn);
    head.appendChild(el('p', 'ev-group-q', g.query));
    head.appendChild(groundingBadge(g.status));
    group.appendChild(head);

    const list = el('div', 'ev-sources');
    for (const s of g.sources) {
      const btn = el('button', 'ev-source');
      btn.type = 'button';
      btn.dataset.answer = g.answerId;
      btn.dataset.n = s.number;
      const page = s.visual?.status === 'exact' ? s.visual.pages?.[0]?.page : null;
      const eyebrow = `<p class="ev-source-eyebrow"><span>[${s.number}] ${escHtml(prettyDoc(s.document))}</span>` +
        (page ? `<span class="ev-page">p. ${escHtml(String(page))}</span>` : '') + `</p>`;
      btn.innerHTML = eyebrow +
        `<p class="ev-source-title">${escHtml(s.title)}</p>` +
        (s.quote ? `<p class="ev-source-quote">“${escHtml(s.quote)}”</p>` : '');
      btn.addEventListener('click', () => focusSourceCard(g.answerId, s.number));
      list.appendChild(btn);
    }
    group.appendChild(list);
    timeline.appendChild(group);
  }
  wrap.appendChild(timeline);
}

function selectEvidenceSource(answerId, n) {
  document.querySelectorAll('.ev-source').forEach(b => {
    b.classList.toggle('selected', b.dataset.answer === answerId && b.dataset.n === String(n));
  });
}

/* ── Chatbot inspector: model, engine, prompts ─────────────── */

const PROMPT_LABELS = {
  synthesis:        ['Answer synthesis', 'Turns the retrieved passages into the structured, cited answer.'],
  section_pruning:  ['Section pruning', 'Phase 1 of retrieval — decides which branches of each document tree are worth reading.'],
  leaf_evaluation:  ['Leaf evaluation', 'Phase 2 — judges every surviving passage and demands a verbatim quote for a yes.'],
  why_not_explainer:['“Why not?” explainer', 'On-demand, grounded explanation for passages that were not selected.'],
};

async function openChatConfig() {
  const modal = document.getElementById('chat-config-modal');
  const body = document.getElementById('chat-config-body');
  modal.hidden = false;
  body.innerHTML = '<p class="chat-loading">Loading…</p>';
  let cfg, modelsRes;
  try {
    [cfg, modelsRes] = await Promise.all([
      apiGet('/api/chat/config'),
      apiGet('/api/models').catch(() => ({ models: [] }))
    ]);
  } catch (e) {
    body.innerHTML = `<p class="chat-error">Could not load the configuration: ${escHtml(e.message)}</p>`;
    return;
  }

  body.innerHTML = '';
  const facts = el('dl', 'ccm-facts');
  const fact = (label, valueNodeOrHtml) => {
    facts.appendChild(el('dt', '', label));
    const dd = el('dd', '');
    if (valueNodeOrHtml instanceof HTMLElement) {
      dd.appendChild(valueNodeOrHtml);
    } else {
      dd.innerHTML = valueNodeOrHtml;
    }
    facts.appendChild(dd);
  };

  const avail = modelsRes.models && modelsRes.models.length > 0
    ? modelsRes.models
    : [cfg.model];

  const configuredRetrieval = cfg.retrieval_model || cfg.model;
  const configuredSynthesis = cfg.synthesis_model || cfg.model;
  const uniqueAvail = Array.from(new Set([...avail, configuredRetrieval, configuredSynthesis]));

  const retrievalSelect = document.createElement('select');
  retrievalSelect.className = 'ccm-select-model';

  const synthesisSelect = document.createElement('select');
  synthesisSelect.className = 'ccm-select-model';

  for (const m of uniqueAvail) {
    const opt1 = document.createElement('option');
    opt1.value = m;
    opt1.textContent = m;
    opt1.selected = m === configuredRetrieval;
    retrievalSelect.appendChild(opt1);

    const opt2 = document.createElement('option');
    opt2.value = m;
    opt2.textContent = m;
    opt2.selected = m === configuredSynthesis;
    synthesisSelect.appendChild(opt2);
  }

  const saveConfig = async () => {
    try {
      await apiPost('/api/config', {
        ollama_instances: cfg.ollama_instances,
        retrieval_model: retrievalSelect.value,
        synthesis_model: synthesisSelect.value
      });
      toast('Model configuration updated!', 'info');
    } catch (e) {
      toast(`Failed to update config: ${e.message}`, 'err');
    }
  };

  retrievalSelect.addEventListener('change', saveConfig);
  synthesisSelect.addEventListener('change', saveConfig);

  fact('Retrieval model', retrievalSelect);
  fact('Synthesis model', synthesisSelect);
  fact('Engine', `${cfg.ollama_instances} local Ollama instance${cfg.ollama_instances > 1 ? 's' : ''} — ` +
    cfg.ollama_urls.map(u => `<code>${escHtml(u)}</code>`).join(', ') +
    (cfg.any_busy ? ' · <b>busy</b>' : ' · idle'));
  body.appendChild(facts);

  const promptsLabel = el('p', 'ccm-section-label', 'Prompts');
  body.appendChild(promptsLabel);
  for (const [key, prompt] of Object.entries(cfg.prompts || {})) {
    if (!prompt) continue;
    const [label, hint] = PROMPT_LABELS[key] || [key, ''];
    const det = document.createElement('details');
    det.className = 'ccm-prompt';
    const sum = document.createElement('summary');
    sum.innerHTML = `<b>${escHtml(label)}</b><span>${escHtml(hint)}</span>`;
    const pre = el('pre', '', prompt);
    det.appendChild(sum);
    det.appendChild(pre);
    body.appendChild(det);
  }
}

function initChatConfig() {
  const modal = document.getElementById('chat-config-modal');
  document.getElementById('chat-config-btn').addEventListener('click', openChatConfig);
  document.getElementById('chat-config-close').addEventListener('click', () => { modal.hidden = true; });
  modal.querySelector('.ccm-backdrop').addEventListener('click', () => { modal.hidden = true; });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.hidden) modal.hidden = true;
  });
}

/* ── Boot ──────────────────────────────────────────────────── */

function initChatTab() {
  document.querySelectorAll('.app-tab').forEach(b =>
    b.addEventListener('click', () => setAppTab(b.dataset.tab)));

  const form = document.getElementById('chat-form');
  const input = document.getElementById('chat-input');
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (text && !state.running) sendChat(text);
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener('input', () => {
    autoGrowTextarea(input);
    document.getElementById('chat-send').disabled = !input.value.trim() || state.running;
  });

  initChatConfig();

  document.querySelectorAll('.ev-mode-btn').forEach(b => {
    b.addEventListener('click', () => {
      chatState.evMode = b.dataset.mode;
      document.querySelectorAll('.ev-mode-btn').forEach(x =>
        x.classList.toggle('active', x === b));
      updateEvidenceLibrary();
    });
  });

  setAppTab(prefs.get('appTab', 'chat'));
}

document.addEventListener('DOMContentLoaded', initChatTab);
