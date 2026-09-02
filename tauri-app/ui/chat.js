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
  // Set per-question by the replay prompt below.
  useCacheForNext: false,
  replayAnswerForNext: false,
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

  // Debug cache: if this exact question has already been run against this
  // exact corpus, offer to replay it instead of paying for retrieval again.
  chatState.useCacheForNext = false;
  chatState.replayAnswerForNext = false;
  const hit = await lookupCachedRun(text);
  if (hit) {
    const choice = await askReplay(hit);
    if (choice === 'cancel') return;
    chatState.useCacheForNext = choice !== 'live';
    chatState.replayAnswerForNext = choice === 'replay-answer';
  }

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
  startStatusPolling();
  updateFlowSteps();

  try {
    const data = await apiPost('/api/chat', {
      query: text,
      index_module: selectedModules().index_module,
      selected_answers: selectedAnswers(),
      selection_mode: selectionMode(),
      use_cache: chatState.useCacheForNext,
      replay_answer: chatState.replayAnswerForNext,
    });
    // Feed the retrieval tab the identical run state (the two tabs talk).
    state.currentResults = data.run;
    renderResults(data.run);
    finalizeAnswerCard(pending, data);
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
  // Not a verdict: the agent's context filled before the ranking reached it.
  deferred:  ['⋯', 'Not read'],
  error:     ['!', 'Failed'],
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

/* Reveal the answer at ~900 characters/second — above skim speed, so the
   answer is seen ARRIVING rather than pasted, without anyone waiting on the
   animation. Re-rendering the prefix each frame (rather than typing raw HTML)
   is what keeps a half-written "[4" from flashing as a broken citation chip. */
const TYPE_CPS = 900;

function typeSequence(queue, sources, answerId) {
  let k = 0;
  (function next() {
    if (k >= queue.length) return;
    const [node, text] = queue[k++];
    typeInto(node, text, sources, answerId, next);
  })();
}

function typeInto(node, text, sources, answerId, done) {
  text = (text || '').trim();
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduce || text.length > 2200) {   // long answers paste; nobody waits on a reveal
    node.innerHTML = renderRichText(text, sources, answerId);
    if (done) done();
    return;
  }
  let i = 0, last = performance.now();
  node.classList.add('typing');
  requestAnimationFrame(function frame(now) {
    const dt = Math.min(80, now - last); last = now;
    i = Math.min(text.length, i + Math.max(1, Math.round((TYPE_CPS * dt) / 1000)));
    node.innerHTML = renderRichText(text.slice(0, i), sources, answerId);
    if (i < text.length) requestAnimationFrame(frame);
    else { node.classList.remove('typing'); if (done) done(); }
  });
}

/* ══ Evidence deck ═══════════════════════════════════════════════════
   Retrieval finishes long before synthesis does — measured on this corpus,
   ~27s of reasoning before the first answer token. The passages are already
   sitting there for all of it, so the wait shows them instead of a spinner.

   It mounts INSIDE the pending answer card rather than taking the screen:
   the chat history, the trace and the composer all stay where they are, and
   the deck collapses into the answer when it arrives. */

/* One hue per top-level folder. The OUTERMOST folder picks the colour, so
   everything under research/ reads as one family however deep it nests; the
   chip names the innermost folder, so a card still says where it came from. */
const FOLDER_HUES = {
  guidelines: '#1d4ed8',
  internal:   '#b45309',
  research:   '#7e22ce',
  preprints:  '#a855f7',
  '':         '#52525b',
};

function folderHue(folder) {
  const outer = (folder && folder.length) ? folder[0] : '';
  return FOLDER_HUES[outer] || FOLDER_HUES[''];
}
function chipHue(folder) {
  const leaf = (folder && folder.length) ? folder[folder.length - 1] : '';
  return FOLDER_HUES[leaf] || folderHue(folder);
}
function folderLabel(folder) {
  return (folder && folder.length) ? folder.join(' / ') : 'library root';
}

/* Source colour by node id, so citation chips and source cards in the finished
   answer carry the same hue the deck used. Populated when the deck mounts. */
const sourceHues = new Map();

function mountDeck(p, cards) {
  if (!cards || !cards.length || p.deck) return;
  // Keyed by citation number, which restarts at 1 for every question — so the
  // previous question's hues have to go, or [1] would keep a colour from a
  // document this answer never cited.
  sourceHues.clear();
  cards.forEach(c => sourceHues.set(c.n, folderHue(c.folder)));

  // The trace has done its job by now; collapsing it is what makes room.
  p.trace.classList.remove('open');

  const deck = { i: 0, seen: new Set([0]), cards, engaged: false, held: null };
  const wrap = el('div', 'deck-wrap');

  const head = el('div', 'deck-head');
  head.appendChild(el('span', 'deck-eyebrow', 'Evidence found'));
  head.appendChild(el('span', 'deck-sub',
    `${cards.length} passage${cards.length === 1 ? '' : 's'} · read them while the answer is written`));
  wrap.appendChild(head);

  const stack = el('div', 'deck-stack');
  cards.forEach((c, i) => {
    const card = el('article', 'ev-card');
    card.dataset.i = String(i);
    card.style.setProperty('--folder', folderHue(c.folder));

    const ch = el('div', 'ev-head');
    const chip = el('span', 'ev-chip' + (c.folder && c.folder.length > 1 ? ' nested' : ''),
                    folderLabel(c.folder));
    chip.style.setProperty('--folder', chipHue(c.folder));
    ch.appendChild(chip);
    ch.appendChild(el('span', 'ev-doc', prettyDoc(c.doc)));
    ch.appendChild(el('span', 'ev-n', `[${c.n}]`));
    card.appendChild(ch);

    card.appendChild(el('h4', 'ev-title', c.title));
    if (c.breadcrumb) card.appendChild(el('div', 'ev-crumb', c.breadcrumb));

    const body = el('div', 'ev-body');
    body.appendChild(el('p', 'ev-quote', c.quote || '(no verbatim quote was returned)'));
    card.appendChild(body);

    if (c.reason) {
      const why = el('div', 'ev-why');
      why.appendChild(el('b', null, 'Why this was kept'));
      why.appendChild(document.createTextNode(c.reason));
      card.appendChild(why);
    }
    card.addEventListener('click', () => deckGo(deck, 1));
    stack.appendChild(card);
  });
  wrap.appendChild(stack);
  wrap.appendChild(el('div', 'deck-controls'));

  const body = p.card.querySelector('.answer-body');
  body.innerHTML = '';
  body.appendChild(wrap);

  deck.wrap = wrap;
  deck.stack = stack;
  p.deck = deck;
  layoutDeck(deck);
  scrollChatToBottom();
}

function layoutDeck(deck) {
  deck.stack.querySelectorAll('.ev-card').forEach(card => {
    const rel = Number(card.dataset.i) - deck.i;
    card.classList.toggle('front', rel === 0);
    if (rel < 0) {
      card.style.transform = 'translate(-118%, -4%) rotate(-7deg) scale(.95)';
      card.style.opacity = '0'; card.style.zIndex = '0'; card.style.pointerEvents = 'none';
    } else if (rel === 0) {
      card.style.transform = 'none';
      card.style.opacity = '1'; card.style.zIndex = '30'; card.style.pointerEvents = 'auto';
    } else if (rel <= 3) {
      card.style.transform =
        `translateY(${rel * 8}px) rotate(${rel * 0.8}deg) scale(${1 - rel * 0.03})`;
      card.style.opacity = String(1 - rel * 0.24);
      card.style.zIndex = String(30 - rel); card.style.pointerEvents = 'none';
    } else {
      card.style.transform = 'translateY(24px) scale(.91)';
      card.style.opacity = '0'; card.style.zIndex = '0'; card.style.pointerEvents = 'none';
    }
  });
  renderDeckControls(deck);
}

function deckGo(deck, delta) {
  const next = deck.i + delta;
  if (next < 0 || next >= deck.cards.length) return;
  deck.i = next;
  deck.seen.add(next);
  deck.engaged = true;
  layoutDeck(deck);
}

function renderDeckControls(deck) {
  const c = deck.wrap.querySelector('.deck-controls');
  c.innerHTML = '';

  const prev = el('button', 'deck-btn', '← Back');
  prev.type = 'button';
  prev.disabled = deck.i === 0;
  prev.addEventListener('click', e => { e.stopPropagation(); deckGo(deck, -1); });
  c.appendChild(prev);

  const dots = el('div', 'deck-dots');
  deck.cards.forEach((card, i) => {
    const d = el('button', 'deck-dot' + (i === deck.i ? ' on' : (deck.seen.has(i) ? ' seen' : '')));
    d.type = 'button';
    d.style.setProperty('--folder', folderHue(card.folder));
    d.title = card.title;
    d.setAttribute('aria-label', card.title);
    d.addEventListener('click', e => {
      e.stopPropagation();
      deck.i = i; deck.seen.add(i); deck.engaged = true; layoutDeck(deck);
    });
    dots.appendChild(d);
  });
  c.appendChild(dots);

  // Held answer: the reader is mid-card, so the answer waits for a click
  // rather than yanking the card away. See finalizeAnswerCard.
  if (deck.held) {
    const b = el('button', 'deck-btn primary', 'Read the answer →');
    b.type = 'button';
    b.addEventListener('click', e => { e.stopPropagation(); deck.held(); });
    c.appendChild(b);
  } else {
    const nxt = el('button', 'deck-btn', 'Next →');
    nxt.type = 'button';
    nxt.disabled = deck.i === deck.cards.length - 1;
    nxt.addEventListener('click', e => { e.stopPropagation(); deckGo(deck, 1); });
    c.appendChild(nxt);
  }
}

/* How long the model call behind a decision took.

   Two shapes, and conflating them would misreport the cost of retrieval:
   a leaf evaluation is one call for one passage, while pruning decides a
   whole sibling group in ONE comparative call — so that duration is shared,
   and the line says so rather than implying each sibling cost it. Nodes
   pruned by inheritance carry no timing at all, because they cost no call. */
function traceTiming(m) {
  if (!m || typeof m.ms !== 'number') return '';
  const secs = m.ms >= 1000 ? `${(m.ms / 1000).toFixed(1)}s` : `${m.ms}ms`;
  if (m.ms_shared > 1) {
    return ` <span class="trace-ms shared" title="One comparative call decided`
      + ` ${m.ms_shared} sibling sections; this is that call's total time.">`
      + `${secs} · 1 call / ${m.ms_shared}</span>`;
  }
  return ` <span class="trace-ms" title="Model call for this passage,`
    + ` retries included.">${secs}</span>`;
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

  // Retrieval is done and synthesis has started: the passages exist now, so
  // show them rather than spinning for the ~27s the model spends reasoning.
  if (status.chat?.phase === 'synthesis' && status.chat.cards && !p.deck) {
    mountDeck(p, status.chat.cards);
  }

  const meta = status.live?.meta || {};
  let added = false;
  for (const [nodeId, m] of Object.entries(meta)) {
    if (p.seen.has(nodeId)) continue;
    p.seen.add(nodeId);
    const [ico, word] = TRACE_META[m.status] || ['·', m.status];
    const line = el('div', `trace-line ${m.status}`);
    const info = chatState.nodeInfo[nodeId] || {};
    const label = info.title || nodeId;
    const docTag = info.doc ? ` <span class="trace-doc">· ${escHtml(prettyDoc(info.doc))}</span>` : '';
    line.innerHTML = `<span class="trace-ico">${ico}</span><span>` +
      `<span class="trace-what">${word} — ${escHtml(label)}</span>${docTag}` +
      traceTiming(m) +
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
    const hue = sourceHues.get(num);
    const tint = hue ? ` style="--folder:${hue}"` : '';
    return `<button type="button" class="cite-chip${hue ? ' tinted' : ''}"${tint} data-cite="${num}" data-answer="${answerId}" title="Jump to source [${num}]">${num}</button>`;
  });
  return html;
}

/* ── Debug cache — replay prompt and autocomplete ─────────────
   Exact-key replay only. The prompt is per-question rather than a silent
   toggle so a replayed answer is never mistaken for a fresh run. */

async function lookupCachedRun(text) {
  try {
    // The pre-filter answers and mode are PART of the cache key, so they have
    // to be part of the lookup — asking without them tests a key the run will
    // never use, and offers a replay of a different retrieval.
    const params = new URLSearchParams({
      query: text,
      selected_answers: selectedAnswers().join(','),
      selection_mode: selectionMode(),
    });
    const r = await apiGet(`/api/cache/lookup?${params}`);
    return (r && r.enabled && r.hit) ? r : null;
  } catch { return null; }
}

function askReplay(hit) {
  return new Promise((resolve) => {
    const age = hit.age_seconds || 0;
    const ago = age < 90 ? `${age} seconds ago`
      : age < 5400 ? `${Math.round(age / 60)} minutes ago`
      : `${Math.round(age / 3600)} hours ago`;

    const overlay = el('div', 'replay-overlay');
    const box = el('div', 'replay-box');
    box.appendChild(el('h3', '', 'You have asked this before'));
    box.appendChild(el('p', 'replay-detail',
      `Run ${ago}, retrieving ${hit.node_count} passage${hit.node_count === 1 ? '' : 's'} ` +
      `from ${hit.doc_count} document${hit.doc_count === 1 ? '' : 's'}. ` +
      `The index has not changed since.`));

    const actions = el('div', 'replay-actions');
    const add = (label, value, cls, title) => {
      const b = el('button', cls, label);
      if (title) b.title = title;
      b.addEventListener('click', () => { overlay.remove(); resolve(value); });
      actions.appendChild(b);
    };
    add('Replay retrieval, re-answer', 'replay', 'primary',
        'Reuse the cached passages but generate a fresh answer — the usual choice when iterating on the synthesis prompt.');
    if (hit.has_answer) {
      add('Replay everything', 'replay-answer', '',
          'Reuse the cached passages AND the cached answer.');
    }
    add('Re-run live', 'live', '', 'Ignore the cache and retrieve again.');
    add('Cancel', 'cancel', 'ghost');
    box.appendChild(actions);

    overlay.appendChild(box);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) { overlay.remove(); resolve('cancel'); }
    });
    document.body.appendChild(overlay);
  });
}

async function loadCachedQuestions() {
  try {
    const r = await apiGet('/api/cache/questions');
    return (r && r.enabled) ? (r.questions || []) : [];
  } catch { return []; }
}

/* What the pre-filter did to THIS answer.

   The filter runs before any model call, so a passage it drops leaves no
   trace anywhere else in the answer — no "rejected" mark, no reasoning,
   nothing. Whatever it removed has to be stated here or it is invisible. */
function buildChoiceNotice(choices) {
  if (!choices || !choices.applied) return null;

  const wrap = el('div', 'choice-notice');
  const n = (choices.selected_answers || []).length;
  const fast = choices.mode === 'all';
  const text = el('span');
  text.innerHTML =
    `<strong>Filtered by your ${n} answer${n === 1 ? '' : 's'}</strong> — ` +
    `${choices.kept} of ${choices.total} passages searched` +
    (fast ? ', in Fast mode (a passage had to match every question)' : '') + '.';
  wrap.appendChild(text);

  // A document searched WHOLE because it has no judgements is the one thing
  // here that is a coverage hole rather than a choice, so it is named.
  const unfiltered = choices.docs_unfiltered || [];
  if (unfiltered.length) {
    const warn = el('span', 'choice-notice-warn');
    warn.textContent =
      ` ${unfiltered.map(d => d.replace(/_/g, ' ')).join(', ')} ` +
      `${unfiltered.length === 1 ? 'has' : 'have'} no judgements yet and ` +
      `${unfiltered.length === 1 ? 'was' : 'were'} searched unfiltered.`;
    wrap.appendChild(warn);
  }
  return wrap;
}

function buildBudgetNotice(budget) {
  if (!budget || !budget.deferred) return null;

  const n = budget.deferred;
  const wrap = el('div', 'budget-notice');

  const used = budget.tokens_max
    ? Math.min(100, Math.round((budget.tokens_used / budget.tokens_max) * 100))
    : 100;
  const bar = el('div', 'budget-bar');
  const fill = el('i');
  fill.style.width = `${used}%`;
  bar.appendChild(fill);
  wrap.appendChild(bar);

  const text = el('span');
  const cap = budget.capped_by === 'max_evals'
    ? 'the per-run evaluation cap was reached'
    : `the agent's context window filled up (${budget.tokens_used.toLocaleString()} of ` +
      `${budget.tokens_max.toLocaleString()} tokens)`;
  text.innerHTML =
    `<strong>${n} further passage${n === 1 ? '' : 's'} ranked below the cut were not read</strong> — ` +
    `${escHtml(cap)}. They were not judged irrelevant.`;
  wrap.appendChild(text);

  const btn = el('button', '', 'Browse them');
  btn.title = 'Open the retrieval tab filtered to the passages that were not read';
  btn.addEventListener('click', () => {
    window.__deferredFilter = (budget.deferred_nodes || []).map(d => d.node_id);
    setAppTab('retrieval');
    setView('results');
    toast(`${n} passage${n === 1 ? '' : 's'} highlighted as "not read"`, 'info', 5000);
  });
  wrap.appendChild(btn);

  return wrap;
}

function finalizeAnswerCard(p, data) {
  // Someone reading a card should not have it yanked away mid-sentence, so a
  // reader who has actually flipped through the deck gets a button instead.
  // A passive one — deck untouched — sees the answer immediately, because for
  // them the deck was never more than a progress indicator.
  if (p.deck && p.deck.engaged) {
    p.deck.held = () => { p.deck.held = null; applyAnswer(p, data); };
    renderDeckControls(p.deck);
    const txt = p.progressEl.querySelector('.trace-progress-text');
    if (txt) txt.textContent = 'Answer ready';
    return;
  }
  applyAnswer(p, data);
}

function applyAnswer(p, data) {
  // No deck mounted (a cached replay never reaches the synthesis phase), so
  // there is no provenance to show. Untinted beats wrongly tinted.
  if (!p.deck) sourceHues.clear();
  const answerId = `a${++chatState.answerCount}`;
  p.card.id = `answer-${answerId}`;
  chatState.activeAnswerId = answerId;

  const sources = data.grounding?.sources || [];
  const status = data.grounding?.status || 'not_connected';

  // Header: grounding badge, plus a CACHED marker when this run was replayed
  // rather than retrieved. Deliberately unmissable — a replayed answer is not
  // evidence that the current pipeline still behaves this way.
  const head = p.card.querySelector('.answer-head');
  head.appendChild(groundingBadge(status));
  if (data.cache?.cached) {
    const badge = el('span', 'cached-badge');
    const age = data.cache.age_seconds || 0;
    const ago = age < 90 ? `${age}s ago`
      : age < 5400 ? `${Math.round(age / 60)}m ago`
      : `${Math.round(age / 3600)}h ago`;
    badge.textContent = `CACHED · ${ago}`;
    badge.title = 'Retrieval was replayed from the debug cache, not re-run.';
    head.appendChild(badge);
  }

  // Body: structured sections, else the raw answer
  const body = p.card.querySelector('.answer-body');
  body.innerHTML = '';
  const a = data.answer || {};
  const present = chatCopy.sections.filter(([key]) => (a[key] || '').trim());
  const typeQueue = [];
  if (present.length) {
    for (const [key, label] of present) {
      const sec = el('section', 'answer-section');
      const h = el('h3', '', label);
      const txt = el('p', 'answer-text');
      sec.appendChild(h); sec.appendChild(txt);
      body.appendChild(sec);
      typeQueue.push([txt, a[key]]);
    }
    typeSequence(typeQueue, sources, answerId);
  } else if ((a.content || '').trim()) {
    const div = el('div', 'answer-fallback');
    div.innerHTML = renderRichText(a.content, sources, answerId);
    body.appendChild(div);
  } else {
    body.appendChild(el('p', 'chat-loading', 'The model returned no answer text — see the sources below.'));
  }

  // What the evidence set was narrowed by, before and during retrieval.
  // Deliberately placed with the answer, not buried in the trace: the user is
  // being told their answer was composed from a reduced evidence set.
  const choiceNotice = buildChoiceNotice(data.choices);
  if (choiceNotice) body.appendChild(choiceNotice);

  const budgetNotice = buildBudgetNotice(data.budget);
  if (budgetNotice) body.appendChild(budgetNotice);

  // Trace: freeze into a collapsed audit block
  p.progressEl.remove();
  p.trace.classList.remove('open');
  if (p.stats.total === 0) p.trace.remove();

  // Sources
  const srcWrap = el('div', 'answer-sources');
  const rule = el('div', 'answer-sources-rule');
  rule.appendChild(el('p', 'sources-label', chatCopy.sourcesLabel));
  rule.appendChild(el('p', 'sources-summary', data.grounding?.summary || ''));
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
  const card = el('article', 'source-card');
  card.id = `src-${answerId}-${s.n}`;

  const eyebrow = el('p', 'source-eyebrow');
  eyebrow.innerHTML = `[${s.n}] ${escHtml(prettyDoc(s.doc))}` +
    (s.page ? ` <span class="source-page-pill">p. ${escHtml(String(s.page))}</span>` : '') +
    (s.synthetic ? ` <span class="source-page-pill">overview</span>` : '');
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

  // Visual citation: asset crop, or the source-PDF page with the bbox highlighted
  const pin = s.pin || null;
  if (s.image) {
    const btn = el('button', 'source-preview asset');
    btn.type = 'button';
    btn.innerHTML = `<img loading="lazy" src="${escHtml(s.image)}" alt="${escHtml(s.title)}">` +
      `<span class="preview-hint">${ICONS.zoom} Figure from the original document — click to open the source page</span>`;
    btn.addEventListener('click', () => pin?.page ? openSourceView(s.doc, pin) : null);
    card.appendChild(btn);
  } else if (s.page && s.has_source_pdf) {
    const params = new URLSearchParams();
    if (Array.isArray(pin?.bbox)) params.set('bbox', pin.bbox.join(','));
    if (Array.isArray(pin?.regions)) params.set('regions', JSON.stringify(pin.regions));
    const btn = el('button', 'source-preview');
    btn.type = 'button';
    btn.innerHTML = `<img loading="lazy" src="/api/document/${encodeURIComponent(s.doc)}/page/${s.page}?${params}" alt="page ${s.page}">` +
      `<span class="preview-hint">${ICONS.zoom} Original page ${s.page}, passage highlighted — click for fullscreen</span>`;
    // Aim the cropped preview strip at the highlighted bbox, not the page top.
    const img = btn.querySelector('img');
    img.addEventListener('load', () => {
      if (Array.isArray(pin?.bbox) && img.naturalHeight > 0) {
        const centerY = ((pin.bbox[1] + pin.bbox[3]) / 2) / img.naturalHeight * 100;
        img.style.objectPosition = `center ${Math.max(0, Math.min(100, centerY))}%`;
      }
    });
    btn.addEventListener('click', () => openSourceView(s.doc, pin));
    card.appendChild(btn);
  }

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
  if (pin?.page && s.has_source_pdf) {
    const b = el('button', 'source-action');
    b.type = 'button';
    b.innerHTML = `${ICONS.zoom} Open source page`;
    b.addEventListener('click', () => openSourceView(s.doc, pin));
    actions.appendChild(b);
  } else {
    // No provenance pin or the source PDF is unavailable — keep the button,
    // but as an explicit warning instead of silently hiding the affordance.
    const b = el('button', 'source-action warn');
    b.type = 'button';
    b.innerHTML = `${ICONS.alert} Open source page`;
    b.title = 'No source PDF is available for this passage';
    b.addEventListener('click', () => toast(
      `No source PDF was found for “${prettyDoc(s.doc)}”. The original may have been ` +
      `markdown-only (no page provenance), the PDF may have been moved — or the ` +
      `reference could be a hallucination. Verify against the cited text.`,
      'warn', 8000));
    actions.appendChild(b);
  }
  const read = el('button', 'source-action');
  read.type = 'button';
  read.innerHTML = `${ICONS.book} Read in document`;
  read.title = 'Open the annotated reader at this section';
  read.addEventListener('click', () => openDocViewer(s.doc, s.title));
  actions.appendChild(read);

  const mapBtn = el('button', 'source-action');
  mapBtn.type = 'button';
  mapBtn.innerHTML = `${ICONS.map} Show in retrieval map`;
  mapBtn.title = 'Switch to the Retrieval tab, focused on this passage';
  mapBtn.addEventListener('click', () => openInRetrieval(s.doc, s.node_id));
  actions.appendChild(mapBtn);
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
      summary: m.data.grounding.summary,
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
      btn.dataset.n = s.n;
      const eyebrow = `<p class="ev-source-eyebrow"><span>[${s.n}] ${escHtml(prettyDoc(s.doc))}</span>` +
        (s.page ? `<span class="ev-page">p. ${escHtml(String(s.page))}</span>` : '') + `</p>`;
      btn.innerHTML = eyebrow +
        `<p class="ev-source-title">${escHtml(s.title)}</p>` +
        (s.quote ? `<p class="ev-source-quote">“${escHtml(s.quote)}”</p>` : '');
      btn.addEventListener('click', () => focusSourceCard(g.answerId, s.n));
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
    : [cfg.model, 'gemma3:4b', 'gemma4:e2b'];

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
  fact('Pipeline', ['ingest', 'index', 'query']
    .map(s => `${s}: <code>${escHtml(cfg.pipeline?.[s] || '—')}</code>`).join(' · '));
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
