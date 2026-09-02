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
  cancelled: false,
  // 'followup' talks over the passages already retrieved; 'search' runs a new
  // retrieval. Chosen by the reader, never inferred — there is no router.
  mode: 'search',
  ctx: null,               // last /api/chat/context payload, for the meter
  activeRunId: null,       // server-side run the composer is talking to
  // A deck waiting on "Read the answer" outlives the run that built it:
  // sendChat's `finally` clears chatState.pending while the deck is still on
  // screen, which is what left ↵ pointing at nothing.
  heldDeck: null,
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
  // While a run is live the button STOPS it. Same control, and its label
  // always names what pressing it does — a disabled "Sending…" told the user
  // nothing and left them no way out of a slow run.
  btn.classList.toggle('stopping', !enabled);
  btn.disabled = enabled && !input.value.trim();
  // The label names the mode, not a generic "Send": the two buttons above do
  // materially different things and the reader should be able to tell which
  // one ↵ is about to fire.
  document.getElementById('chat-send-label').textContent = !enabled ? 'Stop'
    : (chatState.mode === 'followup' && chatState.activeAnswerId) ? 'Ask' : 'Search';
}

async function cancelRun() {
  chatState.cancelled = true;
  // A rewrite is stopped from its own card, with the composer sitting idle —
  // relabelling the send button "Stopping" there would name the wrong control.
  const label = state.running ? document.getElementById('chat-send-label') : null;
  if (label) label.textContent = 'Stopping';
  try { await apiPost('/api/chat/cancel', {}); }
  catch (e) { toast(`Could not stop the run: ${e.message}`, 'err', 6000); }
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

/* Bring the TOP of a card into view and then leave the scroll alone, so an
   answer writes itself downward from where the reader is looking. Jumping to
   the bottom put the typing off-screen above and landed the reader in the
   source list, which is the one part that was not being written. */
function revealCardTop(card, smooth = true) {
  const sc = document.getElementById('chat-scroll');
  if (!sc || !card) return;
  // Measured against the SCROLLER, not the offset parent: offsetTop is
  // relative to whichever ancestor happens to be positioned, which put the
  // card's header just above the fold.
  const target = Math.max(0,
    card.getBoundingClientRect().top - sc.getBoundingClientRect().top + sc.scrollTop - 14);
  if (Math.abs(sc.scrollTop - target) < 6) return;
  sc.scrollTo({ top: target, behavior: smooth ? 'smooth' : 'auto' });
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
  chatState.heldDeck = null;
  // The server drops the follow-up thread when a new run lands; the meter
  // stops describing it here so the two never disagree.
  chatState.ctx = null;
  renderContextMeter();

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
    stopComposer(deckComposer); deckComposer = null;
    if (chatState.cancelled || /\b499\b/.test(e.message || '')) {
      renderStoppedCard(pending);
    } else {
      pending.card.querySelector('.answer-body').innerHTML =
        `<p class="chat-error">The question could not be processed: ${escHtml(e.message)}</p>`;
      pending.progressEl.remove();
      toast(`Chat failed: ${e.message}`, 'err', 7000);
    }
  } finally {
    chatState.cancelled = false;
    state.running = false;
    setRunPill(null);
    await stopStatusPolling();     // final status flush → last trace lines land
    chatState.pending = null;
    setSendEnabled(true);
    updateFlowSteps();
    if (chatState.tab === 'chat') document.getElementById('chat-input').focus();
  }
}

/* ── 3b · Follow-up conversation ──────────────────────────────
   The second half of the pair the composer offers. Retrieval is NOT re-run:
   these turns reason over the passages the last answer was built from, which
   is what lets [n] keep meaning the same passage and lets a chip jump to a
   source card that is already on screen. Asking for something the passages do
   not hold is answered with "a new search would be needed" rather than a
   guess — which is why the other button exists. */

function setChatMode(mode) {
  chatState.mode = mode;
  document.querySelectorAll('.mode-tab').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode));
  const input = document.getElementById('chat-input');
  input.placeholder = mode === 'followup'
    ? 'Ask about this answer — the passages above are the evidence…'
    : 'Ask a new question — this searches the library again…';
  const label = document.getElementById('chat-send-label');
  if (!state.running) label.textContent = mode === 'followup' ? 'Ask' : 'Search';
  document.getElementById('ctx-meter').hidden = mode !== 'followup' || !chatState.ctx;
}

/* The mode bar only appears once there IS an answer to talk about — before
   that "ask about this answer" would be an affordance for nothing. */
function updateModeBar() {
  const bar = document.getElementById('chat-modebar');
  const has = !!chatState.activeAnswerId;
  bar.hidden = !has;
  if (!has && chatState.mode === 'followup') setChatMode('search');
}

async function refreshContextMeter() {
  try {
    const q = chatState.activeRunId
      ? `?run_id=${encodeURIComponent(chatState.activeRunId)}` : '';
    const r = await apiGet(`/api/chat/context${q}`);
    chatState.ctx = (r && r.available) ? r : null;
    if (r && r.run_id) chatState.activeRunId = r.run_id;
  } catch { chatState.ctx = null; }
  renderContextMeter();
}

function renderContextMeter() {
  const meter = document.getElementById('ctx-meter');
  const c = chatState.ctx;
  if (!c || chatState.mode !== 'followup') { meter.hidden = true; return; }
  meter.hidden = false;

  const seg = c.segments || {};
  const bar = document.getElementById('ctx-bar');
  bar.innerHTML = '';
  // Segments in the order they are pinned: what can never be dropped first,
  // what gets dropped last. The bar is a picture of the drop order.
  const parts = [
    ['instructions', seg.instructions || 0],
    ['passages',     seg.passages || 0],
    ['answer',       seg.answer || 0],
    ['conversation', seg.conversation || 0],
  ];
  for (const [name, tok] of parts) {
    if (!tok) continue;
    const i = el('i', `ctx-seg ctx-${name}`);
    i.style.width = `${Math.min(100, (tok / c.max) * 100)}%`;
    bar.appendChild(i);
  }
  const pct = Math.round((c.used / c.max) * 100);
  meter.classList.toggle('tight', pct >= 80);
  document.getElementById('ctx-label').textContent =
    `${fmtTokens(c.used)} / ${fmtTokens(c.max)}` + (c.dropped_turns ? ' · trimmed' : '');
  meter.title = c.dropped_turns
    ? `${c.dropped_turns} earlier turn${c.dropped_turns === 1 ? '' : 's'} dropped to fit`
    : 'What this conversation is holding in the model’s context window';
}

function fmtTokens(n) {
  return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n);
}

function toggleContextPopover() {
  const pop = document.getElementById('ctx-popover');
  const c = chatState.ctx;
  if (!pop.hidden || !c) { pop.hidden = true; return; }

  pop.innerHTML = '';
  pop.appendChild(el('p', 'ctx-pop-eyebrow', 'Context window'));
  pop.appendChild(el('p', 'ctx-pop-sub',
    `${fmtTokens(c.used)} of ${fmtTokens(c.max)} tokens, with ${fmtTokens(c.reserve)} `
    + `held back for the reply. Counts are estimates, calibrated against the `
    + `real token counts the model reports.`));

  const rows = [
    ['Instructions', c.segments.instructions, 'ctx-instructions', 'The follow-up prompt itself. Pinned.'],
    ['Retrieved passages', c.segments.passages, 'ctx-passages', 'The evidence the answer was built from. Pinned — dropping one would break its [n].'],
    ['The answer', c.segments.answer, 'ctx-answer', 'What is being discussed. Pinned.'],
    ['This conversation', c.segments.conversation, 'ctx-conversation',
      `${c.turns} turn${c.turns === 1 ? '' : 's'} kept. Oldest are dropped first when the window fills.`],
  ];
  const list = el('div', 'ctx-pop-rows');
  for (const [label, tok, cls, note] of rows) {
    const row = el('div', 'ctx-pop-row');
    row.appendChild(el('i', `ctx-swatch ${cls}`));
    const t = el('div', 'ctx-pop-text');
    t.appendChild(el('b', null, `${label} · ${fmtTokens(tok || 0)}`));
    t.appendChild(el('span', null, note));
    row.appendChild(t);
    list.appendChild(row);
  }
  pop.appendChild(list);

  if (c.dropped_turns) {
    pop.appendChild(el('p', 'ctx-pop-warn',
      `${c.dropped_turns} earlier turn${c.dropped_turns === 1 ? '' : 's'} no longer `
      + `fit and ${c.dropped_turns === 1 ? 'was' : 'were'} dropped from the model's `
      + `view. The passages and the answer are never dropped, so citations stay valid.`));
  }
  const cleared = c.restorable;
  const reset = el('button', 'ctx-pop-reset',
    cleared ? 'Restore the conversation' : 'Clear the conversation');
  reset.type = 'button';
  reset.title = cleared
    ? 'Bring the cleared turns back into the model’s context.'
    : 'Keeps the answer and its passages; sets the follow-up turns aside. '
      + 'They stay on screen, greyed, and can be brought back.';
  reset.addEventListener('click', async () => {
    const runId = chatState.activeRunId;
    try {
      await apiPost(`/api/chat/followup/${cleared ? 'restore' : 'reset'}`,
                    { run_id: runId });
    } catch (e) {
      toast(`Could not ${cleared ? 'restore' : 'clear'}: ${e.message}`, 'err', 6000);
      return;
    }
    pop.hidden = true;
    setFollowupCleared(runId, !cleared);
    await refreshContextMeter();
    updateEvidenceLibrary();
    toast(cleared
      ? 'Conversation restored — the model can see those turns again.'
      : 'Conversation cleared. The turns are greyed out and can be restored.',
      'info', 5000);
  });
  pop.appendChild(reset);
  pop.hidden = false;
}

/* Grey (or un-grey) a run's follow-up turns. They are still readable — the
   point is that the MODEL can no longer see them, which is a different thing
   from the user no longer having them. */
function setFollowupCleared(runId, cleared) {
  if (!runId) return;
  chatState.clearedRuns = chatState.clearedRuns || new Set();
  if (cleared) chatState.clearedRuns.add(runId);
  else chatState.clearedRuns.delete(runId);
  document.querySelectorAll(`[data-followup-run="${runId}"]`).forEach(row =>
    row.classList.toggle('followup-cleared', cleared));
}

/* The sources the follow-up cites are the active answer's, so a chip can
   point at a source card that is already rendered rather than re-rendering
   the evidence under every turn. */
function activeAnswerSources() {
  const m = chatState.messages.find(
    x => x.role === 'assistant' && x.id === chatState.activeAnswerId);
  return m ? (m.data?.grounding?.sources || []) : [];
}

function appendFollowupBubble(role, text, { pending = false } = {}) {
  const messagesEl = document.getElementById('chat-messages');
  const row = el('div', `chat-row ${role === 'user' ? 'user' : 'assistant'}`);
  // Which conversation this turn belongs to — clearing greys these, and only
  // these, rather than every bubble on the page.
  if (chatState.activeRunId) row.dataset.followupRun = chatState.activeRunId;
  if (role === 'user') {
    row.appendChild(el('div', 'chat-user-bubble', text));
  } else {
    const bubble = el('div', 'chat-reply' + (pending ? ' pending' : ''));
    if (pending) {
      // Same widget the deck uses while an answer is written: activity and an
      // elapsed clock, not a fake percentage. A follow-up call costs the same
      // ~20-30s as the first answer, and a button that merely stops responding
      // does not tell anyone whether the model is working or wedged.
      bubble.appendChild(buildComposer(activeAnswerSources().length,
        { title: 'Thinking it over',
          sub: 'reading the passages already retrieved' }));
    } else {
      bubble.innerHTML = renderRichText(text, activeAnswerSources(),
                                        chatState.activeAnswerId);
      bubble.addEventListener('click', (e) => {
        const chip = e.target.closest('.cite-chip');
        if (chip) focusSourceCard(chip.dataset.answer, parseInt(chip.dataset.cite, 10));
      });
      const foot = el('p', 'reply-foot',
        'From the passages already retrieved — no new search was run.');
      bubble.appendChild(foot);
    }
    row.appendChild(bubble);
  }
  messagesEl.appendChild(row);
  scrollChatToBottom();
  return row;
}

async function sendFollowup(text) {
  if (state.running) { toast('A run is already in progress.', 'warn'); return; }
  if (!chatState.activeAnswerId) { toast('There is no answer to discuss yet.', 'warn'); return; }

  chatState.messages.push({ role: 'user', content: text, followup: true });
  appendFollowupBubble('user', text);
  const waiting = appendFollowupBubble('assistant', '', { pending: true });

  const input = document.getElementById('chat-input');
  input.value = ''; input.style.height = '';
  state.running = true;
  setRunPill('Answering…');
  setSendEnabled(false);

  try {
    const data = await apiPost('/api/chat/followup',
                               { message: text, run_id: chatState.activeRunId });
    stopComposer(waiting.querySelector('.composing'));
    waiting.remove();
    appendFollowupBubble('assistant', data.reply || '(no reply)');
    chatState.messages.push({ role: 'assistant', content: data.reply, followup: true });
    chatState.ctx = { ...(chatState.ctx || {}), ...(data.context || {}), available: true };
    renderContextMeter();
    if (data.context && data.context.dropped_turns) {
      toast(`${data.context.dropped_turns} earlier turn${data.context.dropped_turns === 1 ? '' : 's'} `
            + `dropped — the context window is full.`, 'warn', 6000);
    }
  } catch (e) {
    stopComposer(waiting.querySelector('.composing'));
    waiting.remove();
    if (chatState.cancelled || /\b499\b/.test(e.message || '')) {
      appendFollowupBubble('assistant', '_Stopped before the reply was written._');
    } else {
      const row = appendFollowupBubble('assistant', '');
      row.querySelector('.chat-reply').innerHTML =
        `<p class="chat-error">The follow-up failed: ${escHtml(e.message)}</p>`;
    }
  } finally {
    chatState.cancelled = false;
    state.running = false;
    setRunPill(null);
    setSendEnabled(true);
    setChatMode(chatState.mode);          // restores the send label
    if (chatState.tab === 'chat') input.focus();
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

function typeSequence(queue, sources, answerId, whenDone) {
  let k = 0;
  (function next() {
    if (k >= queue.length) { if (whenDone) whenDone(); return; }
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

/* The one place the app asks the clinician for something rather than telling
   them something. Shared by both outcomes that can use it: an answer the
   model called incomplete, and a search that found nothing at all. The run is
   named explicitly so the rewrite lands on THIS answer — the composer may
   have moved on to a later one since. */
function buildRewriteForm(p, runId, { placeholder, action }) {
  const form = el('form', 'gap-form');
  const input = document.createElement('textarea');
  input.rows = 2;
  input.placeholder = placeholder;
  input.className = 'gap-input';
  const send = el('button', 'deck-btn primary', action);
  send.type = 'submit';
  form.appendChild(input);
  form.appendChild(send);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const extra = input.value.trim();
    if (!extra) return;

    // A rewrite is a full model call — the same ~20-30s the first answer took.
    // A button that greys out says nothing about whether it is working, and
    // left no way out of a slow one, so the form is replaced by the same
    // activity widget the rest of the app uses, with a Stop beside it.
    const wait = el('div', 'gap-waiting');
    const composer = buildComposer(0, {
      title: 'Rewriting the answer',
      sub: 'reading the passages again with your context',
    });
    wait.appendChild(composer);
    const stop = el('button', 'deck-btn gap-stop', 'Stop');
    stop.type = 'button';
    stop.title = 'Stop the rewrite and keep the answer as it is';
    stop.addEventListener('click', () => { stop.disabled = true; cancelRun(); });
    wait.appendChild(stop);
    form.replaceWith(wait);

    const restore = () => { stopComposer(composer); wait.replaceWith(form); };
    try {
      // Named explicitly: a rewrite must hit the answer whose form was
      // submitted, not whichever run happens to be most recent.
      const r = await apiPost('/api/chat/refine',
                              { context: extra, run_id: runId || chatState.activeRunId });
      stopComposer(composer);
      applyRefinedAnswer(p, r, extra);
    } catch (err) {
      const stopped = chatState.cancelled || /\b499\b/.test(err.message || '');
      restore();
      // What was typed is not thrown away — it is the whole point of the form.
      input.value = extra;
      input.disabled = false;
      toast(stopped ? 'Rewrite stopped — the answer is unchanged.'
                    : `Could not rewrite: ${err.message}`,
            stopped ? 'info' : 'err', 6000);
    } finally {
      chatState.cancelled = false;
    }
  });
  return form;
}

function buildNoEvidenceCard(data, pending, runId) {
  const wrap = el('div', 'outcome-card empty');
  wrap.appendChild(el('b', null, 'No passage in the library answered this'));

  const c = data.choices || {};
  const p = el('p');
  p.textContent = c.applied
    ? `Every document was searched, but your pre-filter answers narrowed it to `
      + `${c.kept} of ${c.total} passages and none of them matched.`
    : 'The whole library was searched and nothing matched closely enough to cite.';
  wrap.appendChild(p);

  const list = el('ul', 'outcome-next');
  if (c.applied) {
    list.appendChild(el('li', null, 'Clear the filters above the composer and ask again — they are the most likely cause.'));
  }
  list.appendChild(el('li', null, 'Try naming the condition and the decision explicitly ("which drug class", "what dose").'));
  list.appendChild(el('li', null, 'The library may simply not contain this topic — the retrieval trace below shows what was considered.'));
  wrap.appendChild(list);

  // Finding nothing is not the same as having nothing to say. The clinician
  // can still add what they know and have the answer rewritten around it —
  // which, with no passages to cite, will say honestly that the library holds
  // nothing rather than filling the gap from the model's own training.
  wrap.appendChild(el('p', 'outcome-aside',
    'You can still add what you know — the rewrite will be explicit that it '
    + 'has no document evidence behind it.'));
  wrap.appendChild(buildRewriteForm(pending, runId, {
    placeholder: 'Add context — what you already know about this case',
    action: 'Rewrite with my context',
  }));
  return wrap;
}

/* The evidence was thin, and the model said so. This is the one place the app
   asks the clinician for something rather than telling them something: what is
   missing is usually patient detail the documents cannot contain. */
function buildGapCard(a, p, runId) {
  const wrap = el('div', 'outcome-card gap');
  wrap.appendChild(el('b', null, 'This answer is incomplete'));

  const needed = (a.still_needed || '').split('\n')
    .map(l => l.replace(/^[-•*]\s*/, '').trim()).filter(Boolean);
  if (needed.length) {
    wrap.appendChild(el('p', null, 'To answer fully it would need:'));
    const ul = el('ul', 'outcome-next');
    needed.forEach(n => ul.appendChild(el('li', null, n)));
    wrap.appendChild(ul);
  } else {
    wrap.appendChild(el('p', null,
      'The retrieved passages did not cover every part of the question.'));
  }

  wrap.appendChild(buildRewriteForm(p, runId, {
    placeholder: 'Add the missing detail — e.g. the patient\'s age, renal function, or allergies',
    action: 'Rewrite the answer',
  }));
  return wrap;
}

/* Retrieval was not re-run, so the document passages stay exactly where the
   reader left them. What DOES change is the source list: the context the
   clinician supplied becomes a numbered source of its own, because the
   rewritten answer cites it — and an answer that leans on something with no
   card behind it cannot be checked. */
function applyRefinedAnswer(p, r, extra) {
  const body = p.card.querySelector('.answer-body');
  const a = r.answer || {};
  // The card this answer lives in, not whichever is active now: a rewrite of
  // an earlier answer must renumber ITS chips, not the latest one's.
  const answerId = (p.card.id || '').replace(/^answer-/, '')
    || chatState.activeAnswerId || 'a1';
  const sources = r.sources || [];
  body.innerHTML = '';

  const note = el('div', 'outcome-card refined');
  note.appendChild(el('b', null, 'Rewritten with your context'));
  note.appendChild(el('p', null, extra));
  // What the revision actually changed. The whole point of adding context is
  // that the answer may now differ — saying so beats making the reader diff
  // two paragraphs from memory.
  if ((a.what_changed || '').trim()) {
    const ch = el('p', 'refined-changed');
    ch.innerHTML = `<b>What changed</b> ${escHtml(a.what_changed.trim())}`;
    note.appendChild(ch);
  }
  if (r.context_source_n) {
    note.appendChild(el('p', 'outcome-aside',
      `Your context is source [${r.context_source_n}] below — the answer cites `
      + `it wherever it relies on it.`));
  }
  body.appendChild(note);

  const queue = [];
  for (const [key, label] of chatCopy.sections) {
    if (!(a[key] || '').trim()) continue;
    const sec = el('section', 'answer-section');
    sec.appendChild(el('h3', '', label));
    const txt = el('p', 'answer-text');
    sec.appendChild(txt);
    body.appendChild(sec);
    queue.push([txt, a[key]]);
  }
  if (!queue.length && (a.content || '').trim()) {
    const div = el('div', 'answer-fallback');
    div.innerHTML = renderRichText(a.content, sources, answerId);
    body.appendChild(div);
  }
  typeSequence(queue, sources, answerId, () => {});

  // Re-render the source list so the new context card is actually there for
  // its [n] to jump to.
  rerenderSources(p.card, sources, answerId);

  // Keep the in-memory record in step, or the follow-up would cite a source
  // list the reader can no longer see.
  const msg = chatState.messages.find(m => m.role === 'assistant' && m.id === answerId);
  if (msg) {
    msg.data.grounding = { ...(msg.data.grounding || {}), sources };
    msg.data.answer = a;
    updateEvidenceLibrary();
  }
  p.card.classList.remove('partial');
}

/* Replace an answer card's source list in place. */
function rerenderSources(card, sources, answerId) {
  const rule = card.querySelector('.answer-sources-rule');
  if (!rule) return;
  // An answer that retrieved nothing has no list at all, only the "no sources"
  // line — so a rewrite that adds a context card has to build one.
  let list = card.querySelector('.sources-list');
  if (!list) { list = el('div', 'sources-list'); rule.appendChild(list); }
  list.innerHTML = '';
  for (const s of sources) list.appendChild(buildSourceCard(s, answerId));
  const none = card.querySelector('.sources-none');
  if (none && sources.length) none.remove();
  const summary = card.querySelector('.sources-summary');
  if (summary) summary.textContent =
    `${sources.length} source${sources.length === 1 ? '' : 's'} under this answer.`;
}

/* A stopped run keeps whatever it had found — the passages were retrieved and
   paid for, so throwing them away as well would punish the stop. */
function renderStoppedCard(p) {
  const body = p.card.querySelector('.answer-body');
  const found = p.deck ? p.deck.cards.length : 0;
  body.innerHTML = '';
  const box = el('div', 'stopped-card');
  box.appendChild(el('b', null, 'Stopped'));
  box.appendChild(el('p', null, found
    ? `The answer was not written. ${found} passage${found === 1 ? '' : 's'} had already been retrieved and are listed below.`
    : 'The search was stopped before any passage was retrieved.'));
  body.appendChild(box);
  if (p.progressEl) p.progressEl.remove();
  p.trace.classList.remove('open');
  p.trace.classList.add('frozen');
}

/* Keys belong to the deck only while one is on screen and the composer does
   not have focus — otherwise ← and → would fight the text cursor. */
document.addEventListener('keydown', (e) => {
  const deck = (chatState.pending && chatState.pending.deck) || chatState.heldDeck;
  if (!deck || chatState.tab !== 'chat') return;
  const typing = document.activeElement
    && /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName);
  if (typing && e.key !== 'Escape') return;

  if (e.key === 'ArrowRight') { e.preventDefault(); deckGo(deck, 1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); deckGo(deck, -1); }
  else if (e.key === 'Enter' && deck.held) { e.preventDefault(); deck.held(); }
});

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
/* Hues come from the stylesheet (--fh-*) so they follow the theme and stay
   in one place. Fallbacks only matter if the stylesheet fails to load. */
const FOLDER_FALLBACK = {
  guidelines: '#5b7ba6', internal: '#a3794c',
  research: '#866a9e', preprints: '#a08cb5', '': '#7c8785',
};

function hueFor(name) {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(`--fh-${name || 'none'}`).trim();
  return v || FOLDER_FALLBACK[name] || FOLDER_FALLBACK[''];
}
function folderHue(folder) {
  return hueFor((folder && folder.length) ? folder[0] : '');
}
function chipHue(folder) {
  return hueFor((folder && folder.length) ? folder[folder.length - 1] : '');
}
function folderLabel(folder) {
  return (folder && folder.length) ? folder.join(' / ') : 'library root';
}

/* Source colour by node id, so citation chips and source cards in the finished
   answer carry the same hue the deck used. Populated when the deck mounts. */
const sourceHues = new Map();
let deckComposer = null;

/* The model is thinking for ~27s and says nothing while it does. A progress
   bar would be a lie — nothing here knows how long it will take — so this
   shows ACTIVITY and elapsed time instead of a fake percentage: a row of
   bars that breathe, and the passage count it is working from. */
function buildComposer(count, copy = {}) {
  const w = el('div', 'composing');
  const viz = el('div', 'composing-viz');
  for (let i = 0; i < 5; i++) {
    const b = el('i');
    b.style.animationDelay = `${i * 0.13}s`;
    viz.appendChild(b);
  }
  w.appendChild(viz);

  const txt = el('div', 'composing-text');
  txt.appendChild(el('b', null, copy.title || 'Writing the answer'));
  txt.appendChild(el('span', 'composing-sub',
    copy.sub || `reading ${count} passage${count === 1 ? '' : 's'}`));
  w.appendChild(txt);

  const clock = el('span', 'composing-clock', '0s');
  w.appendChild(clock);

  const t0 = performance.now();
  w.__timer = setInterval(() => {
    clock.textContent = Math.round((performance.now() - t0) / 1000) + 's';
  }, 1000);
  return w;
}

function stopComposer(w) {
  if (w && w.__timer) clearInterval(w.__timer);
  if (w) w.remove();
}

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

  deckComposer = buildComposer(cards.length);
  wrap.appendChild(deckComposer);

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
    const excerpt = el('blockquote', 'ev-excerpt');
    // Same highlighter the finished answer's source cards use, so a passage
    // looks the same wherever it is read.
    excerpt.innerHTML = (c.excerpt || '').trim()
      ? highlightRelevantContent(c.excerpt, c.quote || '')
      : escHtml(c.quote || '(no verbatim quote was returned)');
    body.appendChild(excerpt);
    card.appendChild(body);

    if (c.reason) {
      const why = el('div', 'ev-why');
      why.appendChild(el('b', null, 'Why this was kept'));
      why.appendChild(document.createTextNode(c.reason));
      card.appendChild(why);
    }
    const acts = el('div', 'ev-acts');
    const open = el('button', 'ev-act', 'Open document');
    open.type = 'button';
    open.addEventListener('click', e => {
      e.stopPropagation();
      openDocViewer(c.doc, c.title || null);
    });
    acts.appendChild(open);
    if (c.page && c.has_source_pdf) {
      const pg = el('button', 'ev-act', `Original page ${c.page}`);
      pg.type = 'button';
      pg.addEventListener('click', e => { e.stopPropagation(); openSourceView(c.doc, c.pin); });
      acts.appendChild(pg);
    }
    acts.appendChild(el('span', 'ev-advance', 'click anywhere to advance'));
    card.appendChild(acts);

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
  revealCardTop(p.card);
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
  // Cycles: reaching the end and being stuck there is a dead end when the
  // whole point is flipping through while you wait.
  const n = deck.cards.length;
  const next = ((deck.i + delta) % n + n) % n;
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
    const b = el('button', 'deck-btn primary');
    b.innerHTML = 'Read the answer <span class="deck-key">↵</span>';
    b.type = 'button';
    b.addEventListener('click', e => { e.stopPropagation(); deck.held(); });
    c.appendChild(b);
  } else {
    const nxt = el('button', 'deck-btn');
    nxt.innerHTML = 'Next <span class="deck-key">→</span>';
    nxt.type = 'button';
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
  if (status.chat?.phase === 'synthesis' && status.chat.cards_ready
      && !p.deck && !p.deckFetching) {
    p.deckFetching = true;
    apiGet('/api/chat/cards')
      .then(r => mountDeck(p, r.cards || []))
      .catch(() => { p.deckFetching = false; });
  }

  const meta = status.live?.meta || {};
  let added = false;
  for (const [nodeId, m] of Object.entries(meta)) {
    if (p.seen.has(nodeId)) continue;
    p.seen.add(nodeId);
    const [ico, word] = TRACE_META[m.status] || ['·', m.status];
    const line = el('div', `trace-line ${m.status} arriving`);
    const info = chatState.nodeInfo[nodeId] || {};
    // Clicking a verdict opens the passage it was about. A decision you
    // cannot inspect is an assertion, not a trace.
    if (info.doc) {
      line.classList.add('inspectable');
      line.tabIndex = 0;
      line.title = 'Open this section in the document';
      const open = () => openDocViewer(info.doc, info.title || null);
      line.addEventListener('click', open);
      line.addEventListener('keydown', ev => {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); open(); }
      });
    }

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
    // Follow every new verdict — watching the search work is the point. The
    // pointer sitting over the log means the trail is being read, so hovering
    // pauses the follow rather than dragging the reader forward.
    if (!p.logEl.matches(':hover')) p.logEl.scrollTop = p.logEl.scrollHeight;
  }
}

/* ── 5 · Answer card + source cards ────────────────────────── */

// Escape + linebreaks + [n] → clickable citation chips.
function renderRichText(text, sources, answerId) {
  let html = (typeof marked !== 'undefined')
    ? (marked.parse ? marked.parse(text || '') : marked(text || ''))
    : escHtml(text || '').replace(/\n/g, '<br>');

  // Models write grouped citations as [2][3] AND as [2, 3]; the second form
  // was left as literal text with no chip and nothing to click. One chip per
  // number, whichever way they were written.
  html = html.replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (whole, group) => {
    const nums = group.split(',').map(x => parseInt(x, 10));
    if (!sources || nums.some(n => !n || n < 1 || n > sources.length)) return whole;
    return nums.map(num => {
      const hue = sourceHues.get(num);
      const tint = hue ? ` style="--folder:${hue}"` : '';
      return `<button type="button" class="cite-chip${hue ? ' tinted' : ''}"${tint} data-cite="${num}" data-answer="${answerId}" title="Jump to source [${num}]">${num}</button>`;
    }).join('');
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
  if (deckComposer) {
    clearInterval(deckComposer.__timer);
    deckComposer.classList.add('done');
    deckComposer.querySelector('b').textContent = 'Answer ready';
    const sub = deckComposer.querySelector('.composing-sub');
    if (sub) sub.textContent = 'written from the passages behind this card';
  }
  if (p.deck && p.deck.engaged) {
    chatState.heldDeck = p.deck;
    p.deck.held = () => {
      p.deck.held = null;
      chatState.heldDeck = null;
      applyAnswer(p, data);
    };
    renderDeckControls(p.deck);
    const txt = p.progressEl.querySelector('.trace-progress-text');
    if (txt) txt.textContent = 'Answer ready';
    return;
  }
  applyAnswer(p, data);
}

function applyAnswer(p, data) {
  stopComposer(deckComposer); deckComposer = null;
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

  // (a) Nothing was retrieved. Not an answer with empty sections — a distinct
  // outcome that says what was searched and what to try, because "no result"
  // with a filter on usually means the filter, not the library.
  const noEvidence = !sources.length;
  if (noEvidence) body.appendChild(buildNoEvidenceCard(data, p, data.run_id));
  const present = noEvidence
    ? [] : chatCopy.sections.filter(([key]) => (a[key] || '').trim());
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
    // The sources are built now but revealed when the writing stops — a
    // finished list sitting under a sentence still being typed gives away
    // that the answer already exists.
    p.card.classList.add('writing');
    typeSequence(typeQueue, sources, answerId,
                 () => p.card.classList.remove('writing'));
  } else if (!noEvidence && (a.content || '').trim()) {
    const div = el('div', 'answer-fallback');
    div.innerHTML = renderRichText(a.content, sources, answerId);
    body.appendChild(div);
  } else if (!noEvidence) {
    body.appendChild(el('p', 'chat-loading', 'The model returned no answer text — see the sources below.'));
  }

  // (b) Answered, but the model says the passages did not cover everything.
  // The gap and the way to close it belong WITH the answer, not in a footnote.
  const insufficient = !noEvidence && /^\s*no\b/i.test(a.evidence_sufficient || '');
  if (insufficient) {
    p.card.classList.add('partial');
    body.appendChild(buildGapCard(a, p, data.run_id));
  }

  // What the evidence set was narrowed by, before and during retrieval.
  // Deliberately placed with the answer, not buried in the trace: the user is
  // being told their answer was composed from a reduced evidence set.
  const choiceNotice = buildChoiceNotice(data.choices);
  if (choiceNotice) body.appendChild(choiceNotice);

  const budgetNotice = buildBudgetNotice(data.budget);
  if (budgetNotice) body.appendChild(budgetNotice);

  // Trace: freeze into a collapsed audit block. It stays in the card — the
  // toggle below is how the trail is reopened once the answer is here.
  p.progressEl.remove();
  p.trace.classList.remove('open');
  p.trace.classList.add('frozen');
  if (p.stats.total === 0) p.trace.remove();
  else {
    const count = p.trace.querySelector('.trace-count');
    if (count) count.textContent =
      `· ${p.stats.total} decisions · ${p.stats.retrieved} retrieved · click any to open it`;
  }

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

  chatState.messages.push({ role: 'assistant', id: answerId, query: data.query,
                            runId: data.run_id || null, data });
  chatState.activeRunId = data.run_id || null;
  updateEvidenceLibrary();
  // There is now something to talk about, so the composer offers it — and
  // defaults to it, since the next message after reading an answer is far
  // more often about that answer than a fresh trip to the library.
  // Offered even when nothing was retrieved: "why did this find nothing?" is
  // the most likely next question there, not the least.
  updateModeBar();
  setChatMode('followup');
  refreshContextMeter();
  revealCardTop(p.card);
}

function buildContextSourceCard(s, answerId) {
  const card = el('article', 'source-card context-source');
  card.id = `src-${answerId}-${s.n}`;

  const eyebrow = el('p', 'source-eyebrow');
  eyebrow.innerHTML = `<span class="source-num">${s.n}</span> You` +
    ` <span class="source-page-pill">not from a document</span>`;
  card.appendChild(eyebrow);
  card.appendChild(el('h3', 'source-title', s.title || 'Context you provided'));

  const quote = el('blockquote', 'source-excerpt');
  quote.textContent = s.excerpt || '';
  card.appendChild(quote);

  const why = el('div', 'source-why');
  why.appendChild(el('p', 'source-why-label', 'Why this is here'));
  why.appendChild(el('p', '', s.reason
    || 'You supplied this when the retrieved passages were not enough.'));
  card.appendChild(why);

  const note = el('p', 'context-source-note');
  note.textContent = 'This is your own input, not evidence from the library. '
    + 'Claims resting on it are only as good as the detail you gave.';
  card.appendChild(note);
  return card;
}

function buildSourceCard(s, answerId) {
  // Context the clinician typed is evidence the answer cites, so it gets a
  // card and a number like everything else — but it has no document behind
  // it, so none of the "open the source" affordances apply. Showing them
  // greyed or broken would imply a provenance that does not exist.
  if (s.user_context) return buildContextSourceCard(s, answerId);

  const card = el('article', 'source-card');
  card.id = `src-${answerId}-${s.n}`;
  const hue = sourceHues.get(s.n);
  if (hue) { card.style.setProperty('--folder', hue); card.classList.add('tinted'); }

  const eyebrow = el('p', 'source-eyebrow');
  eyebrow.innerHTML = `<span class="source-num">${s.n}</span> ${escHtml(prettyDoc(s.doc))}` +
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
    // Zero-source answers are listed too: "this found nothing" is a run you
    // may well want to come back to and ask about.
    .filter(m => m.role === 'assistant' && m.data && m.id)
    .map((m, i) => ({
      answerId: m.id,
      runId: m.runId || null,
      run: m.data?.run || null,
      label: `Answer ${i + 1}`,
      query: m.query,
      status: m.data.grounding?.status || 'not_connected',
      summary: m.data.grounding?.summary || '',
      sources: m.data.grounding?.sources || [],
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

    // Pick this answer's evidence back up. The composer follows the run it is
    // pointed at, so without this an answer three questions back was readable
    // but no longer answerable — its passages were still on the server.
    const resume = el('button', 'ev-resume');
    resume.type = 'button';
    const isActive = g.answerId === chatState.activeAnswerId;
    const wasCleared = (chatState.clearedRuns || new Set()).has(g.runId);
    if (isActive && wasCleared) {
      // The button comes back to life: with the conversation cleared there IS
      // something left to do to this answer, and this is where the user is
      // already looking at the greyed-out turns.
      resume.textContent = 'Restore this conversation';
      resume.classList.add('restore');
      resume.title = 'Bring the cleared turns back into the model’s context.';
      resume.addEventListener('click', () => restoreConversation(g));
    } else if (isActive) {
      resume.textContent = 'Talking to this answer';
      resume.disabled = true;
      resume.title = 'The composer is already pointed at this answer.';
    } else {
      resume.textContent = 'Continue from this answer';
      resume.title = 'Point the composer at this answer — follow-ups and '
        + 'rewrites will use its passages.';
      resume.addEventListener('click', () => resumeAnswer(g));
    }
    head.appendChild(resume);
    group.appendChild(head);

    const list = el('div', 'ev-sources');
    if (!g.sources.length) {
      list.appendChild(el('p', 'ev-empty', 'Nothing was retrieved for this question.'));
    }
    for (const s of g.sources) {
      const btn = el('button', 'ev-source');
      btn.type = 'button';
      btn.dataset.answer = g.answerId;
      btn.dataset.n = s.n;
      // Context the clinician typed is labelled as theirs, not dressed up as
      // a document name the library does not contain.
      const origin = s.user_context ? 'You' : prettyDoc(s.doc);
      const eyebrow = `<p class="ev-source-eyebrow"><span>[${s.n}] ${escHtml(origin)}</span>` +
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

/* Point the composer back at an earlier answer's evidence. */
async function resumeAnswer(g) {
  if (!g.runId) {
    toast('That answer was not recorded on the server and cannot be resumed.',
          'warn', 6000);
    return;
  }
  try {
    const r = await apiPost('/api/chat/activate', { run_id: g.runId });
    chatState.activeRunId = g.runId;
    chatState.activeAnswerId = g.answerId;
    chatState.ctx = { ...r, available: true };
    setFollowupCleared(g.runId, !!r.restorable);
    // The Retrieval tab shows ONE run, and every later question overwrote it —
    // so resuming an answer here had to bring its decision map back too, or
    // the two tabs would be describing different questions.
    if (g.run) {
      state.currentResults = g.run;
      renderResults(g.run);
    }
    updateModeBar();
    setChatMode('followup');
    renderContextMeter();
    updateEvidenceLibrary();
    document.getElementById(`answer-${g.answerId}`)
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    toast(`Now talking to ${g.label} — ${g.sources.length} passage`
          + `${g.sources.length === 1 ? '' : 's'} in context.`, 'info', 5000);
    document.getElementById('chat-input').focus();
  } catch (e) {
    toast(`Could not resume that answer: ${e.message}`, 'err', 6000);
  }
}

async function restoreConversation(g) {
  try {
    const r = await apiPost('/api/chat/followup/restore', { run_id: g.runId });
    chatState.ctx = { ...(chatState.ctx || {}), ...r, available: true };
    setFollowupCleared(g.runId, false);
    renderContextMeter();
    updateEvidenceLibrary();
    toast(`Restored ${r.turns} turn${r.turns === 1 ? '' : 's'} — the model can `
          + `see them again.`, 'info', 5000);
  } catch (e) {
    toast(`Could not restore the conversation: ${e.message}`, 'err', 6000);
  }
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
    // Mid-run the same button stops it, which is why it is never disabled.
    if (state.running) { cancelRun(); return; }
    const text = input.value.trim();
    if (!text) return;
    if (chatState.mode === 'followup' && chatState.activeAnswerId) sendFollowup(text);
    else sendChat(text);
  });
  input.addEventListener('keydown', (e) => {
    // The composer keeps focus for the whole run, so the document-level deck
    // keys never fired: ↵ went here instead and submitted an empty composer,
    // which looked like a dead key. With nothing typed the deck owns them.
    const deck = (chatState.pending && chatState.pending.deck) || chatState.heldDeck;
    const idle = !input.value.trim();
    if (deck && idle) {
      if (e.key === 'Enter' && !e.shiftKey && deck.held) {
        e.preventDefault(); deck.held(); return;
      }
      if (e.key === 'ArrowRight') { e.preventDefault(); deckGo(deck, 1); return; }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); deckGo(deck, -1); return; }
    }
    // ⌘/Ctrl+↵ belongs to the force-a-new-search handler below.
    if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  input.addEventListener('input', () => {
    autoGrowTextarea(input);
    document.getElementById('chat-send').disabled = !input.value.trim() || state.running;
  });

  initChatConfig();

  document.querySelectorAll('.mode-tab').forEach(b => {
    b.addEventListener('click', () => { setChatMode(b.dataset.mode); input.focus(); });
  });
  document.getElementById('ctx-meter').addEventListener('click', toggleContextPopover);
  // ⌘/Ctrl+↵ is the escape hatch: force a new search without leaving the
  // composer to click a tab.
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      const text = input.value.trim();
      if (text && !state.running) sendChat(text);
    }
  });
  document.addEventListener('click', (e) => {
    const pop = document.getElementById('ctx-popover');
    if (!pop.hidden && !pop.contains(e.target)
        && !e.target.closest('#ctx-meter')) pop.hidden = true;
  });
  updateModeBar();

  document.querySelectorAll('.ev-mode-btn').forEach(b => {
    b.addEventListener('click', () => {
      chatState.evMode = b.dataset.mode;
      document.querySelectorAll('.ev-mode-btn').forEach(x =>
        x.classList.toggle('active', x === b));
      updateEvidenceLibrary();
    });
  });

  // Always the chat tab, never the last one used: this is the tab someone
  // asks a question in, and the retrieval views only make sense once there is
  // a run to look at. The stored value is kept for anything that wants to
  // know where you were, but it no longer decides where you land.
  setAppTab('chat');
}

document.addEventListener('DOMContentLoaded', initChatTab);
