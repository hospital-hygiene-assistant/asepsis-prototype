'use strict';

/* ════════════════════════════════════════════════════════════════════
   Recorded runs — replay the 100-question evaluation in the chat.

   A collapsed line above the composer, the same shape as the Filters bar.
   Open it, pick a question, and the chat plays back what the system did on
   it when the evaluation was run: the pruning walk, the passages it kept,
   and how the answering model behaved. No model is called; everything comes
   from /api/demo (evaluation/results/asepsis100 joined to the index).

   What is and is not in the records decides what is shown — nothing here
   pretends to be more than it is:
     · the real question → recorded RETRIEVAL only (no answer was saved);
     · its twin with an invented detail → recorded answer + judgment;
     · the same question over wrong passages → recorded answer + judgment.
   Every card is badged as a recording.
   ════════════════════════════════════════════════════════════════════ */

const demoState = {
  cases: null,          // [{id, question, doc_title, hit, judgment, ...}]
  available: false,
  open: false,
  query: '',
  doc: '',              // '' = every document
  playing: false,
};

async function loadDemoCases() {
  try {
    const r = await apiGet('/api/demo/cases');
    demoState.available = !!r.available;
    demoState.cases = r.cases || [];
  } catch {
    demoState.available = false;
    demoState.cases = [];
  }
  renderDemoBar();
}

// Labels arrive already shortened from the stem (demo_replay._doc_label).
function shortDocTitle(t) {
  const s = t || '';
  return s.length > 38 ? s.slice(0, 36).trimEnd() + '…' : s;
}

/* ── The bar ─────────────────────────────────────────────────── */

function renderDemoBar() {
  const host = document.getElementById('chat-demo-bar');
  if (!host) return;
  host.innerHTML = '';
  if (!demoState.available || !demoState.cases.length) { host.hidden = true; return; }
  host.hidden = false;
  host.classList.toggle('open', demoState.open);

  const bar = el('div', 'choice-summary');
  const toggle = () => { demoState.open = !demoState.open; renderDemoBar(); };
  const handle = el('button', 'choice-toggle');
  handle.type = 'button';
  handle.setAttribute('aria-expanded', String(demoState.open));
  handle.title = demoState.open ? 'Collapse' : 'Replay a question from the recorded evaluation';
  handle.addEventListener('click', toggle);
  handle.appendChild(el('span', 'choice-caret', '▸'));
  handle.appendChild(el('span', 'choice-eyebrow', 'Recorded runs'));
  handle.appendChild(el('span', 'choice-count', String(demoState.cases.length)));
  bar.appendChild(handle);

  const hint = el('button', 'choice-empty',
    'instant replays of the evaluation — no model call');
  hint.type = 'button';
  hint.addEventListener('click', toggle);
  bar.appendChild(hint);

  const lucky = el('button', 'demo-lucky', '↻ Random');
  lucky.type = 'button';
  lucky.title = 'Replay a random recorded question';
  lucky.addEventListener('click', () => {
    const pool = filteredDemoCases();
    const pick = (pool.length ? pool : demoState.cases)[Math.floor(Math.random() * (pool.length || demoState.cases.length))];
    if (pick) playDemoCase(pick.id);
  });
  bar.appendChild(lucky);
  host.appendChild(bar);

  if (demoState.open) host.appendChild(buildDemoPanel());
}

function filteredDemoCases() {
  const q = demoState.query.trim().toLowerCase();
  return demoState.cases.filter(c =>
    (!demoState.doc || c.doc_title === demoState.doc) &&
    (!q || c.question.toLowerCase().includes(q) || c.id.toLowerCase() === q));
}

function buildDemoPanel() {
  const panel = el('div', 'choice-panel demo-panel');

  const head = el('div', 'demo-head');
  const search = document.createElement('input');
  search.type = 'search';
  search.className = 'demo-search';
  search.placeholder = 'Filter questions…';
  search.value = demoState.query;
  search.spellcheck = false;
  head.appendChild(search);

  const docs = [...new Set(demoState.cases.map(c => c.doc_title))];
  const seg = el('div', 'demo-docs');
  const addDoc = (value, label) => {
    const n = value ? demoState.cases.filter(c => c.doc_title === value).length : demoState.cases.length;
    const b = el('button', 'choice-opt' + (demoState.doc === value ? ' on' : ''), `${label} · ${n}`);
    b.type = 'button';
    if (value) b.title = value;
    b.addEventListener('click', () => { demoState.doc = value; renderDemoBar(); });
    seg.appendChild(b);
  };
  addDoc('', 'All');
  docs.forEach(d => addDoc(d, shortDocTitle(d)));
  head.appendChild(seg);
  panel.appendChild(head);

  const list = el('div', 'demo-list');
  const fill = () => {
    list.innerHTML = '';
    const rows = filteredDemoCases();
    if (!rows.length) list.appendChild(el('p', 'demo-none', 'No recorded question matches.'));
    for (const c of rows) {
      const row = el('button', 'demo-row');
      row.type = 'button';
      row.appendChild(el('span', 'demo-id', c.id));
      row.appendChild(el('span', 'demo-q', c.question));
      const tags = el('span', 'demo-tags');
      tags.appendChild(el('span', `demo-tag ${c.hit ? 'ok' : 'miss'}`,
        c.hit ? '★ found' : '✕ missed'));
      tags.appendChild(el('span', 'demo-tag', `${c.n_retrieved} passage${c.n_retrieved === 1 ? '' : 's'}`));
      tags.appendChild(el('span', 'demo-tag faint', `${Math.round(c.secs / 60)} min live`));
      row.appendChild(tags);
      row.title = `${c.doc_title}\nClick to replay`;
      row.addEventListener('click', () => { demoState.open = false; renderDemoBar(); playDemoCase(c.id); });
      list.appendChild(row);
    }
  };
  search.addEventListener('input', () => { demoState.query = search.value; fill(); });
  fill();
  panel.appendChild(list);
  requestAnimationFrame(() => search.focus());
  return panel;
}

/* ── Playback ────────────────────────────────────────────────── */

const demoSleep = (ms) => new Promise(r => setTimeout(r, ms));
const demoReduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

async function playDemoCase(id) {
  if (state.running) { toast('A live run is in progress — wait for it to finish.', 'warn'); return; }
  if (demoState.playing) return;
  let c;
  try { c = await apiGet(`/api/demo/case/${encodeURIComponent(id)}`); }
  catch (e) { toast(`Could not load ${id}: ${e.message}`, 'err'); return; }

  demoState.playing = true;
  try {
    chatState.messages.push({ role: 'user', content: c.question, demo: true });
    renderChatEmptyState();
    document.getElementById('view-chat').classList.add('has-conversation');

    const messagesEl = document.getElementById('chat-messages');
    const userRow = el('div', 'chat-row user');
    const bubble = el('div', 'chat-user-bubble', c.question);
    userRow.appendChild(bubble);
    messagesEl.appendChild(userRow);

    const row = el('div', 'chat-row assistant');
    const card = el('article', 'answer-card demo-card');
    row.appendChild(card);
    messagesEl.appendChild(row);

    const head = el('div', 'answer-head');
    head.appendChild(el('p', 'answer-label', `Recorded run · ${c.id}`));
    const rec = el('span', 'demo-rec', '● REPLAY');
    rec.title = 'Played back from the committed evaluation records. No model was called.';
    head.appendChild(rec);
    card.appendChild(head);
    card.appendChild(el('p', 'demo-sub',
      `${c.doc_title} · took ${fmtSecs(c.a.secs)} live on this machine`));
    scrollChatToBottom();

    await playTrace(card, c);
    renderRetrieved(card, c);
    revealCardTop(card);
    await demoSleep(demoReduced() ? 0 : 250);
    renderStress(card, c);
  } finally {
    demoState.playing = false;
  }
}

function fmtSecs(s) {
  if (!s && s !== 0) return '—';
  return s < 90 ? `${Math.round(s)} s` : `${Math.round(s / 60)} min`;
}

async function playTrace(card, c) {
  const trace = el('div', 'trace-block open');
  const toggle = el('button', 'trace-toggle');
  toggle.type = 'button';
  toggle.innerHTML = `<span class="trace-caret">▶</span> Retrieval reasoning <span class="trace-count"></span>`;
  toggle.addEventListener('click', () => trace.classList.toggle('open'));
  const body = el('div', 'trace-body');
  const log = el('div', 'trace-log');
  body.appendChild(log);
  trace.appendChild(toggle);
  trace.appendChild(body);
  card.appendChild(trace);

  const lines = [];
  for (const s of c.a.steps) lines.push({ cls: 'kept', ico: '○', text: s });
  for (const p of c.a.retrieved) {
    lines.push({
      cls: 'retrieved', ico: '✓', text: p.title, doc: p.doc, docTitle: p.doc_title,
      gold: p.is_gold,
    });
  }
  const step = demoReduced() ? 0 : Math.max(120, Math.min(420, 2600 / Math.max(1, lines.length)));
  for (const ln of lines) {
    const line = el('div', `trace-line ${ln.cls} arriving`);
    line.innerHTML = `<span class="trace-ico">${ln.ico}</span> ` +
      `<span class="trace-text">${escHtml(ln.text)}</span>` +
      (ln.docTitle ? ` <span class="trace-doc">· ${escHtml(shortDocTitle(ln.docTitle))}</span>` : '') +
      (ln.gold ? ` <span class="demo-gold">★ gold</span>` : '');
    if (ln.doc) {
      line.classList.add('inspectable');
      line.title = 'Open this section in the document';
      line.addEventListener('click', () => openDocViewer(ln.doc, ln.text));
    }
    log.appendChild(line);
    scrollChatToBottom(false);
    await demoSleep(step);
  }
  toggle.querySelector('.trace-count').textContent =
    `· ${c.a.retrieved.length} retrieved`;
  trace.classList.remove('open');
}

function renderRetrieved(card, c) {
  const verdict = el('div', `demo-verdict ${c.a.hit ? 'ok' : 'miss'}`);
  verdict.innerHTML = c.a.hit
    ? `<b>★ The passage that holds the answer was retrieved</b>` +
      (c.a.how === 'rehydrated' ? ' — rescued by lexical rehydration after pruning dropped it' : '') +
      (c.a.reason ? `<span class="demo-why">Model’s reason: ${escHtml(c.a.reason)}</span>` : '')
    : `<b>✕ The passage that holds the answer was not retrieved</b>` +
      `<span class="demo-why">Expected: ${escHtml(c.gold.title)}</span>`;
  card.appendChild(verdict);

  const sec = el('section', 'answer-section');
  sec.appendChild(el('h3', '', `Retrieved passages · ${c.a.retrieved.length}`));
  // The one that answers first; the rest are one click away, not a wall.
  const ordered = [...c.a.retrieved].sort((x, y) => (y.is_gold ? 1 : 0) - (x.is_gold ? 1 : 0));
  const SHOWN = 3;
  ordered.slice(0, SHOWN).forEach(p => sec.appendChild(demoPassageCard(p)));
  if (ordered.length > SHOWN) {
    const rest = ordered.slice(SHOWN);
    const more = el('button', 'demo-link demo-more', `Show ${rest.length} more retrieved passage${rest.length === 1 ? '' : 's'}`);
    more.type = 'button';
    more.addEventListener('click', () => {
      rest.forEach(p => sec.insertBefore(demoPassageCard(p), more));
      more.remove();
    });
    sec.appendChild(more);
  }
  card.appendChild(sec);
  card.appendChild(el('p', 'demo-note',
    'The answer text for this exact question was not saved during the evaluation — ' +
    'only its retrieval. The answering model’s recorded behaviour on this question is below.'));
}

function demoPassageCard(p, n) {
  const box = el('article', 'demo-passage' + (p.is_gold ? ' is-gold' : ''));
  const eyebrow = el('p', 'demo-passage-eyebrow');
  eyebrow.innerHTML = (n ? `<span class="source-num">${n}</span> ` : '') +
    escHtml(shortDocTitle(p.doc_title)) +
    (p.page ? ` <span class="source-page-pill">p. ${escHtml(String(p.page))}</span>` : '') +
    (p.is_gold ? ` <span class="demo-gold">★ holds the answer</span>` : '');
  box.appendChild(eyebrow);
  box.appendChild(el('h4', 'demo-passage-title', p.title));
  const txt = el('p', 'demo-passage-text clamped', p.excerpt || p.summary || '');
  box.appendChild(txt);
  const actions = el('div', 'demo-passage-actions');
  if ((p.excerpt || '').length > 240) {
    const more = el('button', 'demo-link', 'Show more');
    more.type = 'button';
    more.addEventListener('click', () => {
      const open = txt.classList.toggle('clamped');
      more.textContent = open ? 'Show more' : 'Show less';
    });
    actions.appendChild(more);
  }
  if (p.doc) {
    const open = el('button', 'demo-link', 'Open in document ↗');
    open.type = 'button';
    open.addEventListener('click', () => openDocViewer(p.doc, p.title));
    actions.appendChild(open);
  }
  box.appendChild(actions);
  return box;
}

/* The two recorded answers, as tabs: the model is the same, the input is
   what was changed. */
function renderStress(card, c) {
  if (!c.b && !c.c) return;
  const wrap = el('section', 'demo-stress');
  wrap.appendChild(el('h3', 'demo-stress-title', 'How the answering model behaved — recorded'));

  const tabs = el('div', 'demo-tabs');
  const pane = el('div', 'demo-pane');
  const variants = [];
  if (c.b) variants.push(['b', 'Invented detail added', c.b]);
  if (c.c) variants.push(['c', 'Given the wrong passages', c.c]);

  const show = (key) => {
    tabs.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.k === key));
    pane.innerHTML = '';
    const [, , v] = variants.find(x => x[0] === key);
    renderVariant(pane, key, v, c);
  };
  for (const [key, label] of variants) {
    const b = el('button', 'choice-opt', label);
    b.type = 'button';
    b.dataset.k = key;
    b.addEventListener('click', () => show(key));
    tabs.appendChild(b);
  }
  wrap.appendChild(tabs);
  wrap.appendChild(pane);
  card.appendChild(wrap);
  show(variants[0][0]);
}

function renderVariant(pane, key, v, c) {
  if (key === 'b') {
    const q = escHtml(v.question).replace(escHtml(v.edit_term || '\u0000'),
      `<mark class="demo-edit">${escHtml(v.edit_term || '')}</mark>`);
    pane.appendChild(htmlEl('p', 'demo-explain',
      `The same question with a made-up detail inserted. The right passage is ` +
      `${v.gold_retrieved ? 'still retrieved' : '<b>not</b> retrieved this time'}, ` +
      `but it cannot say anything about the invented part — a good answer says so.`));
    pane.appendChild(htmlEl('blockquote', 'demo-variant-q', q));
  } else {
    pane.appendChild(htmlEl('p', 'demo-explain',
      `The original question, answered from ${v.passages.length} passages retrieved for ` +
      `<b>other</b> questions. None of them hold the answer — a good answer refuses rather than guesses.`));
  }

  const passed = key === 'c' ? v.passed : v.judgment === 'insufficient';
  const badge = el('div', `demo-judgment ${passed ? 'ok' : 'warn'}`);
  badge.innerHTML = `<b>Evidence sufficient: ${v.judgment === 'insufficient' ? 'no' : escHtml(v.judgment || '?')}</b>` +
    ` <span>${passed ? '— correctly flagged' : '— should have been flagged'}</span>`;
  pane.appendChild(badge);

  const a = v.answer || {};
  const sections = [['short_answer', 'Short answer'], ['recommended_action', 'Recommended action'],
                    ['rationale', 'Rationale'], ['limitations', 'Limitations']];
  const queue = [];
  for (const [k, label] of sections) {
    if (!(a[k] || '').trim()) continue;
    const sec = el('section', 'answer-section');
    sec.appendChild(el('h3', '', label));
    const p = el('p', 'answer-text');
    sec.appendChild(p);
    pane.appendChild(sec);
    queue.push([p, a[k]]);
  }
  if (v.still_needed?.length) {
    const sec = el('section', 'answer-section demo-needed');
    sec.appendChild(el('h3', '', 'Still needed'));
    const ul = el('ul');
    v.still_needed.forEach(s => ul.appendChild(el('li', null, s)));
    sec.appendChild(ul);
    pane.appendChild(sec);
  }
  typeDemo(queue, v.passages);

  if (v.passages?.length) {
    const det = document.createElement('details');
    det.className = 'demo-sources';
    det.appendChild(htmlEl('summary', null,
      `${v.passages.length} passage${v.passages.length === 1 ? '' : 's'} it was given`));
    v.passages.forEach((p, i) => det.appendChild(demoPassageCard(
      { ...p, is_gold: p.node_id === c.gold.node_id }, i + 1)));
    pane.appendChild(det);
  }
}

function htmlEl(tag, cls, html) {
  const e = el(tag, cls);
  e.innerHTML = html;
  return e;
}

function demoRich(text, passages) {
  return escHtml(text || '')
    .replace(/\s*\[\]/g, '')
    .replace(/\[(\d+)\]/g, (m, n) => {
      const p = passages?.[Number(n) - 1];
      return `<span class="cite-chip" title="${p ? escHtml(p.title) : ''}">${n}</span>`;
    });
}

function typeDemo(queue, passages) {
  const cps = 1400;
  let k = 0;
  (function next() {
    if (k >= queue.length) return;
    const [node, text] = queue[k++];
    if (demoReduced()) { node.innerHTML = demoRich(text, passages); next(); return; }
    let i = 0, last = performance.now();
    requestAnimationFrame(function frame(now) {
      const dt = Math.min(80, now - last); last = now;
      i = Math.min(text.length, i + Math.max(1, Math.round(cps * dt / 1000)));
      node.innerHTML = demoRich(text.slice(0, i), passages);
      if (i < text.length) requestAnimationFrame(frame); else next();
    });
  })();
}

document.addEventListener('DOMContentLoaded', loadDemoCases);
