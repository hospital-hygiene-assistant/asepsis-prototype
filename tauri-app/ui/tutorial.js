'use strict';

/* ═══════════════════════════════════════════════════════════════════
   Guided tour — a card deck that shows what the app does

   Deliberately a REPLICA, not a driver. Every card is static markup over
   invented data: no query runs, no model is called, nothing in the real
   library is touched. A tour that drove the live app would take minutes,
   need a warm model, and break the moment a corpus differed — and a first
   run is exactly when none of that is ready.

   It reuses the evidence deck's shuffle mechanics on purpose: the deck is
   how this app already asks you to page through things, so learning it here
   is learning the app.
   ═══════════════════════════════════════════════════════════════════ */

const TOUR_VERSION = 3;      // bump to re-show the tour after a big change

/* Each card: an eyebrow, a title, a line of prose, and a mock. The mock is
   plain markup — the same class names the real surfaces use where that keeps
   it honest, scoped under .tour so it can never restyle the app itself. */
const TOUR_CARDS = [
  {
    eyebrow: 'What this is',
    title: 'Answers you can check',
    body: 'Ask a clinical question in plain language. The answer is built only '
        + 'from the documents in your library, and every claim points at the '
        + 'passage it came from — so you can verify it rather than trust it.',
    mock: () => mockAnswer(),
  },
  {
    eyebrow: 'Your library',
    title: 'Everything stays on this machine',
    body: 'Your documents are indexed locally and the model runs locally. '
        + 'Nothing is uploaded, and the app works with the network off. '
        + 'The map shows every section the app can see.',
    mock: () => mockLibrary(),
  },
  {
    eyebrow: 'Asking',
    title: 'Type a question, watch it work',
    body: 'While the answer is written, the passages it found are already '
        + 'here — flip through them with ← and →, or click. Press ↵ when you '
        + 'are ready for the answer.',
    mock: () => mockDeck(),
  },
  {
    eyebrow: 'Checking',
    title: 'Every [n] is a link',
    body: 'Click a citation and it jumps to that source: the passage, the '
        + 'deciding quote highlighted, the page it sits on, and why the '
        + 'retrieval model chose it.',
    mock: () => mockSource(),
  },
  {
    eyebrow: 'Following up',
    title: 'Two buttons, no guessing',
    body: '“Ask about this answer” keeps the passages you already have and '
        + 'talks about them. “New search” goes back to the library. You '
        + 'choose which — the app never decides for you.',
    mock: () => mockComposer(),
  },
  {
    eyebrow: 'When it falls short',
    title: 'It tells you what it is missing',
    body: 'If the passages did not cover your question, the answer says so '
        + 'and names the gap. Add the missing detail and it rewrites — and '
        + 'says what changed.',
    mock: () => mockGap(),
  },
  {
    eyebrow: 'Behind the answer',
    title: 'The Retrieval tab shows the working',
    body: 'Every section the model kept, pruned, rejected or never reached, '
        + 'with its reasoning. This is the audit trail for an answer you are '
        + 'about to act on. Click any line to open the passage it judged.',
    mock: () => mockRetrieval(),
  },
  {
    eyebrow: 'Two ways to look',
    title: 'Graph, or boxes',
    body: 'The same run, two shapes. Graph follows the document’s own '
        + 'structure — how a decision about one section ruled out the branch '
        + 'beneath it. Boxes gives every section area, so you see how much of '
        + 'the library a question actually touched.',
    mock: () => mockViews(),
  },
  {
    eyebrow: 'Steering it',
    title: 'Narrow the search before you ask',
    body: 'The Filters bar picks which parts of the library to search. It '
        + 'stays collapsed with your choices visible, and the meter shows how '
        + 'much of the library is left — so a thin answer is never a mystery.',
    mock: () => mockFilters(),
  },
  {
    eyebrow: 'Starting over',
    title: 'Clear a conversation, or bring it back',
    body: 'A long thread can drift. Clearing it gives the model a clean slate '
        + 'while keeping the answer and its passages. The turns stay on screen, '
        + 'greyed — and one click puts them back.',
    mock: () => mockCleared(),
  },
  {
    eyebrow: 'Nothing hidden',
    title: 'You can see — and change — how it works',
    body: 'Model & prompts shows the exact wording sent to the model for every '
        + 'step. Settings lets you change the model, how much it reads at once, '
        + 'and where your documents come from.',
    mock: () => mockSettings(),
  },
  {
    eyebrow: 'Growing it',
    title: 'Add your own documents',
    body: 'Point the app at a folder of PDFs and it indexes them. Sub-folder '
        + 'names become tags you can filter by — click one to highlight just '
        + 'those documents.',
    mock: () => mockTags(),
  },
  {
    eyebrow: 'Awkward documents',
    title: 'Scanned pages, corrected by hand',
    body: 'Scanned PDFs are read with OCR, and OCR mislabels things — a '
        + 'heading read as body text, a caption on the wrong figure. A '
        + 'passage the app never sees correctly is one it cannot cite. Our '
        + 'annotation tool lets you fix those regions before ingesting.',
    mock: () => mockAnnotator(),
  },
  {
    eyebrow: 'Whenever you need it',
    title: 'This tour is not going anywhere',
    body: 'Everything here is in Settings — this walk-through, and the full '
        + 'written guide covering what to do when an answer is wrong, thin, or '
        + 'missing. You will not have to remember any of it.',
    mock: () => mockHelp(),
  },
];

/* ── Mocks. Invented content, clearly synthetic, no live data ── */

function tEl(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function mockAnswer() {
  const w = tEl('div', 'tour-mock tour-answer');
  w.appendChild(tEl('div', 'tour-eyebrow-sm', 'ANSWER'));
  const s = tEl('div', 'tour-sec');
  s.appendChild(tEl('h6', null, 'SHORT ANSWER'));
  const p = tEl('p');
  p.innerHTML = 'Start the first-line oral agent at the standard adult dose '
    + '<span class="tour-cite">1</span>, and review at two weeks '
    + '<span class="tour-cite">2</span>.';
  s.appendChild(p);
  w.appendChild(s);
  const badge = tEl('span', 'tour-badge ok', '✓ Grounded in document passages');
  w.appendChild(badge);
  return w;
}

function mockLibrary() {
  const w = tEl('div', 'tour-mock tour-lib');
  const grid = tEl('div', 'tour-grid');
  const blocks = [
    ['guidelines', 4], ['internal', 3], ['research', 3], ['protocols', 2],
  ];
  for (const [name, n] of blocks) {
    const b = tEl('div', 'tour-block');
    b.appendChild(tEl('span', 'tour-block-name', name));
    const inner = tEl('div', 'tour-block-inner');
    for (let i = 0; i < n; i++) inner.appendChild(tEl('i'));
    b.appendChild(inner);
    grid.appendChild(b);
  }
  w.appendChild(grid);
  w.appendChild(tEl('p', 'tour-caption', 'Every box is a section the app can read.'));
  return w;
}

function mockDeck() {
  const w = tEl('div', 'tour-mock tour-deckmock');
  const stack = tEl('div', 'tour-stack');
  for (let i = 2; i >= 0; i--) {
    const c = tEl('div', 'tour-evcard');
    c.style.transform = `translateY(${i * 7}px) rotate(${i * 0.9}deg) scale(${1 - i * 0.03})`;
    c.style.opacity = String(1 - i * 0.3);
    c.style.zIndex = String(10 - i);
    if (i === 0) {
      c.appendChild(tEl('span', 'tour-chip', 'guidelines'));
      c.appendChild(tEl('h6', null, 'First-line therapy'));
      c.appendChild(tEl('p', null, '“…the standard adult dose is recommended as '
        + 'initial therapy…”'));
    }
    stack.appendChild(c);
  }
  w.appendChild(stack);
  const keys = tEl('div', 'tour-keys');
  keys.innerHTML = '<kbd>←</kbd><kbd>→</kbd> flip · <kbd>↵</kbd> read the answer';
  w.appendChild(keys);
  return w;
}

function mockSource() {
  const w = tEl('div', 'tour-mock tour-source');
  const head = tEl('p', 'tour-src-eyebrow');
  head.innerHTML = '<b>1</b> GUIDELINES · HYPERTENSION <span class="tour-pill">p. 12</span>';
  w.appendChild(head);
  w.appendChild(tEl('h6', null, 'Management › First-line therapy'));
  const q = tEl('blockquote');
  q.innerHTML = 'Initial therapy should begin with '
    + '<mark>the standard adult dose</mark>, reviewed at two weeks.';
  w.appendChild(q);
  w.appendChild(tEl('p', 'tour-why', 'WHY SELECTED'));
  w.appendChild(tEl('p', 'tour-whytext', 'States the initial dose the question asked for.'));
  return w;
}

function mockComposer() {
  const w = tEl('div', 'tour-mock tour-composer');
  const tabs = tEl('div', 'tour-tabs');
  tabs.appendChild(tEl('span', 'tour-tab on', 'Ask about this answer'));
  tabs.appendChild(tEl('span', 'tour-tab', 'New search'));
  w.appendChild(tabs);
  const meter = tEl('div', 'tour-meter');
  const bar = tEl('div', 'tour-meter-bar');
  for (const [cls, pct] of [['a', 6], ['b', 46], ['c', 8], ['d', 12]]) {
    const seg = tEl('i', `tour-seg-${cls}`);
    seg.style.width = `${pct}%`;
    bar.appendChild(seg);
  }
  meter.appendChild(bar);
  meter.appendChild(tEl('span', 'tour-meter-label', '18k / 49k'));
  w.appendChild(meter);
  w.appendChild(tEl('p', 'tour-caption',
    'The meter shows how much of the model’s memory this conversation is using.'));
  return w;
}

function mockGap() {
  const w = tEl('div', 'tour-mock tour-gap');
  w.appendChild(tEl('b', null, 'This answer is incomplete'));
  w.appendChild(tEl('p', null, 'To answer fully it would need:'));
  const ul = tEl('ul');
  ul.appendChild(tEl('li', null, 'The patient’s renal function'));
  ul.appendChild(tEl('li', null, 'Whether there is a documented allergy'));
  w.appendChild(ul);
  const row = tEl('div', 'tour-gap-row');
  row.appendChild(tEl('span', 'tour-input', 'eGFR 38, no known allergies'));
  row.appendChild(tEl('span', 'tour-btn', 'Rewrite the answer'));
  w.appendChild(row);
  return w;
}

function mockRetrieval() {
  const w = tEl('div', 'tour-mock tour-trace');
  const rows = [
    ['ok', '✓ Retrieved', 'Management › First-line therapy'],
    ['no', '✗ Rejected', 'Appendix › Abbreviations'],
    ['pr', '⊘ Pruned', 'Paediatrics (whole branch)'],
    ['ok', '✓ Retrieved', 'Monitoring › Review interval'],
  ];
  for (const [k, verdict, what] of rows) {
    const r = tEl('div', `tour-trace-row tour-${k}`);
    r.appendChild(tEl('span', 'tour-verdict', verdict));
    r.appendChild(tEl('span', 'tour-what', what));
    w.appendChild(r);
  }
  w.appendChild(tEl('p', 'tour-caption', 'Click any line to open the passage it judged.'));
  return w;
}

function mockTags() {
  const w = tEl('div', 'tour-mock tour-tagmock');
  const row = tEl('div', 'tour-tagrow');
  for (const [t, on] of [['guidelines', true], ['internal', false], ['research', false]]) {
    row.appendChild(tEl('span', 'tour-tagpill' + (on ? ' on' : ''), t));
  }
  w.appendChild(row);
  const grid = tEl('div', 'tour-grid');
  for (let i = 0; i < 8; i++) {
    const b = tEl('div', 'tour-block' + (i < 3 ? '' : ' dim'));
    // Filled, not empty: the point of this card is what dimming LOOKS like,
    // and an outline with nothing in it dims into invisibility instead.
    const inner = tEl('div', 'tour-block-inner');
    for (let k = 0; k < 3; k++) inner.appendChild(tEl('i'));
    b.appendChild(inner);
    grid.appendChild(b);
  }
  w.appendChild(grid);
  w.appendChild(tEl('p', 'tour-caption', 'Folder names become tags. Click one to highlight it.'));
  return w;
}

function mockViews() {
  const w = tEl('div', 'tour-mock tour-views');
  const seg = tEl('div', 'tour-tabs');
  seg.appendChild(tEl('span', 'tour-tab on', '⊟ Graph'));
  seg.appendChild(tEl('span', 'tour-tab', '▦ Boxes'));
  w.appendChild(seg);

  const split = tEl('div', 'tour-split');
  // Left: a tree, one branch cut. Right: the same run as areas.
  const tree = tEl('div', 'tour-tree');
  for (const [depth, cls, label] of [
    [0, 'ok', 'Hypertension guideline'],
    [1, 'ok', 'Management'],
    [2, 'ok', 'First-line therapy'],
    [1, 'pr', 'Paediatrics — pruned'],
    [2, 'pr', 'Dosing'],
  ]) {
    const r = tEl('div', `tour-tree-row tour-${cls}`);
    r.style.paddingLeft = `${depth * 13}px`;
    r.appendChild(tEl('i'));
    r.appendChild(tEl('span', null, label));
    tree.appendChild(r);
  }
  split.appendChild(tree);

  const boxes = tEl('div', 'tour-boxes');
  for (const [flex, cls] of [[3, 'ok'], [2, ''], [1, 'no'], [2, ''], [1, 'pr'], [1, '']]) {
    const b = tEl('i', cls ? `tour-b-${cls}` : null);
    b.style.flex = String(flex);
    boxes.appendChild(b);
  }
  split.appendChild(boxes);
  w.appendChild(split);
  w.appendChild(tEl('p', 'tour-caption',
    'Graph: how one decision ruled out a branch. Boxes: how much was touched.'));
  return w;
}

function mockFilters() {
  const w = tEl('div', 'tour-mock tour-filters');
  const bar = tEl('div', 'tour-filterbar');
  bar.appendChild(tEl('span', 'tour-filter-eyebrow', '▸ FILTERS'));
  bar.appendChild(tEl('span', 'tour-filter-count', '2'));
  for (const t of ['Infection & antimicrobials', 'Intensive care']) {
    const chip = tEl('span', 'tour-filter-chip');
    chip.appendChild(tEl('b', null, t));
    chip.appendChild(tEl('i', null, '✕'));
    bar.appendChild(chip);
  }
  const scope = tEl('span', 'tour-filter-scope');
  const meter = tEl('span', 'tour-filter-meter');
  const fill = tEl('i');
  fill.style.width = '20%';
  meter.appendChild(fill);
  scope.appendChild(meter);
  scope.appendChild(tEl('span', null, '26 of 131'));
  bar.appendChild(scope);
  w.appendChild(bar);
  w.appendChild(tEl('p', 'tour-caption',
    'Click a ✕ to drop one filter. The bar opens for the full set of questions.'));
  return w;
}

function mockCleared() {
  const w = tEl('div', 'tour-mock tour-cleared');
  const a = tEl('div', 'tour-bubble user faded', 'What about renal impairment?');
  const b = tEl('div', 'tour-bubble faded', 'The passages do not cover dose adjustment [1].');
  w.appendChild(a);
  w.appendChild(b);
  const btn = tEl('div', 'tour-restore', 'Restore this conversation');
  w.appendChild(btn);
  w.appendChild(tEl('p', 'tour-caption',
    'Greyed means the model has stopped seeing them — not that you have lost them.'));
  return w;
}

function mockSettings() {
  const w = tEl('div', 'tour-mock tour-settings');
  for (const [label, value] of [
    ['Answering model', 'runs on this machine'],
    ['Agent context', 'how much it reads per question'],
    ['PDF corpus', 'which folder your documents come from'],
    ['Flag incomplete answers', 'on'],
  ]) {
    const r = tEl('div', 'tour-set-row');
    r.appendChild(tEl('span', 'tour-set-label', label));
    r.appendChild(tEl('span', 'tour-set-value', value));
    w.appendChild(r);
  }
  const chip = tEl('p', 'tour-caption');
  chip.innerHTML = '<b>Model &amp; prompts</b> shows the exact text sent to the '
    + 'model at every step — retrieval, answering, follow-up and rewrite.';
  w.appendChild(chip);
  return w;
}

function mockHelp() {
  const w = tEl('div', 'tour-mock tour-help');
  for (const [icon, title, sub] of [
    ['?', 'Take the guided tour', 'This walk-through, any time.'],
    ['📖', 'Open the user guide', 'The full written guide, in the app.'],
  ]) {
    const r = tEl('div', 'tour-help-row');
    r.appendChild(tEl('span', 'tour-help-icon', icon));
    const t = tEl('span');
    t.appendChild(tEl('b', null, title));
    t.appendChild(tEl('i', null, sub));
    r.appendChild(t);
    w.appendChild(r);
  }
  w.appendChild(tEl('p', 'tour-caption', 'Both live under ⚙ Settings.'));
  return w;
}

/* The one card showing a real screenshot rather than a drawing: this is a
   separate application, so a stylised mock-up of it would teach the wrong
   shape. Served from /guide-assets, the same file the user guide uses. */
function mockAnnotator() {
  const w = tEl('div', 'tour-mock tour-shot');
  const img = document.createElement('img');
  img.src = '/guide-assets/annotation-tool.webp';
  img.alt = 'The Asepsis Annotation & Correction Tool: a document page with each '
    + 'detected region boxed and labelled, a list of regions, an edit history, '
    + 'and a summary of how many were corrected.';
  img.loading = 'lazy';
  // If the asset is missing the card must still say something useful rather
  // than showing a broken-image glyph.
  img.addEventListener('error', () => {
    img.remove();
    w.prepend(tEl('p', 'tour-caption',
      'Screenshot unavailable — see the user guide for the tool.'));
  });
  w.appendChild(img);
  const link = tEl('p', 'tour-caption');
  link.innerHTML = 'Asepsis Annotation &amp; Correction Tool · '
    + '<span class="tour-url">github.com/hospital-hygiene-assistant/'
    + 'asepsis-annotation-tool</span><br>We plan to build this in; for now it '
    + 'runs alongside, and its corrected output can be ingested here.';
  w.appendChild(link);
  return w;
}

/* ── The deck itself ────────────────────────────────────────────── */

let _tour = null;

function openTour(startAt = 0) {
  if (_tour) return;

  const overlay = tEl('div', 'tour-overlay');
  const win = tEl('div', 'tour-window');
  win.setAttribute('role', 'dialog');
  win.setAttribute('aria-modal', 'true');
  win.setAttribute('aria-label', 'Guided tour');

  const head = tEl('header', 'tour-head');
  const hl = tEl('div');
  hl.appendChild(tEl('p', 'tour-kicker', 'ASEPSIS · Guided tour'));
  hl.appendChild(tEl('p', 'tour-sub', 'A walk-through on example data. Nothing here runs.'));
  head.appendChild(hl);
  const close = tEl('button', 'tour-close', '✕');
  close.type = 'button';
  close.title = 'Close the tour (Esc)';
  close.addEventListener('click', closeTour);
  head.appendChild(close);
  win.appendChild(head);

  const stage = tEl('div', 'tour-stage');
  win.appendChild(stage);

  const controls = tEl('div', 'tour-controls');
  win.appendChild(controls);

  overlay.appendChild(win);
  // Deliberately NOT closed by a backdrop click, unlike the app's other
  // overlays: the tour is shown once, and where to find it again is on its
  // last card — so a stray click outside it would cost someone the one
  // explanation they get. The ✕ (and Esc) are the way out.
  document.body.appendChild(overlay);

  _tour = { overlay, stage, controls, i: Math.min(startAt, TOUR_CARDS.length - 1),
            seen: new Set([startAt]) };
  document.addEventListener('keydown', tourKeys);
  renderTour();
  close.focus();
}

function closeTour() {
  if (!_tour) return;
  document.removeEventListener('keydown', tourKeys);
  _tour.overlay.remove();
  _tour = null;
  prefs.set('tourSeen', TOUR_VERSION);
}

function tourKeys(e) {
  if (!_tour) return;
  if (e.key === 'Escape') { e.preventDefault(); closeTour(); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); tourGo(1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); tourGo(-1); }
  else if (e.key === 'Enter') {
    e.preventDefault();
    if (_tour.i === TOUR_CARDS.length - 1) closeTour(); else tourGo(1);
  }
}

function tourGo(delta) {
  if (!_tour) return;
  const n = TOUR_CARDS.length;
  // Clamped, not cyclic: a tour has a beginning and an end, and wrapping from
  // the last card back to the first reads as a bug rather than a loop.
  _tour.i = Math.max(0, Math.min(n - 1, _tour.i + delta));
  _tour.seen.add(_tour.i);
  renderTour();
}

function renderTour() {
  const { stage, controls, i } = _tour;
  const card = TOUR_CARDS[i];

  stage.innerHTML = '';
  const art = tEl('article', 'tour-card');
  art.appendChild(tEl('p', 'tour-card-eyebrow', card.eyebrow));
  art.appendChild(tEl('h3', 'tour-card-title', card.title));
  art.appendChild(tEl('p', 'tour-card-body', card.body));
  art.appendChild(card.mock());
  stage.appendChild(art);
  // Re-trigger the entry animation on every card, so moving between them
  // reads as the same shuffle the evidence deck uses.
  art.classList.add('in');

  controls.innerHTML = '';
  const prev = tEl('button', 'tour-btn-ghost', '← Back');
  prev.type = 'button';
  prev.disabled = i === 0;
  prev.addEventListener('click', () => tourGo(-1));
  controls.appendChild(prev);

  const dots = tEl('div', 'tour-dots');
  TOUR_CARDS.forEach((c, k) => {
    const d = tEl('button', 'tour-dot' + (k === i ? ' on' : (_tour.seen.has(k) ? ' seen' : '')));
    d.type = 'button';
    d.title = c.title;
    d.setAttribute('aria-label', c.title);
    d.addEventListener('click', () => { _tour.i = k; _tour.seen.add(k); renderTour(); });
    dots.appendChild(d);
  });
  controls.appendChild(dots);

  const last = i === TOUR_CARDS.length - 1;
  const next = tEl('button', 'tour-btn-primary');
  next.type = 'button';
  next.innerHTML = last ? 'Start using Asepsis' : 'Next <span class="tour-key">→</span>';
  next.addEventListener('click', () => last ? closeTour() : tourGo(1));
  controls.appendChild(next);
}

/* ── Wiring ─────────────────────────────────────────────────────── */

function initTour() {
  const btn = document.getElementById('tour-btn');
  if (btn) btn.addEventListener('click', () => {
    document.getElementById('settings-popover').hidden = true;
    openTour(0);
  });

  // First run only. Stored by version, so the tour can be shown again after a
  // release that changes what it describes — but never twice for the same one.
  if (prefs.get('tourSeen', 0) < TOUR_VERSION) {
    // After first paint, so the tour opens over a drawn app rather than a
    // blank one — the point is to show what the surfaces look like.
    setTimeout(() => openTour(0), 700);
  }
}

document.addEventListener('DOMContentLoaded', initTour);

/* ═══ User guide reader ══════════════════════════════════════════════
   The shipped USER_GUIDE.md, rendered in place. Fetched rather than
   duplicated here so the file in the repo is the one you read in the app —
   a hand-copied help panel drifts out of date the first time either changes.
   ═══════════════════════════════════════════════════════════════════ */

let _guide = null;

async function openGuide() {
  if (_guide) return;

  const overlay = tEl('div', 'tour-overlay guide-overlay');
  const win = tEl('div', 'tour-window guide-window');
  win.setAttribute('role', 'dialog');
  win.setAttribute('aria-modal', 'true');
  win.setAttribute('aria-label', 'User guide');

  const head = tEl('header', 'tour-head');
  const hl = tEl('div');
  hl.appendChild(tEl('p', 'tour-kicker', 'ASEPSIS · User guide'));
  hl.appendChild(tEl('p', 'tour-sub', 'Everything the app does, and why it does it that way.'));
  head.appendChild(hl);
  const close = tEl('button', 'tour-close', '✕');
  close.type = 'button';
  close.title = 'Close (Esc)';
  close.addEventListener('click', closeGuide);
  head.appendChild(close);
  win.appendChild(head);

  const body = tEl('div', 'guide-body');
  body.innerHTML = '<p class="guide-loading">Loading…</p>';
  win.appendChild(body);

  overlay.appendChild(win);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeGuide(); });
  document.body.appendChild(overlay);
  _guide = { overlay, body };
  document.addEventListener('keydown', guideKeys);
  close.focus();

  try {
    const r = await apiGet('/api/guide');
    const md = r.markdown || '';
    body.innerHTML = (typeof marked !== 'undefined')
      ? (marked.parse ? marked.parse(md) : marked(md))
      : `<pre>${escHtml(md)}</pre>`;
    // In-page anchors: the guide's contents list is the fastest way around a
    // long document, and a plain href would try to navigate the app away.
    body.querySelectorAll('a[href^="#"]').forEach(a => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const slug = decodeURIComponent(a.getAttribute('href').slice(1));
        const target = [...body.querySelectorAll('h2, h3')].find(h =>
          h.textContent.toLowerCase().replace(/[^\w\s-]/g, '').trim()
            .replace(/\s+/g, '-') === slug);
        // Positioned directly rather than scrolled smoothly: smooth scrolling
        // inside this container does not animate in every webview the app runs
        // in, and a contents link that silently does nothing is worse than one
        // that jumps. A jump is also what a table of contents normally does.
        // Measured against the scroll container itself: offsetTop is relative
        // to the nearest POSITIONED ancestor, which for a centred modal is the
        // page — so it overshoots by however far down the window sits.
        if (target) {
          const delta = target.getBoundingClientRect().top
                      - body.getBoundingClientRect().top;
          body.scrollTop = Math.max(0, body.scrollTop + delta - 12);
        }
      });
    });
    // External links leave the app rather than replacing it.
    body.querySelectorAll('a[href^="http"]').forEach(a => {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    });
    body.scrollTop = 0;
  } catch (e) {
    body.innerHTML = `<p class="guide-loading">The guide could not be loaded: `
      + `${escHtml(e.message)}</p>`;
  }
}

function closeGuide() {
  if (!_guide) return;
  document.removeEventListener('keydown', guideKeys);
  _guide.overlay.remove();
  _guide = null;
}

function guideKeys(e) {
  if (_guide && e.key === 'Escape') { e.preventDefault(); closeGuide(); }
}

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('guide-btn');
  if (btn) btn.addEventListener('click', () => {
    document.getElementById('settings-popover').hidden = true;
    openGuide();
  });
});
