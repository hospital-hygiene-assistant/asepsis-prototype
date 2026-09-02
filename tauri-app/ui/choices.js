/* ════════════════════════════════════════════════════════════════════
   Multiple-choice pre-filter — the questions answered before the query.

   Each answer maps to a set of passages judged relevant to it once, at index
   time, so applying them at query time costs no model calls at all.

   Two modes, and the user picks:
     ANY  (default) — a passage matching ANY selected answer is searched.
                      More answers = wider search. Recall-first, because a
                      passage excluded here can never reach the answer.
     ALL  (Fast mode) — a passage must satisfy EVERY question answered.
                      Much smaller, much faster, and it will drop passages a
                      clinician might have wanted. Opt-in, never default.

   The bar is one collapsed line until you open it: the app's chrome should
   not cost more vertical space than the question being asked. Selections and
   mode are sent as `selected_answers` / `selection_mode` on /api/chat and
   /api/run.
   ════════════════════════════════════════════════════════════════════ */

const choiceState = {
  questions: [],
  selected: new Set(),
  mode: 'any',           // 'any' = recall-first union, 'all' = Fast mode
  precompute: null,
  open: false,
  count: null,           // {candidates, total, unjudged} from the last probe
  loaded: false,
};

function selectedAnswers() {
  return [...choiceState.selected];
}

function selectionMode() {
  return choiceState.mode;
}

function choiceAnswerById(id) {
  for (const q of choiceState.questions) {
    for (const a of q.answers) if (a.id === id) return a;
  }
  return null;
}

/* Which questions actually have an answer selected — the unit Fast mode
   operates on, and the only honest way to describe what it will do. */
function answeredQuestions() {
  return choiceState.questions.filter(
    q => q.answers.some(a => choiceState.selected.has(a.id)));
}

async function loadChoiceQuestions() {
  try {
    const data = await apiGet('/api/choices');
    choiceState.questions = data.questions || [];
    choiceState.precompute = data.precompute || null;
    choiceState.loaded = true;
    // A pass left running by a previous page load is still running on the
    // server; pick it up rather than showing a stale "not started".
    if (choiceState.precompute?.state === 'running') pollChoicePrecompute();
  } catch {
    choiceState.questions = [];
    choiceState.loaded = true;
  }
  renderChoiceBars();
}

function renderChoiceBars() {
  for (const id of ['choice-bar', 'chat-choice-bar']) {
    const host = document.getElementById(id);
    if (host) renderChoiceBar(host);
  }
}

function renderChoiceBar(host) {
  host.innerHTML = '';
  if (!choiceState.questions.length) { host.hidden = true; return; }
  host.hidden = false;
  host.classList.toggle('open', choiceState.open);

  host.appendChild(buildChoiceSummary());
  if (choiceState.open) host.appendChild(buildChoicePanel());

  const notice = buildPrecomputeNotice();
  if (notice) host.appendChild(notice);
}

/* ── The collapsed line: what is filtering, and how much it leaves ── */

function buildChoiceSummary() {
  // A row, not one big button: the chips inside it are individually removable,
  // and a button nested in a button is neither valid nor clickable.
  const bar = el('div', 'choice-summary');
  const toggle = () => { choiceState.open = !choiceState.open; renderChoiceBars(); };

  const handle = el('button', 'choice-toggle');
  handle.type = 'button';
  handle.setAttribute('aria-expanded', String(choiceState.open));
  handle.title = choiceState.open ? 'Collapse the filters' : 'Choose what to search';
  handle.addEventListener('click', toggle);
  const caret = el('span', 'choice-caret');
  caret.textContent = '▸';
  handle.appendChild(caret);
  handle.appendChild(el('span', 'choice-eyebrow', 'Filters'));
  if (choiceState.selected.size) {
    handle.appendChild(el('span', 'choice-count', String(choiceState.selected.size)));
  }
  bar.appendChild(handle);

  const chips = el('span', 'choice-chips');
  if (!choiceState.selected.size) {
    const none = el('button', 'choice-empty', 'none — searching the whole library');
    none.type = 'button';
    none.title = 'Choose what to search';
    none.addEventListener('click', toggle);
    chips.appendChild(none);
  } else {
    for (const id of choiceState.selected) {
      const answer = choiceAnswerById(id);
      // Removable in place. Undoing one filter is the commonest thing anyone
      // wants from this bar, and it should not require opening the panel and
      // hunting for which question the answer belonged to.
      const chip = el('span', 'choice-chip');
      chip.appendChild(el('b', null, answer ? answer.label : id));
      const x = el('button', 'choice-chip-x', '✕');
      x.type = 'button';
      x.title = `Remove “${answer ? answer.label : id}”`;
      x.setAttribute('aria-label', `Remove filter ${answer ? answer.label : id}`);
      x.addEventListener('click', (e) => {
        e.stopPropagation();
        choiceState.selected.delete(id);
        if (!choiceState.selected.size) choiceState.count = null;
        renderChoiceBars();
        refreshChoiceCount();
      });
      chip.appendChild(x);
      chips.appendChild(chip);
    }
  }
  bar.appendChild(chips);
  bar.appendChild(buildChoiceScope());
  return bar;
}

/* How much of the library the next question will actually read.
   A number alone ("148 of 512") does not convey how hard a filter is biting;
   the bar does, at a glance, and turns amber once it is biting hard enough to
   be the likely reason an answer comes back thin. */
function buildChoiceScope() {
  const wrap = el('span', 'choice-scope');
  const count = choiceState.count;
  if (!choiceState.selected.size) return wrap;
  if (!count) { wrap.appendChild(el('span', 'choice-scope-text', 'counting…')); return wrap; }

  const total = count.total || 0;
  const frac = total ? Math.max(0, Math.min(1, count.candidates / total)) : 0;
  const meter = el('span', 'choice-meter');
  const fill = el('i');
  fill.style.width = `${Math.max(frac * 100, count.candidates ? 1.5 : 0)}%`;
  meter.appendChild(fill);
  if (frac <= 0.12) meter.classList.add('tight');
  wrap.appendChild(meter);

  const pct = Math.round(frac * 100);
  const text = el('span', 'choice-scope-text',
    `${count.candidates} of ${total}` + (choiceState.mode === 'all' ? ' · fast' : ''));
  wrap.appendChild(text);
  wrap.title = `${count.candidates} of ${total} passages (${pct}%) will be searched`
    + (count.unjudged ? ` · ${count.unjudged} not yet judged, searched unfiltered` : '')
    + (choiceState.mode === 'all' ? ' · Fast mode: a passage must match every answer' : '');
  return wrap;
}

/* The count is the honest description of what the query will read. It is
   also the only place the two modes are visibly different, so it carries
   the mode's name rather than making the user remember which is on. */
function choiceScopeText() {
  const count = choiceState.count;
  if (!choiceState.selected.size) return '';
  if (!count) return 'counting…';
  const fast = choiceState.mode === 'all' ? ' · fast' : '';
  return `${count.candidates} of ${count.total} passages${fast}`;
}

/* ── The expanded panel: the questions themselves ── */

function buildChoicePanel() {
  const panel = el('div', 'choice-panel');

  for (const q of choiceState.questions) {
    const answeredHere = q.answers.filter(a => choiceState.selected.has(a.id));
    const row = el('div', 'choice-row' + (answeredHere.length ? ' answered' : ''));

    // The prompt sits ABOVE its options rather than in a right-aligned column:
    // the column forced every question to the same width, so a long prompt
    // wrapped to three lines beside two short chips.
    const head = el('div', 'choice-row-head');
    const label = el('span', 'choice-q', q.prompt);
    if (q.help_text) label.title = q.help_text;
    head.appendChild(label);
    if (q.multi_select) head.appendChild(el('span', 'choice-multi', 'pick any'));
    if (answeredHere.length) {
      const undo = el('button', 'choice-row-clear', 'clear');
      undo.type = 'button';
      undo.title = `Unset this question`;
      undo.addEventListener('click', () => {
        for (const a of q.answers) choiceState.selected.delete(a.id);
        if (!choiceState.selected.size) choiceState.count = null;
        renderChoiceBars();
        refreshChoiceCount();
      });
      head.appendChild(undo);
    }
    row.appendChild(head);

    const opts = el('div', 'choice-opts');
    for (const a of q.answers) {
      const btn = el('button', 'choice-opt', a.label);
      btn.type = 'button';
      btn.title = `Matches passages about: ${a.facet}`;
      btn.setAttribute('aria-pressed', String(choiceState.selected.has(a.id)));
      if (choiceState.selected.has(a.id)) btn.classList.add('on');
      btn.addEventListener('click', () => toggleAnswer(q, a));
      opts.appendChild(btn);
    }
    row.appendChild(opts);
    panel.appendChild(row);
  }

  panel.appendChild(buildChoiceFooter());
  return panel;
}

function toggleAnswer(question, answer) {
  if (choiceState.selected.has(answer.id)) {
    choiceState.selected.delete(answer.id);
  } else {
    if (!question.multi_select) {
      for (const other of question.answers) choiceState.selected.delete(other.id);
    }
    choiceState.selected.add(answer.id);
  }
  renderChoiceBars();
  refreshChoiceCount();
}

function buildChoiceFooter() {
  const foot = el('div', 'choice-foot');

  /* Fast mode. Off by default and described in terms of what it COSTS, not
     what it saves — the saving is obvious from the count, the cost is not. */
  const answered = answeredQuestions().length;
  // A checkbox labelled "Fast mode" said what it turns ON and nothing about
  // what it turns OFF. Two named states say what the search IS either way,
  // and the wider one being the left-hand default is the point.
  const seg = el('div', 'choice-modeseg');
  seg.setAttribute('role', 'group');
  seg.setAttribute('aria-label', 'How the answers combine');
  for (const [mode, label, tip] of [
    ['any', 'Any answer',
     'A passage matching ANY of your answers is searched. Wider, and the '
     + 'default — a passage excluded here can never reach the answer.'],
    ['all', 'Every answer',
     'A passage must match EVERY question you answered. Much faster and far '
     + 'more focused — and it will drop passages that only matched some.'],
  ]) {
    const b = el('button', 'choice-mode' + (choiceState.mode === mode ? ' on' : ''), label);
    b.type = 'button';
    b.title = tip;
    b.setAttribute('aria-pressed', String(choiceState.mode === mode));
    b.addEventListener('click', () => {
      if (choiceState.mode === mode) return;
      choiceState.mode = mode;
      renderChoiceBars();
      refreshChoiceCount();
    });
    seg.appendChild(b);
  }
  foot.appendChild(seg);

  // Describes the state the toggle is IN, never the state it would move to —
  // sitting beside the checkbox, an unlabelled description reads as a
  // description of Fast mode itself.
  const explain = el('span', 'choice-explain', choiceState.mode === 'all'
    ? (answered > 1
        ? `a passage must match all ${answered} questions — may miss some`
        : 'starts narrowing once you answer a second question')
    : 'a passage matching any one of your answers is searched');
  foot.appendChild(explain);

  if (choiceState.selected.size) {
    const clear = el('button', 'choice-clear', 'Clear all');
    clear.type = 'button';
    clear.addEventListener('click', () => {
      choiceState.selected.clear();
      choiceState.count = null;
      renderChoiceBars();
      refreshChoiceCount();
    });
    foot.appendChild(clear);
  }
  return foot;
}

/* ── Precompute: whether these answers can filter anything at all ── */

function buildPrecomputeNotice() {
  const pre = choiceState.precompute;
  if (!pre) return null;
  const cov = pre.coverage || {};

  if (pre.state === 'running') {
    const note = el('div', 'choice-note');
    note.appendChild(el('span', 'choice-spinner'));
    note.appendChild(document.createTextNode(pre.message || 'Preparing…'));
    const stop = el('button', 'choice-link', 'stop');
    stop.type = 'button';
    stop.addEventListener('click', cancelChoicePrecompute);
    note.appendChild(stop);
    return note;
  }

  // Everything below is read from disk, so it survives a restart and notices
  // documents ingested since the last pass.
  if (cov.complete || !cov.docs?.length) return null;

  const note = el('div', 'choice-note warn');
  const missing = (cov.docs || []).filter(d => d.state !== 'ready');
  note.textContent = cov.usable
    ? `${missing.length} document${missing.length === 1 ? '' : 's'} not yet judged — `
      + `${missing.length === 1 ? 'it is' : 'they are'} searched unfiltered. `
    : 'These answers cannot filter anything yet — no judgements have been computed. ';
  const link = el('button', 'choice-link',
    cov.usable ? 'update them' : 'compute them now');
  link.type = 'button';
  link.addEventListener('click', startChoicePrecompute);
  note.appendChild(link);
  return note;
}

/* ── The candidate count ──────────────────────────────────────────────
   Sequenced, not just debounced: responses can land out of order, and an
   older one overwriting a newer one shows a count for a selection the user
   has already changed. */
let _countTimer = null;
let _countSeq = 0;

function refreshChoiceCount() {
  clearTimeout(_countTimer);
  if (!choiceState.selected.size) {
    choiceState.count = null;
    renderChoiceBars();
    return;
  }
  const seq = ++_countSeq;
  _countTimer = setTimeout(async () => {
    const params = new URLSearchParams({
      selected_answers: selectedAnswers().join(','),
      mode: choiceState.mode,
    });
    try {
      const r = await apiGet(`/api/choices/candidates?${params}`);
      if (seq !== _countSeq) return;      // a newer selection is in flight
      choiceState.count = r;
      renderChoiceBars();
    } catch { /* leave the previous count rather than blanking it */ }
  }, 200);
}

async function startChoicePrecompute() {
  try {
    await apiPost('/api/choices/precompute', {});
    toast('Judging every passage against the questions — this runs once per corpus.',
          'ok', 6000);
    pollChoicePrecompute();
  } catch (e) {
    toast(`Could not start the precompute: ${e.message}`, 'err', 7000);
  }
}

async function cancelChoicePrecompute() {
  try {
    await apiPost('/api/choices/precompute/cancel', {});
    toast('Stopping — judgements already made are kept.', 'warn', 4000);
  } catch (e) {
    toast(`Could not stop the precompute: ${e.message}`, 'err', 6000);
  }
}

let _prePolling = false;

function pollChoicePrecompute() {
  if (_prePolling) return;
  _prePolling = true;
  const tick = async () => {
    let st;
    try { st = await apiGet('/api/choices/precompute'); }
    catch { setTimeout(tick, 2000); return; }
    choiceState.precompute = st;
    renderChoiceBars();
    if (st.state === 'running') { setTimeout(tick, 1500); return; }
    _prePolling = false;
    if (st.state === 'done') {
      toast(`Pre-filter ready — ${st.calls} judgement${st.calls === 1 ? '' : 's'} made, `
            + `${st.reused} reused.`, 'ok', 6000);
      refreshChoiceCount();
    } else if (st.state === 'cancelled') {
      toast(`Stopped — ${st.calls} judgement${st.calls === 1 ? '' : 's'} kept.`,
            'warn', 5000);
    } else if (st.state === 'error') {
      toast(`Pre-filter precompute failed: ${st.message}`, 'err', 8000);
    }
  };
  tick();
}
