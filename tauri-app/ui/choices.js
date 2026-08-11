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
  const bar = el('button', 'choice-summary');
  bar.type = 'button';
  bar.setAttribute('aria-expanded', String(choiceState.open));
  bar.addEventListener('click', () => {
    choiceState.open = !choiceState.open;
    renderChoiceBars();
  });

  const caret = el('span', 'choice-caret');
  caret.textContent = choiceState.open ? '▾' : '▸';
  bar.appendChild(caret);
  bar.appendChild(el('span', 'choice-eyebrow', 'Filters'));

  const chips = el('span', 'choice-chips');
  if (!choiceState.selected.size) {
    chips.appendChild(el('span', 'choice-empty', 'none — the whole library'));
  } else {
    for (const id of choiceState.selected) {
      const answer = choiceAnswerById(id);
      chips.appendChild(el('span', 'choice-chip', answer ? answer.label : id));
    }
  }
  bar.appendChild(chips);

  const scope = el('span', 'choice-scope', choiceScopeText());
  bar.appendChild(scope);
  return bar;
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
    const row = el('div', 'choice-row');
    const label = el('span', 'choice-q', q.prompt);
    if (q.help_text) label.title = q.help_text;
    row.appendChild(label);

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
  const toggle = el('label', 'choice-fast');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = choiceState.mode === 'all';
  box.addEventListener('change', () => {
    choiceState.mode = box.checked ? 'all' : 'any';
    renderChoiceBars();
    refreshChoiceCount();
  });
  toggle.appendChild(box);
  toggle.appendChild(el('span', 'choice-fast-label', 'Fast mode'));
  toggle.title = answered > 1
    ? 'A passage must match EVERY question you answered, not just one of them. '
      + 'Much faster and far more focused — and it will drop passages that only '
      + 'matched some of your answers.'
    : 'Requires a passage to match every question you answer. Answer a second '
      + 'question to see the difference.';
  foot.appendChild(toggle);

  // Describes the state the toggle is IN, never the state it would move to —
  // sitting beside the checkbox, an unlabelled description reads as a
  // description of Fast mode itself.
  const explain = el('span', 'choice-explain', choiceState.mode === 'all'
    ? (answered > 1
        ? `on · a passage must match all ${answered} questions — may miss some`
        : 'on · it starts narrowing once you answer a second question')
    : 'off · a passage matching any answer is searched');
  foot.appendChild(explain);

  if (choiceState.selected.size) {
    const clear = el('button', 'choice-clear', 'Clear');
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
