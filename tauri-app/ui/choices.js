/* ════════════════════════════════════════════════════════════════════
   Multiple-choice pre-filter — the questions answered before the query.

   Each answer maps to a set of leaves judged relevant to it once, at index
   time. Selections are UNIONED: picking more answers WIDENS the candidate
   set rather than narrowing it. That is easy to get backwards, so the bar
   states the direction and shows the resulting candidate count as you click.

   Selections are sent as `selected_answers` on /api/chat and /api/run.
   ════════════════════════════════════════════════════════════════════ */

const choiceState = {
  questions: [],
  selected: new Set(),
  precompute: null,
  loaded: false,
};

function selectedAnswers() {
  return [...choiceState.selected];
}

async function loadChoiceQuestions() {
  try {
    const data = await apiGet('/api/choices');
    choiceState.questions = data.questions || [];
    choiceState.precompute = data.precompute || null;
    choiceState.loaded = true;
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

  const header = el('div', 'choice-head');
  header.appendChild(el('span', 'choice-eyebrow', 'Narrow the search'));

  const state = el('span', 'choice-state');
  header.appendChild(state);

  const clear = el('button', 'choice-clear', 'Clear');
  clear.hidden = choiceState.selected.size === 0;
  clear.addEventListener('click', () => {
    choiceState.selected.clear();
    renderChoiceBars();
    refreshChoiceCount();
  });
  header.appendChild(clear);
  host.appendChild(header);

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
      if (choiceState.selected.has(a.id)) btn.classList.add('on');
      btn.addEventListener('click', () => {
        if (choiceState.selected.has(a.id)) {
          choiceState.selected.delete(a.id);
        } else {
          if (!q.multi_select) {
            for (const other of q.answers) choiceState.selected.delete(other.id);
          }
          choiceState.selected.add(a.id);
        }
        renderChoiceBars();
        refreshChoiceCount();
      });
      opts.appendChild(btn);
    }
    row.appendChild(opts);
    host.appendChild(row);
  }

  // Direction of the effect, stated plainly — with union semantics, more
  // answers means a bigger candidate set, which reads as backwards unless said.
  const n = choiceState.selected.size;
  state.textContent = n === 0
    ? 'no filter — the whole library is searched'
    : `${n} selected · passages matching any of them are searched`;

  const pre = choiceState.precompute;
  if (pre && pre.state === 'running') {
    const note = el('p', 'choice-note',
      `Preparing pre-filter judgements… ${pre.message || ''}`);
    host.appendChild(note);
  } else if (pre && pre.state === 'idle' && pre.calls === 0) {
    const note = el('p', 'choice-note');
    note.textContent = 'Pre-filter judgements have not been computed yet — ';
    const link = el('button', 'choice-link', 'compute them now');
    link.addEventListener('click', startChoicePrecompute);
    note.appendChild(link);
    note.appendChild(document.createTextNode(
      '. Until then these answers cannot narrow anything.'));
    host.appendChild(note);
  }
}

/* The candidate count makes the union's widening visible instead of
   surprising: it is the number of passages the query will actually run over. */
let _countTimer = null;

function refreshChoiceCount() {
  clearTimeout(_countTimer);
  _countTimer = setTimeout(async () => {
    const target = document.querySelectorAll('.choice-state');
    if (!choiceState.selected.size) return;
    try {
      const r = await apiGet('/api/choices/candidates?selected_answers=' +
        encodeURIComponent(selectedAnswers().join(',')));
      const txt = `${choiceState.selected.size} selected · ` +
        `${r.candidates} of ${r.total} passages in scope`;
      target.forEach(t => { t.textContent = txt; });
    } catch { /* leave the plain description */ }
  }, 220);
}

async function startChoicePrecompute() {
  try {
    await apiPost('/api/choices/precompute', {});
    toast('Computing pre-filter judgements — this runs once per corpus.', 'ok', 6000);
    pollChoicePrecompute();
  } catch (e) {
    toast(`Could not start the precompute: ${e.message}`, 'err', 7000);
  }
}

function pollChoicePrecompute() {
  const tick = async () => {
    let st;
    try { st = await apiGet('/api/choices/precompute'); }
    catch { setTimeout(tick, 2000); return; }
    choiceState.precompute = st;
    renderChoiceBars();
    if (st.state === 'running') { setTimeout(tick, 1500); return; }
    if (st.state === 'done') {
      toast(`Pre-filter ready — ${st.calls} judgement${st.calls === 1 ? '' : 's'} made, ` +
            `${st.reused} reused.`, 'ok', 6000);
    } else if (st.state === 'error') {
      toast(`Pre-filter precompute failed: ${st.message}`, 'err', 8000);
    }
  };
  tick();
}
