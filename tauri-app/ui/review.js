/* Manual figure/table review. Annotorious owns geometry; the backend owns truth. */
(function initIngestReview(global) {
  'use strict';

  const ui = {};
  const review = {
    session: null,
    documentId: null,
    page: 1,
    kind: 'figure',
    annotator: null,
    syncing: false,
    pending: false,
    suppressEvents: false,
    annotations: new Map(),
  };

  function bindUi() {
    ui.dialog = document.getElementById('ingest-review-dialog');
    ui.documents = document.getElementById('review-documents');
    ui.pages = document.getElementById('review-pages');
    ui.regions = document.getElementById('review-regions');
    ui.image = document.getElementById('review-page-image');
    ui.message = document.getElementById('review-message');
    ui.state = document.getElementById('review-state');
    ui.sync = document.getElementById('review-sync');
    ui.confirm = document.getElementById('review-confirm');

    document.getElementById('review-close').addEventListener('click', () => ui.dialog.close());
    document.getElementById('review-undo').addEventListener('click', () => review.annotator?.undo());
    document.getElementById('review-redo').addEventListener('click', () => review.annotator?.redo());
    document.getElementById('review-confirm').addEventListener('click', confirmReview);
    document.getElementById('review-cancel').addEventListener('click', cancelReview);
    document.querySelectorAll('[data-review-kind]').forEach(button => {
      button.addEventListener('click', () => {
        review.kind = button.dataset.reviewKind;
        document.querySelectorAll('[data-review-kind]').forEach(item =>
          item.classList.toggle('active', item === button));
      });
    });
  }

  async function request(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    if (!response.ok) {
      let payload = {};
      try { payload = await response.json(); } catch { /* non-JSON failure */ }
      throw new Error(payload.message || payload.detail || payload.error ||
        `${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  function currentDocument() {
    return review.session?.documents.find(doc => doc.document_id === review.documentId);
  }

  function annotationFor(region) {
    const image = ui.image;
    const x = region.box.x * image.naturalWidth;
    const y = region.box.y * image.naturalHeight;
    const w = region.box.width * image.naturalWidth;
    const h = region.box.height * image.naturalHeight;
    return {
      id: region.region_id,
      bodies: [],
      target: {
        annotation: region.region_id,
        created: new Date(),
        creator: { id: 'asepsis-review', name: 'ASEPSIS review' },
        selector: {
          type: 'RECTANGLE',
          geometry: {
            x, y, w, h,
            bounds: { minX: x, minY: y, maxX: x + w, maxY: y + h },
          },
        },
      },
    };
  }

  function normalizedBox(annotation) {
    const geometry = annotation?.target?.selector?.geometry;
    if (!geometry || !ui.image.naturalWidth || !ui.image.naturalHeight) return null;
    const x = Math.max(0, geometry.x / ui.image.naturalWidth);
    const y = Math.max(0, geometry.y / ui.image.naturalHeight);
    const width = Math.min(1 - x, geometry.w / ui.image.naturalWidth);
    const height = Math.min(1 - y, geometry.h / ui.image.naturalHeight);
    if (!(width > 0 && height > 0)) return null;
    return { x, y, width, height };
  }

  function safeRegionId(candidate) {
    const value = String(candidate || '').replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 80);
    return /^[A-Za-z0-9]/.test(value)
      ? value
      : `manual_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
  }

  function applyAnnotation(annotation, created = false) {
    if (review.suppressEvents) return;
    const document = currentDocument();
    const box = normalizedBox(annotation);
    if (!document || !box) return;
    const mappedId = review.annotations.get(annotation.id) || annotation.id;
    const existing = document.regions.find(region => region.region_id === mappedId);
    if (existing) {
      existing.box = box;
      if (existing.kind === 'table') resetTable(existing);
    } else if (created) {
      const regionId = safeRegionId(annotation.id);
      review.annotations.set(annotation.id, regionId);
      document.regions.push({
        region_id: regionId,
        kind: review.kind,
        page: review.page,
        box,
        caption: `${review.kind === 'table' ? 'Table' : 'Figure'} (manually selected)`,
        asset_filename: `${regionId}.png`,
        deleted: false,
        table_state: review.kind === 'table' ? 'pending' : null,
        table_markdown: null,
        table_payload: null,
        table_error: null,
        table_failure_acknowledged: false,
      });
    }
    scheduleSync();
  }

  function resetTable(region) {
    region.table_state = 'pending';
    region.table_markdown = null;
    region.table_payload = null;
    region.table_error = null;
    region.table_failure_acknowledged = false;
  }

  function deleteAnnotation(annotation) {
    if (review.suppressEvents) return;
    const document = currentDocument();
    if (!document) return;
    const mappedId = review.annotations.get(annotation.id) || annotation.id;
    document.regions = document.regions.filter(region => region.region_id !== mappedId);
    review.annotations.delete(annotation.id);
    scheduleSync();
  }

  function scheduleSync() {
    review.pending = true;
    ui.sync.textContent = review.syncing ? 'Saving again…' : 'Unsaved changes';
    clearTimeout(scheduleSync.timer);
    scheduleSync.timer = setTimeout(syncRegions, 180);
  }

  async function syncRegions() {
    if (review.syncing || !review.pending) return;
    review.pending = false;
    review.syncing = true;
    ui.sync.textContent = 'Saving…';
    const document = currentDocument();
    const editableRegions = document.regions.map(region => ({
      region_id: region.region_id,
      kind: region.kind,
      page: region.page,
      box: region.box,
      caption: region.caption,
      asset_filename: region.asset_filename,
      deleted: Boolean(region.deleted),
    }));
    try {
      review.session = await request(
        `/api/ingest/reviews/${review.session.session_id}/documents/` +
        `${encodeURIComponent(document.document_id)}/regions`,
        {
          method: 'PUT',
          body: JSON.stringify({
            expected_revision: review.session.revision,
            regions: editableRegions,
          }),
        },
      );
      ui.sync.textContent = 'Saved';
      renderSession(false);
    } catch (error) {
      ui.sync.textContent = 'Save failed';
      showMessage(error.message, true);
    } finally {
      review.syncing = false;
      if (review.pending) syncRegions();
    }
  }

  async function flushSync() {
    clearTimeout(scheduleSync.timer);
    if (review.pending) await syncRegions();
    while (review.syncing) await new Promise(resolve => setTimeout(resolve, 30));
  }

  function renderSession(reloadPage = true) {
    const session = review.session;
    ui.state.textContent = session.state === 'ready' ? 'Ready to publish' : session.state;
    ui.state.classList.toggle('ready', session.state === 'ready');
    ui.confirm.disabled = session.state !== 'ready' || review.syncing || review.pending;

    ui.documents.replaceChildren(...session.documents.map(doc => {
      const button = global.document.createElement('button');
      button.type = 'button';
      button.className = `review-nav-item${doc.document_id === review.documentId ? ' active' : ''}`;
      button.textContent = doc.title;
      button.addEventListener('click', async () => {
        await flushSync();
        review.documentId = doc.document_id;
        review.page = 1;
        renderSession(true);
      });
      return button;
    }));

    const currentDoc = currentDocument();
    if (!currentDoc) return;
    ui.pages.replaceChildren(...Array.from({ length: currentDoc.page_count }, (_, index) => {
      const page = index + 1;
      const button = global.document.createElement('button');
      button.type = 'button';
      button.className = `review-page-button${page === review.page ? ' active' : ''}`;
      button.textContent = String(page);
      button.addEventListener('click', async () => {
        await flushSync();
        review.page = page;
        renderSession(true);
      });
      return button;
    }));
    renderRegionList();
    if (reloadPage) renderPage();
  }

  function renderRegionList() {
    const currentDoc = currentDocument();
    const regions = currentDoc.regions.filter(region => region.page === review.page && !region.deleted);
    if (!regions.length) {
      const empty = global.document.createElement('p');
      empty.className = 'review-empty';
      empty.textContent = 'No figures or tables on this page.';
      ui.regions.replaceChildren(empty);
      return;
    }
    ui.regions.replaceChildren(...regions.map(region => {
      const item = global.document.createElement('article');
      item.className = `review-region review-region-${region.kind}`;
      const heading = global.document.createElement('div');
      heading.className = 'review-region-heading';
      const label = global.document.createElement('strong');
      label.textContent = region.caption;
      const remove = global.document.createElement('button');
      remove.type = 'button';
      remove.textContent = 'Delete';
      remove.addEventListener('click', () => review.annotator?.removeAnnotation(region.region_id));
      heading.append(label, remove);
      item.appendChild(heading);
      if (region.kind === 'table') item.appendChild(tableControls(region));
      return item;
    }));
  }

  function tableControls(region) {
    const wrap = document.createElement('div');
    wrap.className = `review-table-state review-table-${region.table_state}`;
    const status = document.createElement('span');
    status.textContent = region.table_state === 'recognized'
      ? 'Structured table ready'
      : region.table_state === 'failed'
        ? `Recognition failed: ${region.table_error || 'unknown error'}`
        : 'Recognition required';
    wrap.appendChild(status);
    const action = document.createElement('button');
    action.type = 'button';
    if (region.table_state === 'failed' && !region.table_failure_acknowledged) {
      action.textContent = 'Acknowledge crop only';
      action.addEventListener('click', () => tableAction(region, 'acknowledge-table-failure'));
    } else {
      action.textContent = region.table_state === 'recognized' ? 'Recognize again' : 'Recognize table';
      action.addEventListener('click', () => tableAction(region, 'recognize-table'));
    }
    wrap.appendChild(action);
    return wrap;
  }

  async function tableAction(region, action) {
    await flushSync();
    const document = currentDocument();
    showMessage(action === 'recognize-table' ? 'Recognizing table locally…' : 'Saving acknowledgement…');
    try {
      review.session = await request(
        `/api/ingest/reviews/${review.session.session_id}/documents/` +
        `${encodeURIComponent(document.document_id)}/regions/` +
        `${encodeURIComponent(region.region_id)}/${action}`,
        {
          method: 'POST',
          body: JSON.stringify({ expected_revision: review.session.revision }),
        },
      );
      showMessage('');
      renderSession(false);
    } catch (error) {
      showMessage(error.message, true);
    }
  }

  function renderPage() {
    review.annotator?.destroy();
    review.annotator = null;
    const document = currentDocument();
    const src = document.page_href_template.replace('{page}', String(review.page));
    ui.image.onload = () => {
      review.annotator = global.Annotorious.createImageAnnotator(ui.image, {
        drawingEnabled: true,
        drawingMode: 'drag',
        theme: 'light',
      });
      review.annotator.setDrawingTool('rectangle');
      review.annotator.on('createAnnotation', annotation => applyAnnotation(annotation, true));
      review.annotator.on('updateAnnotation', annotation => applyAnnotation(annotation, false));
      review.annotator.on('deleteAnnotation', deleteAnnotation);
      const annotations = document.regions
        .filter(region => region.page === review.page && !region.deleted)
        .map(annotationFor);
      review.annotations.clear();
      for (const annotation of annotations) {
        review.annotations.set(annotation.id, annotation.id);
      }
      review.suppressEvents = true;
      review.annotator.setAnnotations(annotations);
      review.suppressEvents = false;
    };
    ui.image.onerror = () => showMessage('Could not render this PDF page.', true);
    ui.image.src = `${src}?revision=${review.session.revision}`;
  }

  function showMessage(message, error = false) {
    ui.message.textContent = message || '';
    ui.message.classList.toggle('error', error);
  }

  async function confirmReview() {
    await flushSync();
    if (review.session.state !== 'ready') return;
    ui.confirm.disabled = true;
    showMessage('Validating and publishing the complete generation…');
    try {
      review.session = await request(
        `/api/ingest/reviews/${review.session.session_id}/confirm`,
        {
          method: 'POST',
          body: JSON.stringify({ expected_revision: review.session.revision }),
        },
      );
      ui.dialog.close();
      global.dispatchEvent(new CustomEvent('asepsis-library-published', {
        detail: { generationId: review.session.published_generation_id },
      }));
    } catch (error) {
      showMessage(error.message, true);
      renderSession(false);
    }
  }

  async function cancelReview() {
    await flushSync();
    try {
      review.session = await request(
        `/api/ingest/reviews/${review.session.session_id}/cancel`,
        {
          method: 'POST',
          body: JSON.stringify({ expected_revision: review.session.revision }),
        },
      );
      ui.dialog.close();
    } catch (error) {
      showMessage(error.message, true);
    }
  }

  async function open(sessionId) {
    if (!ui.dialog) bindUi();
    review.session = await request(`/api/ingest/reviews/${encodeURIComponent(sessionId)}`);
    review.documentId = review.session.documents[0]?.document_id || null;
    review.page = 1;
    review.pending = false;
    review.syncing = false;
    showMessage('');
    renderSession(true);
    ui.dialog.showModal();
  }

  async function resumeActive() {
    const active = await request('/api/ingest/review-active');
    if (!active.review_id) return false;
    await open(active.review_id);
    return true;
  }

  global.AsepsisReview = Object.freeze({ open, resumeActive });
})(window);
