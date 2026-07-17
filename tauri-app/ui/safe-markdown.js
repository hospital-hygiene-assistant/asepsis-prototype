/* One rendering interface for every untrusted Markdown value shown locally. */
(function initSafeMarkdown(global) {
  'use strict';

  const SAFE_LINK = /^(?:https?:|mailto:|\/|#)/i;

  function render(markdown) {
    if (!global.marked || !global.DOMPurify) {
      throw new Error('The Markdown renderer is unavailable');
    }
    const source = String(markdown || '').replace(
      /^[\u200B\u200C\u200D\u200E\u200F\uFEFF]+/,
      '',
    );
    const parsed = global.marked.parse(source);
    const clean = global.DOMPurify.sanitize(parsed, {
      USE_PROFILES: { html: true },
      ALLOW_DATA_ATTR: false,
      FORBID_TAGS: ['form', 'input', 'button', 'textarea', 'select', 'option'],
    });

    const template = document.createElement('template');
    template.innerHTML = clean;
    for (const link of template.content.querySelectorAll('a[href]')) {
      const href = link.getAttribute('href') || '';
      if (!SAFE_LINK.test(href)) {
        link.removeAttribute('href');
        continue;
      }
      if (/^https?:/i.test(href)) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
      }
    }
    return template.innerHTML;
  }

  global.AsepsisMarkdown = Object.freeze({ render });
})(window);
