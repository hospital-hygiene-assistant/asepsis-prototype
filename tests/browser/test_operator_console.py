"""Characterise the vendored browser modules on the real browser interface."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "tauri-app" / "ui"

pytestmark = pytest.mark.browser


def _load(page, *scripts: str) -> None:
    for script in scripts:
        page.add_script_tag(path=UI / script)


def test_untrusted_markdown_is_sanitized_before_insertion(page):
    page.set_content('<main id="output"></main>')
    _load(page, "marked.min.js", "dompurify.min.js", "safe-markdown.js")
    page.locator("#output").evaluate(
        """(node) => {
          node.innerHTML = window.AsepsisMarkdown.render(
            '# Safe\\n<img src=x onerror="window.PWNED=1">' +
            '<script>window.PWNED=2</script>' +
            '[bad](javascript:window.PWNED=3)'
          );
        }"""
    )
    assert page.locator("#output h1").text_content() == "Safe"
    assert page.locator("#output script").count() == 0
    assert page.locator("#output [onerror]").count() == 0
    assert page.locator("#output a[href^='javascript:']").count() == 0
    assert page.evaluate("window.PWNED") is None


def test_annotorious_rectangle_edit_delete_and_undo(page):
    page.set_content(
        '<img id="page" width="400" height="600" '
        'src="data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' '
        'width=\'400\' height=\'600\'%3E%3Crect width=\'400\' height=\'600\' '
        'fill=\'white\'/%3E%3C/svg%3E">'
    )
    _load(page, "annotorious.js")
    page.wait_for_function("document.getElementById('page').complete")
    result = page.evaluate(
        """async () => {
          const image = document.getElementById('page');
          const anno = Annotorious.createImageAnnotator(image, { drawingMode: 'drag' });
          const rect = {
            id: 'figure_1', bodies: [],
            target: {
              annotation: 'figure_1',
              created: new Date(),
              creator: { id: 'asepsis-test', name: 'ASEPSIS test' },
              selector: {
              type: 'RECTANGLE', geometry: {
                x: 20, y: 30, w: 100, h: 120,
                bounds: { minX: 20, minY: 30, maxX: 120, maxY: 150 }
              }
            }}
          };
          anno.setAnnotations([rect]);
          // Annotorious batches local changes inside a 250 ms history window.
          // Let the initial remote synchronization finish before characterizing
          // the first operator edit.
          await new Promise(resolve => setTimeout(resolve, 300));
          const original = anno.getAnnotations()[0];
          const changed = structuredClone(original);
          changed.target.selector.geometry.x = 40;
          changed.target.selector.geometry.bounds.minX = 40;
          changed.target.selector.geometry.bounds.maxX = 140;
          anno.updateAnnotation(changed);
          const moved = anno.getAnnotations()[0].target.selector.geometry.x;
          anno.removeAnnotation('figure_1');
          const removed = anno.getAnnotations().length;
          anno.undo();
          const restored = anno.getAnnotations().length;
          anno.destroy();
          return { moved, removed, restored };
        }"""
    )
    assert result == {"moved": 40, "removed": 0, "restored": 1}


def test_native_dialog_provides_modal_focus_and_escape(page):
    page.set_content(
        '<button id="outside">Outside</button><dialog id="review">'
        '<button id="inside" autofocus>Inside</button></dialog>'
    )
    page.evaluate("document.getElementById('review').showModal()")
    assert page.locator("#review").evaluate("node => node.open") is True
    assert page.evaluate("document.activeElement.id") == "inside"
    page.keyboard.press("Escape")
    assert page.locator("#review").evaluate("node => node.open") is False
