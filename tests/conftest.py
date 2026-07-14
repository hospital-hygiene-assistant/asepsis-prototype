"""Put the backend importable before any test module loads.

The repo is not installed as a package (no pyproject), and `tauri-app/` is not a
package at all, so both the root and `tauri-app/` have to be on `sys.path` for
`import pageindex` / `import server` to resolve. pytest imports conftest before
collecting, so doing it once here replaces the same three lines in every file.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tauri-app"):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
