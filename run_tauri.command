#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  Asepsis Prototype — one-click launcher (macOS)
#
#  Double-click this file in Finder. It installs anything that is
#  missing (a private Python environment, the Python packages, the
#  local Ollama model) and then starts the app — desktop window if
#  the Tauri CLI is available, otherwise the default browser.
# ══════════════════════════════════════════════════════════════════

cd "$(dirname "$0")" || exit 1

say()  { printf '\033[1;36m▸ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*"; printf 'Press Enter to close… '; read -r; exit 1; }

echo
echo "  Asepsis Prototype"
echo "  ────────────────────────────"

# ── 1 · Python ────────────────────────────────────────────────
PY=""
for cand in python3.12 python3.13 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
[ -n "$PY" ] || die "python3 was not found. Install it from https://www.python.org/downloads/ and double-click again."
say "Python: $($PY --version 2>&1)"

# ── 2 · Private environment + dependencies ────────────────────
if [ ! -x .venv/bin/python ]; then
  say "Creating the app's Python environment (.venv)…"
  "$PY" -m venv .venv || die "Could not create the virtual environment."
fi
say "Checking Python dependencies…"
.venv/bin/python -m pip install -q --upgrade pip >/dev/null 2>&1
.venv/bin/python -m pip install -q -r tauri-app/requirements-tauri.txt \
  || die "Dependency installation failed. Check your network connection and retry."

# ── 3 · Ollama runtime + model ────────────────────────────────
# The model comes from config.py, so the launcher can never pull a different
# one than the app actually calls. (It used to hardcode gemma3:4b while the
# app asked Ollama for something else — every call then failed.)
MODEL="$(.venv/bin/python -c 'import config; print(config.DEFAULT_RETRIEVAL_MODEL)' 2>/dev/null)"
[ -n "$MODEL" ] || MODEL="gemma4:e4b"
if ! command -v ollama >/dev/null 2>&1; then
  die "Ollama is not installed. Download it from https://ollama.com/download, open it once, then double-click this file again."
fi
if ! curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  say "Starting the Ollama server…"
  (ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 40); do
    curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 0.5
  done
  curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
    || warn "Ollama did not answer yet — the app will still open; retrieval starts working once it is up."
fi
if ! ollama list 2>/dev/null | grep -q "^${MODEL}[[:space:]]"; then
  say "Downloading the language model ($MODEL — first launch only)…"
  ollama pull "$MODEL" || die "Could not pull '"'"'$MODEL'"'"'. Check the tag exists (\`ollama pull $MODEL\`) or set a different model in config.py, then retry."
fi

# Ollama serves one request at a time by default, which caps indexing and
# retrieval at single-stream speed. Measured on this project: ~2x faster with
# batching enabled. Only a hint — changing a user's system-wide environment
# without asking would be overstepping.
if [ "$(launchctl getenv OLLAMA_NUM_PARALLEL 2>/dev/null)" = "" ]; then
  say "Tip: Ollama is set to one request at a time. For ~2x faster indexing:"
  say "     launchctl setenv OLLAMA_NUM_PARALLEL 4"
  say "     launchctl setenv OLLAMA_FLASH_ATTENTION 1"
  say "     …then restart Ollama."
fi

# ── 4 · Launch ────────────────────────────────────────────────
say "Starting the app…"
echo
.venv/bin/python run-tauri.py "$@"
status=$?
if [ $status -ne 0 ]; then
  printf 'The app exited with an error (%s). Press Enter to close… ' "$status"
  read -r
fi
exit $status
