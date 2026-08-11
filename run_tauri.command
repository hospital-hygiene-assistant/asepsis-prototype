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
# Ollama serves one request at a time by default, which caps indexing and
# retrieval at single-stream speed. Measured on this project: 4 concurrent
# calls go from no speedup at all to ~2x, and flash attention adds another
# ~1.7x on top.
#
# When WE start the server we set these on the child process, which is not a
# system change and needs nobody's permission. When it is already running we
# can only report it — see below.
OLLAMA_TUNED_BY_LAUNCHER=0
if ! curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  say "Starting the Ollama server (4 parallel slots, flash attention)…"
  OLLAMA_TUNED_BY_LAUNCHER=1
  (OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-4}" \
   OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}" \
   ollama serve >/dev/null 2>&1 &)
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

# An Ollama that was ALREADY running is the common case on macOS, where the
# app starts at login — and it is the case the old check got wrong. It asked
# `launchctl getenv`, which describes the environment NEW processes will
# inherit, not the one the running server actually has. Set the variables and
# it would report success while the live server still had a single slot.
#
# So ask the server itself. `-np` appears on the llama-server process Ollama
# spawns per loaded model; it is absent until a model is loaded, in which case
# we say we could not tell rather than guessing.
if [ "$OLLAMA_TUNED_BY_LAUNCHER" -eq 0 ]; then
  slots="$(pgrep -fl llama-server 2>/dev/null | grep -o -- '-np [0-9]*' | head -1 | awk '{print $2}')"
  if [ -n "$slots" ] && [ "$slots" -lt 2 ] 2>/dev/null; then
    warn "Ollama is already running and serving one request at a time, which caps"
    warn "indexing and retrieval at single-stream speed (measured here: ~3x slower)."
    say  "     To fix it, quit Ollama and start it from this launcher, or run:"
    say  "         launchctl setenv OLLAMA_NUM_PARALLEL 4"
    say  "         launchctl setenv OLLAMA_FLASH_ATTENTION 1"
    say  "     …then RESTART Ollama — a running server keeps the settings it started with."
  fi
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
