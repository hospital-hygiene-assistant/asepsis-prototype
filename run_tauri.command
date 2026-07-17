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

# ── 2 · Locked environment + dependencies ─────────────────────
command -v uv >/dev/null 2>&1 \
  || die "uv was not found. Install it from https://docs.astral.sh/uv/ and retry."
say "Synchronising the locked Python environment…"
uv sync --frozen --no-group ocr \
  || die "Dependency installation failed. Check your network connection and retry."

# ── 3 · Ollama runtime + model ────────────────────────────────
MODEL=$(uv run --frozen python -c 'from pageindex.settings import DEFAULT_MODEL; print(DEFAULT_MODEL)') \
  || die "Could not read the configured language model."
if ! command -v ollama >/dev/null 2>&1; then
  die "Ollama is not installed. Download it from https://ollama.com/download, open it once, then double-click this file again."
fi
if ! curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  say "Starting the Ollama server…"
  (OLLAMA_VULKAN=0 GGML_VK_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 \
    ROCR_VISIBLE_DEVICES=-1 \
    ollama serve >/dev/null 2>&1 &)
  for _ in $(seq 1 40); do
    curl -s --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 0.5
  done
  curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 \
    || warn "Ollama did not answer yet — the app will still open; retrieval starts working once it is up."
fi
if ! ollama list 2>/dev/null | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
  say "Downloading the language model ($MODEL, ~7.2 GB — first launch only)…"
  ollama pull "$MODEL" || die "The model download failed. Retry when the connection is stable."
fi

# ── 4 · Launch ────────────────────────────────────────────────
say "Starting the app…"
echo
ASEPSIS_OLLAMA_CPU_ONLY=1 uv run --frozen python run-tauri.py "$@"
status=$?
if [ $status -ne 0 ]; then
  printf 'The app exited with an error (%s). Press Enter to close… ' "$status"
  read -r
fi
exit $status
