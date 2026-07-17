# Runtime model manifest

ASEPSIS runs local models only. Python packages are fixed by `uv.lock`; model
weights have their own lifecycle and must not be confused with package locking.

| Purpose | Runtime identity | Processor policy | Weight integrity |
|---|---|---|---|
| Retrieval, synthesis, optional legacy asset captioning | `gemma4:e2b-it-q4_K_M` through Ollama | CPU requested by default; `/api/config` reports loaded-model VRAM usage | Ollama's local manifest; not pinned in this repository |
| PDF layout | `PP-DocLayoutV3` through PaddleX | CPU path | Downloaded and cached by Paddle; not digest-pinned here |
| German/Latin heading OCR | `latin_PP-OCRv5_mobile_rec` through PaddleOCR | `device="cpu"`, `lang="latin"` | Downloaded and cached by Paddle; not digest-pinned here |
| Confirmed table structure | `TableRecognitionPipelineV2` through PaddleOCR | `device="cpu"`; unused orientation, unwarping, and layout stages disabled | Downloaded and cached by Paddle; not digest-pinned here |

`ASEPSIS_OLLAMA_CPU_ONLY=1` is the default for processes started by ASEPSIS. It
sets Vulkan, CUDA, and ROCm visibility off. A separately started Ollama process
is not controlled by ASEPSIS; check `/api/config` after a model is loaded. A
model with non-zero `size_vram` is reported as not CPU-verified.

Before an offline or regulated handoff, prefetch the exact required Paddle and
Ollama weights, record their local manifests and file digests in the deployment
record, and prevent runtime downloads. The current local prototype does not
claim that stronger supply-chain guarantee.
