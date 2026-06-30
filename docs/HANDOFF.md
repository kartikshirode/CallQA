# Handoff: start here next session

This is the context a fresh session needs to pick up CallQA without re-deriving anything. Read this, then `docs/REPO_MAP.md` for the structure, `PLAN.md` for the full roadmap, and `docs/DATASET_CARD.md` for the data.

## Current state (end of Week 1)

Dataset phase is complete and audited GREEN. Both tiers built, verified, registered. 60 calls in `data/registry/registry.jsonl` (20 synthetic, 40 HarperValley). 77 tests passing. Next up is Week 2, the ASR and WER benchmark.

A four-phase swarm audit ran at the end of Week 1. It found five hardening issues, none in the shipped data, all fixed and validated. Full writeup in `docs/progress/week1-audit.md`. The fixes: atomic registry write, scoped torch.load override, an assembler timeline clamp, a defensive HarperValley parser, and a real sidecar cross-check in verify.

## Machine and environment

- Windows 11, Python 3.12, primary shell PowerShell. GPU is an RTX 4060 Laptop, 8GB VRAM, CUDA working.
- Packages install into the global Python, not a venv. Key versions that must hold:
  - torch 2.6.0+cu124 and torchaudio 2.6.0+cu124 (the cu124 build, from the pytorch index, not PyPI)
  - numpy pinned under 2.4 (2.2.6 now). Above that breaks numba and librosa.
  - pyannote.audio 3.3.2. Do not let 4.x install, it pulls a CPU torch and kills CUDA.
  - kokoro 0.9.4, faster-whisper, librosa, jiwer, soundfile, scipy, streamlit.
- `python scripts/env_check.py` prints what is installed and whether CUDA is on.

## Three traps that will waste time if forgotten

1. Installing kokoro or pyannote can drag numpy or torch to a bad version. After any install, run `python -c "import torch; print(torch.cuda.is_available())"`. If False, reinstall the cu124 torch: `pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124`.
2. pyannote on torch 2.6 needs weights_only forced False to load the checkpoint. That override lives in `callqa/diarization/pyannote_adapter.py`, scoped to a context manager around the load (not a global patch). Reuse that module, do not load pyannote raw.
3. HarperValley human transcripts contain bracketed non-speech tokens like noise and unk. The WER normalizer must strip them or WER reads high for the wrong reason.

## Secrets

- The HuggingFace token sits in `.env` at repo root as `HF_TOKEN=...`. The file is gitignored and never committed. `callqa/diarization/pyannote_adapter.load_token()` reads it.
- Only pyannote needs it. The token can be rotated freely, nothing in the repo stores its value.

## Rebuilding the data from scratch

Audio is not in git, only scripts, labels, and the registry. To regenerate:

```
python scripts/gen_scripts.py            # 20 synthetic scripts
python scripts/build_calls.py            # synthesize + assemble clean wavs
python scripts/degrade_calls.py          # telephone-band 8kHz versions
python scripts/register_synthetic.py     # register synthetic tier
python scripts/fetch_harpervalley.py --limit 40 --seed 1234   # real anchor
python scripts/verify_dataset.py         # consistency checks, exits nonzero on fail
python scripts/dataset_stats.py          # per-tier summary
```

The synthetic build is deterministic from the seed, so it reproduces byte for byte. HarperValley refetches the same 40 calls from the seed manifest.

## Using the data

- Load records with `from callqa.registry.store import Registry; reg = Registry()`. Filter by source ("synthetic-kokoro-v1" or "harpervalley") or domain.
- Each record has audio_path, reference_transcript, speaker_segments, and for synthetic the event_labels. Synthetic call_ids look like `syn-billing-01`, HarperValley like `hv-<sid>`.
- Synthetic event labels also sit in `data/synthetic/labels/<call_id>.json`.

## What each upcoming week needs

- Week 2, ASR and WER. faster-whisper is installed and works on GPU. Build the WER harness on top of jiwer, with a normalizer that lowercases, strips punctuation, and removes the bracketed HarperValley markers. Score both tiers, per call and aggregate. Add CER. Track latency per audio minute and a cost estimate. Keep local Whisper as the default baseline. Deepgram and AssemblyAI adapters come later and stay optional behind keys.
- Week 3, diarization. The adapter and DER metric already exist in `callqa/diarization/`. Run diarization across the registry, score DER and speaker-count accuracy against the gold speaker_segments, and align transcript segments to speakers. HarperValley is the primary DER tier.
- Week 4, event detection. Build silence (energy plus timestamp gaps), interruption (overlap), escalation (keywords plus a classifier or LLM pass), and compliance (required and forbidden phrases). Score precision, recall, and F1 against the synthetic event_labels. Watch the compliance negatives, they are 20% per phrase.
- Week 5, scorecards and summaries. Structured agent scorecard with evidence timestamps, plus call summaries. Evaluate summary factuality on a small sample.
- Week 6, provider comparison. Optional Deepgram and AssemblyAI adapters, gated by keys, all outputs cached.
- Week 7, dashboard. Streamlit timeline view: transcript, speaker turns, event markers, scorecard, provider tables. Streamlit is installed.
- Week 8, final report and polish. Results writeup, screenshots, resume bullets.

## Conventions in this repo

- Coding is done by one background agent per phase, fresh context, with the orchestrator verifying and committing.
- Commits are split into small logical chunks, no AI or co-author trailers.
- No em dashes or en dashes in any file. The dataset tests even assert the banks stay clean.
- Verify before claiming done: run pytest and the actual scripts, do not assume.
- Tests are grouped by stage, not one file per module. Each week adds its own stage file under `tests/`. Throwaway tests written to check a change stay in the scratch dir and are deleted once green, they do not get committed. See `tests/README.md`.
- New structural orientation lives in `docs/REPO_MAP.md`. Update it when modules move.
