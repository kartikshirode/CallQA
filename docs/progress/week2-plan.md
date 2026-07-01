# Week 2 plan: ASR and WER benchmark

Week 1 shipped the dataset, GREEN and audited. Week 2 turns it into the first
real metric: transcribe both tiers with local Whisper, score WER and CER against
the gold transcripts, and report latency and cost. The dataset was built so this
week has honest ground truth to score against.

## What we're building

Local faster-whisper transcription over all 60 registry calls, scored per call
and per tier, with a normalizer that does not inflate error on HarperValley
markup. Three model sizes so the report shows an accuracy vs latency tradeoff,
not a single number. All outputs cached to disk so re-runs are free.

## Decisions locked with the orchestrator

- Model coverage: full sweep, tiny.en / base.en / small.en. The 4060 handles all
  three over 55 minutes of audio in well under an hour, run once and cached. The
  tradeoff curve is the point for the target-company story.
- Metrics: WER, CER, latency per audio minute, cost per audio hour. Per call and
  aggregate per tier.
- Cloud providers: stub the adapter interface now so Deepgram and AssemblyAI slot
  in during Week 6, but do not implement or call them. No dead network code.
- The actual GPU benchmark run is gated. The orchestrator asks before starting it.

## Build order

Follows the repo convention: one background agent per phase, fresh context, the
orchestrator verifies and commits. Pure code first so most of the logic is tested
with no GPU or token.

### Phase 1, pure core (no GPU)

- `callqa/asr/normalize.py` - the WER normalizer. Lowercase, strip punctuation,
  collapse whitespace, drop the HarperValley bracketed markers like `[noise]` and
  `<unk>`. This module decides whether real WER is honest, so it gets the heaviest
  test coverage.
- `callqa/asr/metrics.py` - WER and CER on top of jiwer, per call and aggregate.
- `callqa/asr/transcript.py` - the common transcript object (text, segments,
  timings, model tag). This is also the interface the cloud stubs target.
- `tests/test_asr.py` - the Week 2 stage file. Normalizer edge cases, WER and CER
  math, cache round-trip. Runs with no GPU or token.

### Phase 2, heavy adapter

- `callqa/asr/whisper_adapter.py` - wraps faster-whisper. Loads a model size on
  cuda once and caches it, same pattern as `pyannote_adapter`. Returns the
  transcript object plus wall-clock latency. Carries the cu124 torch rule.
- `callqa/asr/providers.py` - the stub interface for Deepgram and AssemblyAI. The
  abstract adapter contract only, gated behind keys, raising a clear Week 6
  NotImplementedError. No cloud calls.

### Phase 3, benchmark and report (the gated run)

- `scripts/asr_benchmark.py` - runs the sweep across the registry, caches
  transcripts to `data/asr/<model>/<call_id>.json` (gitignored), computes per-tier
  WER and CER, latency per audio minute, a cost-per-hour estimate, and writes
  `docs/asr_benchmark_report.md`.
- Update `docs/REPO_MAP.md`, `docs/HANDOFF.md`, `tests/README.md` for the new
  stage.

## Runtime estimate on the 4060

faster-whisper with float16 on the RTX 4060, over 54.6 minutes of audio:

| Model | Rough speed | Time for the full registry |
|-------|-------------|----------------------------|
| tiny.en | ~30x realtime | ~2 min |
| base.en | ~15-20x | ~3 min |
| small.en | ~8-10x | ~6 min |

So the full sweep lands near 10-12 minutes including model load and warmup. Well
inside the hour budget, and cached after the first run.

## Guardrails carried from Week 1

- After any install, check `torch.cuda.is_available()` is True. If a dep dragged
  torch off the cu124 build, reinstall it before running anything on the GPU.
- The normalizer must strip HarperValley markers or real WER reads high for the
  wrong reason.
- No em or en dashes in any file. The dataset tests already assert this.
- Verify before claiming done. Run pytest and the actual script, do not assume.

## Why this matters for the project

Synthetic WER stays near zero because Kokoro speech is clean, so the real WER
credibility comes from HarperValley. That gap is expected and documented in the
dataset card. What Week 2 adds is the first measurable model tradeoff: pick a
Whisper size and you trade accuracy for latency, and now there's a number on both
sides of that choice.
