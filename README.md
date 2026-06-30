# CallQA - Evaluation-first support call intelligence

Compare transcription, diarization, event detection, and summaries for call QA pipelines, scored on metrics that actually matter in production.

## Project status

Dataset phase complete. Both tiers are built, verified, and registered: 20 synthetic calls with full event ground truth and 40 real HarperValley calls for WER and diarization. See `docs/DATASET_CARD.md`. Still open: the diarization voice-diversity gate (needs a HuggingFace token for pyannote). ASR scoring, event detection, scorecards, and the dashboard come next.

## Dataset strategy

Three tiers. The core is synthetic: short support calls generated locally with Kokoro TTS from LLM-written scripts, assembled on a timeline so every silence, interruption, compliance, and escalation label is exact by construction. The real anchor is a HarperValley Bank subset, which keeps WER and diarization numbers honest on real audio. AMI is deferred past the MVP. Synthetic audio passes through telephone-band degradation first, otherwise the ASR and diarization scores would be meaningless. No real customer calls, no PII, ever.

## Repo layout

```
callqa/
  registry/        audio registry: pydantic schema + JSONL store
  synth/           synthetic calls: script generator, banks, TTS, assembler
  audio/           telephone-band degradation
  datasets/        HarperValley loader and consistency checks
data/
  registry/        registry.jsonl, the dataset index
  synthetic/       scripts, labels, and generated audio
  harpervalley/    subset manifest (raw audio is refetched, not versioned)
scripts/           env check, build, degrade, fetch, verify, stats
tests/             pytest suite
docs/
  DATASET_CARD.md  what the data is, where labels come from, limits
  superpowers/specs/2026-06-30-callqa-dataset-design.md   dataset design
PLAN.md            project thesis and MVP schema
```

## Getting started

```
pip install -r requirements.txt
python scripts/env_check.py
pytest
```

The heavy GPU packages (torch, pyannote.audio, kokoro, faster-whisper) are listed in `requirements.txt` but not needed for Phase 0. Install them before the synthesis and ASR phases, and match the torch wheel to your CUDA version.

## More detail

See PLAN.md for the full thesis, metrics, and milestones. See the dataset design spec for how every label gets trustworthy ground truth.
