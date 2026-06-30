# CallQA - Evaluation-first support call intelligence

Compare transcription, diarization, event detection, and summaries for call QA pipelines, scored on metrics that actually matter in production.

## Project status

Dataset phase, Phase 0 scaffold. The audio registry and an environment smoke test are in place. ASR, diarization, event detection, scorecards, and the dashboard come next.

## Dataset strategy

Three tiers. The core is synthetic: short support calls generated locally with Kokoro TTS from LLM-written scripts, assembled on a timeline so every silence, interruption, compliance, and escalation label is exact by construction. The real anchor is a HarperValley Bank subset, which keeps WER and diarization numbers honest on real audio. AMI is deferred past the MVP. Synthetic audio passes through telephone-band degradation first, otherwise the ASR and diarization scores would be meaningless. No real customer calls, no PII, ever.

## Repo layout

```
callqa/
  registry/        audio registry: pydantic schema + JSONL store
data/
  registry/        registry.jsonl lives here
  synthetic/       generated Kokoro calls
  harpervalley/    real-anchor subset
scripts/
  env_check.py     environment smoke test
tests/             pytest suite for the registry
PLAN.md            project thesis and MVP schema
docs/superpowers/specs/2026-06-30-callqa-dataset-design.md   dataset design
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
