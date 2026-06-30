# Repo map: what lives where

Read this before auditing or running anything. It is the structural orientation
so you do not have to re-derive the layout. For environment, secrets, and the
week roadmap see `HANDOFF.md`. For the data design see `DATASET_CARD.md`.

## Top level

```
callqa/        the importable package, all real logic
scripts/       one-shot CLI scripts that build/verify the dataset
tests/         the predefined suite, grouped by stage (see tests/README.md)
data/          the dataset on disk (mostly gitignored, see below)
docs/          this map, the handoff, the dataset card, progress notes
PLAN.md        the 8-week roadmap
requirements.txt, pyproject.toml   deps and packaging
.env           HF_TOKEN, gitignored, never committed
```

## callqa/ package, by subpackage

Each module notes whether it is pure (no GPU, network, or token, so it is
cheap to test and safe to import) or heavy.

### registry/  the data contract everything reads
- `schema.py` (pure) - pydantic v2 models: `CallRecord`, `SpeakerSegment`,
  `EventLabel`, plus the `Domain`, `EventType`, `ComplianceSubtype`, `Polarity`
  enums. Validators enforce end >= start and non-negative times.
- `store.py` (pure) - `Registry`, the JSONL-backed store. Loads on init, `add`
  enforces a privacy guard (refuses empty source or private/unknown notes) and
  rejects duplicate ids, `_save` writes atomically (temp + os.replace).

### synth/  the synthetic tier, the source of event ground truth
- `script_schema.py` (pure) - `CallScript`, `ScriptTurn`, `EventTag`. A script
  is turns plus per-turn event tags and pauses.
- `banks.py` (pure) - the four domain utterance banks (`BANKS`,
  `REQUIRED_SLOTS`). Was a `banks/` package, now one module.
- `generator.py` (pure) - seeded script generation. `generate_call`,
  `generate_batch` (20 calls, 4 domains x 5), `batch_event_summary`. Decides the
  negative-class balance.
- `assembler.py` (pure) - the important one. Lays per-turn clips on a timeline
  and emits the waveform plus exact labels. Silence and interruption come from
  the layout, compliance and escalation inherit the owning turn's timestamps.
  Labels by construction, so a bug here is undetectable downstream.
- `tts.py`, `voices.py` (heavy, Kokoro) - render turns to audio. Only the build
  scripts touch these.

### audio/
- `telephone.py` (pure-ish, scipy) - `degrade_to_telephone`, the 8 kHz
  band-limit plus codec-style degradation. Seeded, duration preserved.

### datasets/
- `harpervalley.py` (mixed) - the real anchor. Pure helpers (`mix_channels`,
  `gold_from_transcript`, `sample_call_ids`) are tested offline; the network
  helpers (`fetch_call`, `list_call_ids`) hit GitHub. Parser skips malformed
  segments rather than crashing a fetch.
- `verify.py` (pure) - consistency checks run before publishing a metric:
  bounds, transcript present, segment order, synthetic event story, and a
  sidecar-vs-row cross-check.

### diarization/
- `metrics.py` (pure, pyannote.core) - `diarization_error_rate`, `to_annotation`,
  `speaker_count`. The DER math, no pipeline.
- `pyannote_adapter.py` (heavy, GPU + token) - loads the pyannote pipeline. The
  weights_only override for torch 2.6 is scoped to a context manager around the
  checkpoint load, not a global patch.

## scripts/  build order

```
env_check.py            print installed versions and CUDA state
gen_scripts.py          20 synthetic scripts -> data/synthetic/scripts/
build_calls.py          synthesize + assemble clean wavs + label sidecars
degrade_calls.py        telephone-band 8 kHz versions
register_synthetic.py   register the synthetic tier into the registry
fetch_harpervalley.py   download + mix the 40 real calls, register them
verify_dataset.py       consistency checks, exits nonzero on failure
dataset_stats.py        per-tier summary and negative-class ratios
diarization_gate.py     runs the pyannote pipeline, reports DER (needs GPU+token)
```

Everything except `fetch_harpervalley` and `diarization_gate` is offline and
deterministic from a seed.

## tests/  5 stage files, 77 tests

Grouped by what Week 1 delivers, not one file per module. `test_registry`,
`test_synthetic`, `test_real_anchor`, `test_telephone`, `test_verify`. Run with
`python -m pytest -q`, about 1.6s, no GPU or token needed. See `tests/README.md`
for the predefined-vs-scratch rule.

## data/  what is tracked

Tracked in git: the registry, the synthetic scripts and label sidecars, the
HarperValley subset manifest. Gitignored (refetchable or regenerable): all wavs,
`data/harpervalley/raw/` (160 files in 40 dirs, the raw download), and the mixed
HarperValley audio. So the repo stays small; the audio rebuilds from scripts.

```
data/registry/registry.jsonl              60 records, the source of truth
data/synthetic/scripts/<call_id>.json     generated scripts (tracked)
data/synthetic/labels/<call_id>.json      event-label sidecars (tracked)
data/synthetic/audio/clean|telephone/     wavs (gitignored)
data/harpervalley/audio/                  mixed mono wavs (gitignored)
data/harpervalley/raw/<sid>/              raw 4-file downloads (gitignored)
```

## Where to look first in an audit

The risk concentrates in a few pure modules that produce the numbers:
1. `synth/assembler.py` - labels by construction, any timeline math error is silent.
2. `registry/store.py` - the file every metric trusts.
3. `datasets/verify.py` - the safety net; check it is not passing vacuously.
4. `diarization/metrics.py` - DER feeds the gate.
5. `synth/generator.py` - sets the negative-class balance the spec claims.

The heavy modules (`tts`, `voices`, `pyannote_adapter`) need GPU or a token, so
they are usually a blind spot in a static pass. Flag them as not-runtime-checked
rather than assuming they work.

A full four-phase audit and its result sit in `docs/progress/week1-audit.md`.
