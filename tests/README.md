# Tests

These are the predefined tests. They are grouped by what each project stage
delivers, not one file per source module, so the suite reads as "does this
stage hold up" rather than a pile of tiny unit files.

## Week 1: the dataset

| File | What it proves works |
|------|----------------------|
| `test_registry.py` | The data contract: pydantic schema plus the JSONL store and its privacy guard. |
| `test_synthetic.py` | The synthetic tier end to end: script schema, banks, generator, batch summary, timeline assembler. |
| `test_real_anchor.py` | The HarperValley loader (channel mix, transcript parse, seeded sampling) and the diarization DER math. |
| `test_telephone.py` | Telephone-band degradation. |
| `test_verify.py` | The consistency checks run before any metric is published. |

Each file uses classes to keep related behaviors together while still letting a
failure point at one thing. Later weeks add their own stage files here (events,
summary), they do not get folded back into Week 1.

## Week 2: ASR and WER

| File | What it proves works |
|------|----------------------|
| `test_asr.py` | The ASR core: the WER normalizer (marker stripping, joined contractions), the WER and CER math with its empty-reference guard, the pooled corpus score, the transcript JSON round-trip, and the provider stubs raising cleanly. All offline, no GPU. |

The Whisper adapter's live decode path is not in the suite. It needs a GPU, so it
is checked when `scripts/asr_benchmark.py` runs, not in pytest.

## Scratch tests

Tests written just to confirm a change during development are not committed.
They live in the scratch dir while in use and get deleted once the suite is
green. Only the predefined stage tests above stay in the repo.
