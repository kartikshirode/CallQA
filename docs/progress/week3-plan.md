# Week 3 plan: diarization and alignment

Week 2 gave every call a transcript. Week 3 answers who spoke when, scores it
against the gold speaker turns, and stitches the two together so each line of
transcript carries a speaker. HarperValley is the primary tier here, same as the
dataset card says, because its speakers are grounded in the channel-separated
recording, not guessed.

## What already exists

Week 1 left most of the diarization machinery in place, so this week is less
building from scratch and more wiring it across the registry.

- `callqa/diarization/metrics.py` (pure): `diarization_error_rate`,
  `speaker_count`, `to_annotation`. The DER math, already unit-tested.
- `callqa/diarization/pyannote_adapter.py` (heavy): `diarize(wav)` returns
  SpeakerSegment turns on cuda, with the scoped weights_only override.
- `scripts/diarization_gate.py`: runs pyannote on a handful of calls as the Week 1
  voice-diversity gate. Week 3 generalizes this to the whole registry.

What is missing: alignment (transcript segments to speaker turns), a
speaker-attributed transcript object, a full-registry diarization run with
caching, and speaker-count accuracy reporting.

## What Week 3 adds

Mirrors the Week 2 build so the two read the same way.

### Phase 1, pure core (no GPU)

- `callqa/diarization/align.py` - assign each Week 2 transcript segment to a
  speaker by maximum time overlap with the diarized turns. Emit a
  speaker-attributed transcript. Pure, so it tests on hand-made segment lists.
- A speaker-attributed transcript object (extend the Week 2 Transcript or add a
  small wrapper that carries a speaker per segment). Decide which at kickoff.
- Speaker-count accuracy helper alongside the existing DER, if not already
  covered by `speaker_count`.
- `tests/test_diarization.py` - the Week 3 stage file. Alignment edge cases
  (overlap ties, a segment spanning two speakers, gaps), DER wiring, speaker-count
  accuracy. No GPU or token.

### Phase 2, heavy runner

- A cached diarization runner: run `diarize` across the registry once and cache
  each call's speaker turns to disk (same idea as the ASR transcript cache), so
  re-scoring never re-hits the GPU.

### Phase 3, benchmark and report (gated GPU run)

- `scripts/diarization_benchmark.py` - run diarization across all 60 calls, score
  DER and speaker-count accuracy per tier against the gold speaker_segments, align
  the Week 2 transcripts, and write `docs/diarization_benchmark_report.md`. HV is
  the headline DER tier; synthetic is supporting.
- Update `docs/REPO_MAP.md`, `docs/HANDOFF.md`, `tests/README.md`.

## Open decisions for kickoff

These change the build, so confirm them before Phase 1 rather than guessing.

- Alignment rule: max-overlap per segment is the simple default. Word-level
  alignment is more precise but needs word timestamps and more care. Recommend
  starting with max-overlap and noting the limitation.
- Which ASR transcript feeds alignment: the base.en sweep output is the natural
  default. A speaker-attributed transcript is only as good as the transcript under
  it, so tie it to one model size and say which.
- Diarization cache format: JSON speaker turns keyed by call, matching the ASR
  cache layout, versus RTTM. JSON keeps it consistent with the rest of the repo.
- Whether synthetic diarization is worth scoring beyond the Week 1 gate. It is a
  clean 2-speaker split, so DER there is informational; HV carries the real
  number.

## Guardrails carried forward

- pyannote stays on 3.x with the cu124 torch 2.6 build. After any install, check
  cuda is still True.
- The pyannote adapter needs the HF token from the gitignored .env. Only pyannote
  uses it.
- No em or en dashes in any file.
- Verify before claiming done. Run pytest and the actual script.
- The heavy diarize path is runtime-checked only when the GPU run happens, so keep
  the pure alignment and metric logic separately testable.

## Why this matters

A transcript on its own does not tell a QA team who talked over whom or how long
the agent held the floor. Speaker-attributed turns are what the Week 4 event
detectors (interruption, silence) and the Week 5 scorecard read from. Week 3 is
the layer that turns raw text into a conversation.
