# CallQA Dataset Card

This is the benchmark data behind CallQA, a contact-center call analyzer. The
job of the dataset is simple to state and hard to get right: give every metric
the pipeline reports a real ground truth to score against. Word error rate,
diarization error, and the event detectors (silence, interruption, compliance,
escalation) all need labels that someone can actually defend.

## Two tiers, and why

Real support-call corpora almost never ship compliance, escalation, or
interruption labels. You can buy clean transcripts; you cannot easily buy "this
is the second where the agent skipped the recording disclosure." So the dataset
splits into two tiers, each owning the metrics it can honestly support.

**Synthetic core (20 calls).** Built from LLM-written scripts rendered through
Kokoro TTS, then assembled on a single timeline. Because we control the layout,
every event label comes by construction. A deliberate gap is labeled silence
with exact bounds. A deliberate overlap is a labeled interruption. Compliance
and escalation tags ride along from the script. The timestamps are not
annotated after the fact; they fall out of how the clips were placed. This tier
owns the event-detection metrics and the summary references.

**Real anchor (40 calls).** A subset of the HarperValley Bank corpus, a public
set of simulated banking calls with human transcripts and channel-separated
speaker turns. The audio is real telephone audio with real overlap and real
disfluency, so it is what we trust for WER, CER, and diarization error. It does
not carry event labels, and we do not pretend it does.

The split keeps the two honest. Event scores come from the tier where labels are
exact. Transcription and diarization scores come from the tier where the audio
is real.

## Sizes and durations

Numbers below are pulled straight from `data/registry/registry.jsonl`, not
estimated.

| Tier | Source | Calls | Total duration | Avg duration | Domains |
|------|--------|------:|---------------:|-------------:|---------|
| Synthetic core | synthetic-kokoro-v1 | 20 | 930.0 s (15.5 min) | 46.50 s | billing, refund, cancellation, tech_support (5 each) |
| Real anchor | harpervalley | 40 | 2348.7 s (39.1 min) | 58.72 s | support |
| **Total** | | **60** | **3278.7 s (54.6 min)** | | |

Synthetic event labels across the 20 calls: 40 compliance events (32 positive,
8 negative), 12 escalation, 12 interruption, 12 silence.

## Schema

Every call is one `CallRecord` (see `callqa/registry/schema.py`). The fields:

- `call_id`, `audio_path`, `duration_seconds`, `domain`, `source`,
  `privacy_notes`
- `reference_transcript`: gold transcript text
- `speaker_segments`: list of `{speaker, start, end}` turns
- `event_labels`: list of `{event_type, start, end, metadata}`; compliance
  events carry `subtype` (recording_disclosure, identity_verification, closing)
  and `polarity` (positive or negative) in metadata
- `summary_reference`: reference summary, where present

The registry is JSONL, one record per line, so it stays easy to diff and read
by hand.

## Where the labels come from

**Synthetic.** Labels are generated mechanically before the audio is even
rendered. The script tags the turn, the assembler lays it on the timeline, and
the start and end times come from that layout. Each call also has a label
sidecar at `data/synthetic/labels/<call_id>.json` holding the same segments and
events, which is what the registry row was built from.

**HarperValley.** Transcripts are the human transcripts shipped with the corpus.
Speaker turns come from the channel-separated agent and caller tracks, so the
who-spoke-when is grounded in the recording itself, not guessed.

## Which metrics each tier supports

| Metric | Synthetic core | Real anchor |
|--------|:---:|:---:|
| WER / CER | weak (TTS is too clean) | yes, primary |
| Diarization error (DER) | supporting | yes, primary |
| Speaker-count accuracy | supporting | yes |
| Silence F1 | yes, primary | no labels |
| Interruption F1 | yes, primary | no labels |
| Compliance F1 | yes, primary | no labels |
| Escalation F1 | yes, primary | no labels |
| Summary quality | yes (references) | no |

## Negative-class balance

A detector with no negatives has a meaningless F1. The synthetic template forces
negative cases on purpose. Measured across the 20 calls:

- escalation: 8 of 20 calls have no escalation (40% negative)
- interruption: 8 of 20 calls have no interruption (40% negative)
- silence: 8 of 20 calls have no silence event (40% negative)
- compliance, recording_disclosure: 4 of 20 negative (20%)
- compliance, identity_verification: 4 of 20 negative (20%)

Worth being upfront here. The escalation, interruption, and silence rates land
right on the 40% target from the design spec. The two compliance subtypes sit at
20% each, lower than the 40% the spec aimed for. The pooled compliance figure
still has 8 negatives out of 40 events, but spread across two phrases that is 20%
per phrase. For the MVP that is enough negatives to compute a real compliance F1
on each phrase, but a future batch should push the per-subtype miss rate up
toward 40% so the compliance numbers carry the same weight as the others.

## Limitations

Stating these plainly so nobody reads more into the numbers than is there.

- **Synthetic WER stays near zero.** Kokoro speech is clear, so Whisper scores
  it very low even after telephone-band degradation. The synthetic tier is not
  an ASR stress test. Real WER credibility comes from HarperValley.
- **HarperValley transcripts carry non-speech markers.** They include bracketed
  tokens like `[noise]` and `<unk>`. A WER normalizer has to strip these before
  scoring or the error rate will be inflated by markup that is not speech.
- **HarperValley is single-domain.** It is all banking support. So the real
  WER and DER numbers are anchored in one domain, not the four the synthetic
  tier spans.
- **The synthetic tier is small.** 20 calls is an MVP size, picked so all 20
  could be checked by hand. It is enough to validate the pipeline and compute
  event F1, not enough to claim production-grade event accuracy.
- **Compliance negatives are light** (see the section above).

## Licensing and privacy

- **Synthetic core.** Original work. Kokoro TTS is Apache-2.0, and the generated
  audio is ours to ship. The scripts and assembled audio are regenerable from
  the repo, so anyone can rebuild the whole tier offline.
- **HarperValley Bank.** Public research corpus, CC-BY-4.0, from the
  Gridspace-Stanford project. Attribution is recorded in `NOTICE.md`. These are
  simulated calls, not real customers.
- **No real customer PII anywhere.** Both tiers are either generated or simulated
  public data. Every row marks its source and privacy note in the registry so
  the two tiers never get confused in a report.

## Verifying the data

Two scripts back the numbers in this card:

- `python scripts/verify_dataset.py` runs the consistency checks on all 60
  records (bounds, transcript presence, segment ordering, the synthetic event
  story) and cross-checks wav duration against the registered duration when the
  audio is on disk. It exits nonzero on any failure, so it can gate later work.
- `python scripts/dataset_stats.py` prints the per-tier counts and durations
  used above.

The check functions live in `callqa/datasets/verify.py` and are unit-tested in
`tests/test_verify.py`.
