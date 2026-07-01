# Weeks 1 and 2 audit: result and context

Status: **PASS (GREEN).** Weeks 1 and 2 do what their writeups claim. Week 1 is a
clean regression against the earlier audit, and the Week 2 ASR and WER code holds
up. Two low-severity hardening fixes went in, both tested, neither touches the 60
shipped records or their bytes.

Risk tier: 3 (research data and eval pipeline, not customer-facing). Severity is
scaled to that. The GPU is not visible in this shell (torch.cuda.is_available()
reads False), so the faster-whisper decode path and the live benchmark run stay
static-only and are flagged runtime-unverified rather than assumed good.

## What was audited

The job was to treat both weeks as one system and check the claims, not trust the
writeups. Week 2 sits directly on the Week 1 dataset contract, so they have to
hold together. Ran the four phases (map, find, fix, validate) with two independent
swarm auditors on top, one on the Week 2 ASR core and one on the Week 1 regression
plus the cross-week seam, then re-derived the risky claims by hand against the live
data.

## Claims checked, all confirmed

Week 1, still GREEN:

- 60 records in the registry, 20 synthetic and 40 HarperValley. Confirmed from a
  live load.
- Labels by construction in the assembler. Re-derived the interrupt math myself
  with a short clip: a clip shorter than the 0.6s overlap lands its span capped at
  the clip end, the cursor never walks backward, and no span comes out reversed.
- Negative-class balance matches the card against a live verify_dataset run:
  compliance 20% per phrase, escalation and interruption and silence each 40%.
- The verify sidecar cross-check is real, not vacuous. Proved it by feeding a
  tampered sidecar and watching it flag the duration and transcript drift, then a
  clean one and watching it pass. It runs on all 20 synthetic rows.
- Registry writes atomically, temp plus os.replace.
- The pyannote weights_only override is scoped to a context manager around the
  load and restored after, not a global patch.

Week 2:

- The normalizer marker set matches the real markup. Grepped every HarperValley
  reference in the registry: the only non-speech tokens are [noise], <unk>,
  [laughter] and [unintelligible]. The regex strips all four whole, brackets and
  all, before punctuation, so nothing leaks as a bare word. Contraction joining
  runs in the one normalize path both sides use, so it cannot favor reference over
  hypothesis.
- WER and CER pool edits across a corpus rather than averaging per-call rates.
  Checked it: a 10-word call with one error plus a 1-word call with one error
  pools to 2/11 = 0.18, not the 0.55 mean. Empty-reference rows drop from the
  pooled lists but still count in n_calls.
- The empty-reference guard is correct, no divide-by-zero on the normalize path.
- Transcript round-trips through JSON, real_time_factor guards zero latency, and a
  reversed segment is rejected.
- The cloud stubs import no HTTP client, make no network call, and raise a clear
  Week 6 note. All three providers satisfy the protocol.
- The benchmark resolves HarperValley wavs the same way the diarization gate does,
  and that path matches the registry audio_path. Missing audio is skipped and
  counted, not crashed. The cost math is dimensionally right: processing minutes
  per audio hour, checked against a fabricated run.

Cross-week:

- Week 2 reads the Week 1 schema read-only and every optional field it touches
  goes through .get with a fallback.
- Synthetic WER runs on the telephone-band audio the registry points to
  (data/synthetic/audio/telephone/...), not the clean version.
- Nothing in Week 2 mutates or regenerates Week 1 data.

## Findings and fixes

Zero critical, zero high, zero medium in the shipped path. Two low-severity
robustness gaps, both fixed with a test that fails before and passes after.

1. **HarperValley parser still crashed on some malformed segments**
   (`callqa/datasets/harpervalley.py`, low). The earlier audit taught the parser
   to skip a segment missing a timing or role key, so one bad row would not kill a
   whole fetch. The guard only checked for None, though. A segment with a
   string-typed start_ms crashed on the addition, and a negative duration built a
   reversed span the schema rejected. Either one aborts the fetch the skip was
   meant to protect. Not triggered by the 40 rows already registered, so it is
   latent, but the guarantee was incomplete. Fixed the filter to require a real,
   non-negative number for both timings before a segment becomes a turn. Added a
   test with a string timing and a negative duration.

2. **RTF column averaged while the rest of its row pooled**
   (`scripts/asr_benchmark.py`, low). In the results table WER, CER and latency
   per audio minute are pooled total over total, but the throughput column took a
   plain mean of the per-call ratios. Mean of ratios leans toward short calls and
   any zero-latency row drags it, so the two throughput columns in one row could
   disagree. On a long-fast plus short-slow pair the mean reads 5.25x while the
   honest pooled figure is 7.29x, and only the pooled one matches the latency
   column beside it. Fixed it to pool audio over latency like everything else in
   the row, renamed the column from "avg RTF" to "RTF", and corrected the report
   caption to say the throughput columns are pooled too. The benchmark report is
   generated by the pending GPU run, so this lands before any numbers ship.

Items looked at and left alone, on purpose:

- The empty-reference guard on the normalize=False path. The auditor flagged a
  possible divide-by-zero on a whitespace-only reference. Tried it: the installed
  jiwer returns 1.0, no crash. The benchmark always normalizes anyway. No change.
- The per-call versus pooled empty-reference behavior. It is documented and no
  reference in the 60 rows is empty or markers-only, so changing it would only
  risk the intended pooling semantics for no gain.
- The whisper_adapter model cache never evicts. A large custom --models list could
  crowd the 8GB card, but the default tiny and base and small.en sweep fits. This
  needs the GPU to judge, so it is a note, deferred.
- Two copies of wav_for, in the benchmark and the diarization gate, both recompute
  the HarperValley path from the call_id instead of reading audio_path. They agree
  today. Centralizing it would touch two working modules for a latent risk, so it
  is recorded, not changed.

## Validation

- pytest: 109 passed before, 111 after (two new regression tests). Green both
  ways.
- verify_dataset.py: 60 of 60 clean, exit 0, with the sidecar cross-check live.
- dataset_stats.py: 60 calls, 40 plus 20, balance figures unchanged.
- Dataset bytes untouched. git status shows only the two code files and two test
  files, no data.
- No GPU here, so no live ASR smoke run. The faster-whisper decode path gets its
  first real check when the Week 2 benchmark sweep runs on the 4060.

## Where the trail lives

The two swarm auditors ran read-only and returned their findings inline; nothing
was written to disk by them. This file is the committed result. The remaining
Week 2 step is unchanged: run `python scripts/asr_benchmark.py` on the GPU box to
fill in docs/asr_benchmark_report.md.
