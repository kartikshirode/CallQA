# Week 1 audit: result and context

Status: **PASS (GREEN).** Week 1 is complete as described in `week1.md`. The dataset claims all hold, and the five issues the audit turned up were hardening fixes, not defects in the shipped data. All fixed and validated.

## What was audited

The job was to check that Week 1 actually did what its writeup says, then fix and test anything off. Ran a four-phase swarm audit (map, find, fix, validate) plus my own independent verification on top, so the auditor's findings got cross-checked rather than trusted blind.

Risk tier: 3 (research data pipeline, not customer-facing). Severity is scaled to that. The three mediums below would each read as high on a production system.

## Claims checked, all confirmed

- Registry holds 60 records. Confirmed: 60 lines in `data/registry/registry.jsonl`.
- 108 tests passing. Confirmed, before and after fixes.
- 20 synthetic label files structurally sound. Confirmed: segments contiguous, silence events sit in real gaps, the 12 segment overlaps are the 12 intentional interruptions (~0.6s each), no events outside the timeline, no reversed spans.
- Negative-class balance. Confirmed against the live `verify_dataset` run: compliance 20% per phrase, escalation and interruption and silence each 40%. Exactly what the doc says.

The file that got flagged mid-audit, `syn-tech_support-04.json`, checked out fine. The thing that looked odd was a `recording_disclosure` compliance tag with negative polarity on the greeting turn. That is a deliberate negative example, one of only four, not a mislabel.

## Findings and fixes

Five issues. Zero critical, zero high, three medium, two low. None corrupt the 60 records.

1. **Interrupt timeline could walk backward** (`callqa/synth/assembler.py`, medium). If an interrupting clip were ever shorter than the 0.6s overlap, the cursor regressed and the next turn would land inside the previous clip's audio, with a wrong label span. It does not trigger on the current data, every interrupt is 0.6s over a multi-second clip, so this was latent. Since this module is the whole "labels by construction" guarantee, a silent label bug here is the worst kind. Fixed by clamping the cursor so it never moves back and capping the interrupt span at whichever clip ends first.

2. **Registry rewritten in place** (`callqa/registry/store.py`, medium). Every `add` rewrote the full JSONL with no temp-and-rename, so a crash or full disk mid-build could truncate the file everything downstream reads. Fixed with a temp write plus `os.replace`, which is atomic.

3. **torch.load patched globally** (`callqa/diarization/pyannote_adapter.py`, medium). The adapter swapped `torch.load` process-wide at import to force `weights_only=False`, despite the comment saying it was just for the trusted pyannote checkpoint. Anything importing the adapter lost torch's safe-load default for the rest of the run. Fixed by scoping it to a context manager around the checkpoint load and restoring the original after.

4. **HarperValley parser crashed on a bad segment** (`callqa/datasets/harpervalley.py`, low). One segment missing a timing or role key would KeyError and kill the whole fetch. Fixed to skip malformed segments instead.

5. **Sidecar check did nothing** (`callqa/datasets/verify.py`, low). `verify_record` took a `sidecar` argument and the docstring implied a cross-check, but it was never read, and a couple of constants were dead. Fixed by removing the dead code and writing a real comparison: duration, segment and event counts, and transcript. It now runs against all 60 records and passes, so the verification is honest.

Two suspected items from the auditor. One needs the GPU pipeline to confirm and is unlikely with real float timestamps, left as a note. The other was a documentation question about the compliance ratio that the verify run resolved on its own, no change needed.

## Validation

- `pytest`: 108 passed, both before and after the fixes.
- `verify_dataset.py`: 60 of 60 clean, 0 failed, with the new sidecar cross-check live.
- No regression. The dataset bytes are unchanged; the fixes harden the code around it.

## Where the trail lives

Working artifacts from the four phases are in `.swarm-audit/` (map, findings, fixes, validation). That folder is gitignored. This file is the committed result.
