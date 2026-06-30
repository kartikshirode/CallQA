# CallQA Dataset Design

Date: 2026-06-30
Scope: dataset selection and generation strategy for the CallQA MVP. This covers what data we benchmark on and how every metric in PLAN.md gets honest ground truth. It does not cover the ASR/diarization pipeline code, which comes after.

## Goal

Pick and build a dataset that lets the evaluation pipeline compute every metric the plan promises (WER, CER, DER, silence/interruption/compliance/escalation F1, summary quality) with trustworthy ground truth. Priority is measurement coverage: every detector needs real labels, not just clean transcripts.

## Core decision

Measurement coverage was chosen as the top priority over raw domain realism. Real support-call corpora almost never ship compliance, escalation, or interruption labels, so we generate a synthetic core where we control the script and therefore own every label. We pair it with a small real anchor so the transcription and diarization numbers stay honest.

## Three-tier dataset

| Tier | Source | Owns these metrics | MVP size |
|------|--------|-------------------|----------|
| A. Synthetic core | Kokoro TTS + LLM scripts, self-assembled timelines | silence, interruption, compliance, escalation F1; summary references | 20 calls (4 domains x 5) |
| B. Real anchor | HarperValley Bank subset | WER, CER, DER, speaker-count accuracy on real support audio | 30-50 calls |
| C. Stretch (deferred) | AMI headset-mix subset | hard overlapping-speech DER | not in MVP |

AMI stays out of the MVP. HarperValley already gives real WER and DER, and AMI's download size plus meeting-domain mismatch is not worth the scope cost yet.

## Why local TTS (Kokoro)

Kokoro-82M, Apache-2.0, runs fine on the target machine (RTX 4060 Laptop, 8GB VRAM, 24GB RAM, Python 3.12). No API keys, no per-call cost, fully reproducible. Anyone can clone the repo and regenerate the whole dataset and benchmark offline. That reproducibility is a portfolio point in itself.

The usual objection to local TTS is weak timing control. It does not apply here because of the assembly method below: labels come from how we lay out the timeline, not from the engine. So a free local engine is as frame-accurate as a paid SSML service.

## Synthetic generation method

Labels by construction, not by annotation. The flow:

1. An LLM writes a structured script: ordered turns, each tagged with `speaker` (agent or customer), `text`, and event intent fields such as `compliance:recording_disclosure`, `escalation:trigger`, `interrupt_prev`, `pause_before:2.4s`.
2. Kokoro synthesizes each turn as a separate clip, with a distinct voice per speaker role.
3. An assembler places clips on a single timeline. Deliberate gaps become labeled silence. Deliberate overlaps become labeled interruptions. Start and end times come straight from the layout.
4. Compliance and escalation labels come from the script tags, anchored to the timestamp of the owning turn.

Result: every event label is generated mechanically, with exact timestamps, before any audio is even rendered.

## Telephone-band realism (do not skip)

After assembly, each synthetic call passes through telephone-band degradation: downsample to 8kHz, mu-law companding, light line and background noise, occasional codec artifacts. Without this step Whisper would score near-perfect WER and diarization would be trivial, making both numbers meaningless. This step is what keeps the synthetic ASR and DER results worth reporting.

## Negative-class policy

Each event detector needs real negatives or its F1 is misleading. The script template forces roughly 40% negative cases per event type:

- Some calls omit a required compliance phrase (recording disclosure, identity verification, closing).
- Some calls carry no escalation trigger at all.
- Some calls include clean stretches with no interruption.

This is deliberate in the template, not left to chance, so we do not accidentally script mostly positives.

## Voice diversity and the diarization gate

Agent and customer voices must be acoustically distinct enough that pyannote faces a real task, but not so far apart that diarization succeeds trivially. This is a go/no-go checkpoint:

- Generate the first 3 synthetic calls.
- Run diarization on them.
- If DER is near zero (too easy) or the system collapses speakers (too hard), retune voice spacing before generating the remaining calls.

Only after this gate passes do we commit to the full 20.

## Verification policy

Hand-verify all 20 synthetic calls, not a sample. It is only 20, and the auto-label sanity check has to be airtight before any WER or DER number is published. Reviewers will ask how the labels were validated. For HarperValley, spot-check 5 calls to confirm the transcript and speaker-turn alignment we rely on.

## Domains (synthetic)

Billing dispute, refund request, cancellation and retention, technical support. Five calls each. These are standard contact-center scenarios and map onto the `domain` field in the plan schema.

## Schema mapping

Rows follow the PLAN.md MVP schema. Synthetic rows carry full `event_labels`, `speaker_segments`, and `summary_reference`. HarperValley rows carry `reference_transcript` and `speaker_segments`. The `source` and `privacy_notes` fields mark every row as synthetic or licensed-public, so the two tiers never get confused in reporting.

## Licensing and privacy

- Kokoro: Apache-2.0, safe to ship generated audio.
- HarperValley Bank: public research corpus of simulated calls, no real customer data. Record its license note in `privacy_notes`.
- No real customer calls, no PII, ever. Synthetic rows are labeled synthetic in every report so nobody mistakes generated audio for real performance.

## Definition of done for the dataset phase

- 20 synthetic calls generated, telephone-band processed, fully labeled, all 20 hand-verified.
- Diarization gate passed and recorded.
- HarperValley subset of 30-50 calls downloaded, registered, 5 spot-checked.
- A single registry file lists every call with its schema fields filled in.
- Negative-class ratio confirmed at roughly 40% per event type.

## Deferred / open items

- AMI subset for hard overlap DER: revisit after the MVP benchmark runs.
- Larger synthetic batch (toward 40-50): generate once the pipeline is proven, in one batch.
- Multilingual samples: out of scope until the English pipeline is stable.
