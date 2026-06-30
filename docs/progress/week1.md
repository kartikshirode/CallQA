# Week 1 results: dataset

Goal for the week was to pick and build the benchmark data, with real ground truth for every metric the pipeline will report. That is done. Both tiers are built, verified, and registered.

## What got built

The dataset splits into two tiers, each owning the metrics it can honestly support.

- Synthetic core, 20 calls. LLM-style scripts rendered through Kokoro TTS, then assembled on a timeline so every event label falls out of the layout. Owns the event metrics (silence, interruption, compliance, escalation) and the summary references.
- Real anchor, 40 calls. A HarperValley Bank subset, real simulated banking calls with human transcripts and channel-separated speakers. Owns WER, CER, and diarization.

Everything is registered in `data/registry/registry.jsonl`, 60 records total. The full design and the honest limitations are written up in `docs/DATASET_CARD.md`.

## Numbers that matter

- Registry: 60 calls (20 synthetic, 40 HarperValley). Total audio about 54.6 minutes.
- Tests: 108 passing.
- WER spot check (faster-whisper base.en): synthetic 0 to 4%, HarperValley 21 to 35%. The gap is the point. Synthetic is clean so real WER credibility comes from HarperValley.
- Diarization gate: GO. Synthetic calls split cleanly into two speakers, DER around 0.27. HarperValley DER ran 0.12 to 0.56, the spread you expect from real audio.
- Negative-class balance: silence, interruption, escalation all at 40%. Compliance sits at 20% per phrase for the MVP, accepted and documented.

## Decisions made this week

- Measurement coverage was the priority, so the synthetic core carries full event ground truth and the real anchor carries WER and DER.
- Local Kokoro TTS over a paid API. Zero cost, reproducible, runs on the 4060.
- Labels by construction from the timeline, not by hand annotation. This is the core idea that makes the event labels defensible.
- HarperValley over meeting corpora like AMI. It is actual contact-center audio, and the two-channel format gives speaker gold for free. AMI deferred past the MVP.
- Telephone-band degradation kept for diarization and domain realism, even though it does not make synthetic WER hard. The spec was corrected to say this plainly.
- Compliance negatives left at 20% per phrase for the MVP rather than forcing 40%, since real miss rates are low and 4 negatives per phrase is enough for an MVP F1.

## Issues hit and how they were handled

- numpy got pulled to 2.5 by Kokoro, which breaks numba and librosa. Pinned to under 2.4.
- pyannote.audio 4.x pulls a CPU build of torch and kills CUDA. Pinned to 3.3.2 with the cu124 torch 2.6 build. Documented in requirements.txt.
- torch 2.6 defaults torch.load to weights_only=True, which rejects the pyannote checkpoint. Patched load to allow the trusted file. Lives in the pyannote adapter.
- HarperValley transcripts carry bracketed non-speech markers like noise. The Week 2 WER normalizer has to strip these or WER will read high for the wrong reason.

## Where the work lives

- Registry and schema: `callqa/registry/`
- Synthetic generation: `callqa/synth/` (scripts, banks, TTS, assembler)
- Telephone degradation: `callqa/audio/telephone.py`
- HarperValley loader: `callqa/datasets/harpervalley.py`
- Consistency checks: `callqa/datasets/verify.py`
- Diarization: `callqa/diarization/`
- Scripts: `scripts/` (env_check, gen_scripts, build_calls, degrade_calls, fetch_harpervalley, register_synthetic, verify_dataset, dataset_stats, diarization_gate)
