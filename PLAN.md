# CallQA - Early Build Plan

## Project thesis

CallQA is an evaluation-first system for customer support call intelligence. Given support-call-like audio, it should transcribe speech, identify speakers, detect operational events, score agent behavior, detect compliance issues, and summarize the call. The project should compare model and provider choices using measurable tradeoffs: accuracy, latency, cost, and reliability.

The point is not to make another speech assistant. The point is to show that speech AI in production is a pipeline with failure modes, metrics, and business-facing quality checks.

## Target company relevance

This maps strongly to Observe.AI-style work:

- Speech-to-text evaluation.
- Speaker diarization and turn-level analysis.
- Agent quality scoring.
- Silence, interruption, escalation, and compliance detection.
- Summary generation with measurable quality.
- Cost and latency awareness for production deployment.

The project should make an interviewer think: this person understands how conversation intelligence systems are evaluated beyond a nice transcript demo.

## MVP scope

Build a benchmark pipeline for public or synthetic support-call-like audio.

MVP capabilities:

- Audio ingestion and metadata tracking.
- Transcription through local Whisper.
- Optional adapters for Deepgram and AssemblyAI, gated by API keys and skipped by default.
- Speaker diarization with pyannote.audio or a comparable open diarization pipeline.
- Turn-level transcript alignment.
- Silence detection.
- Interruption and overlap detection.
- Basic escalation detection.
- Compliance phrase detection using a rules-plus-classifier approach.
- Agent scorecard generation.
- Call summary generation.
- Experiment report comparing WER, diarization quality, latency, and cost.

MVP dataset size:

- 30-50 short audio samples for the first benchmark.
- 5-10 manually annotated samples for interruption, silence, compliance, and escalation sanity checks.
- Synthetic support scripts can be used for compliance and escalation labels, but the project must clearly label them as synthetic.

Out of scope for MVP:

- Real customer call ingestion.
- Storing private or personally identifiable call data.
- Fine-tuning large ASR models.
- Building a production call center product.

## Full-scope vision

The full version becomes a conversation intelligence benchmark:

- Multi-provider ASR comparison: local, open-source, and API-based.
- Multi-domain calls: billing, refund, cancellation, technical support, healthcare-style appointment calls, and finance-style policy calls.
- More event labels: objection handling, sentiment shift, hold time, dead air, script adherence, callback promise, and unresolved issue.
- A dashboard for quality teams to inspect transcript, audio timeline, speakers, events, and scorecards.
- Reproducible cost model per audio hour.
- Human review workflow for event-label correction.

## Datasets and tools

Candidate public data sources:

- SPGISpeech for professionally transcribed financial audio: https://datasets.kensho.com/datasets/scribe
- SPGISpeech paper: https://arxiv.org/abs/2104.02014
- QMSum for meeting transcripts and summaries: https://github.com/Yale-LILY/QMSum
- AMI Meeting Corpus for multi-speaker meetings and diarization-style work: https://groups.inf.ed.ac.uk/ami/corpus/
- ICSI Meeting Corpus as an additional meeting-style source if access is practical: https://groups.inf.ed.ac.uk/ami/icsi/
- FLEURS for multilingual speech recognition checks if the project later expands beyond English: https://huggingface.co/datasets/google/fleurs

Candidate tools:

- ASR: OpenAI Whisper local package: https://github.com/openai/whisper
- Diarization: pyannote.audio: https://github.com/pyannote/pyannote-audio
- Optional ASR APIs: Deepgram docs and AssemblyAI docs, only when API keys are available.
- Audio processing: ffmpeg, librosa, soundfile, pydub.
- Alignment and WER: jiwer, whisper-timestamped or WhisperX-style alignment if stable.
- NLP scoring: scikit-learn, sentence-transformers, lightweight local LLM or API model for summaries.
- Dashboard: Streamlit for MVP.

Data schema for MVP:

- `call_id`: stable id.
- `audio_path`: local or dataset path.
- `duration_seconds`: audio duration.
- `domain`: billing, refund, support, meeting-surrogate, synthetic.
- `reference_transcript`: gold transcript when available.
- `speaker_segments`: optional gold speaker turns.
- `event_labels`: optional labels for silence, interruption, escalation, compliance.
- `summary_reference`: optional human summary.
- `source`: dataset or synthetic generation method.
- `privacy_notes`: confirms public, licensed, or synthetic source.

## Architecture

1. Audio registry
   - Track audio path, source, duration, domain, license, and annotation availability.
   - Refuse to process files marked as private or unknown source.

2. ASR adapter layer
   - Standardize outputs from Whisper, Deepgram, AssemblyAI, or future providers.
   - Convert provider-specific output into common word and segment objects.
   - Store latency, runtime settings, and estimated cost.

3. Diarization layer
   - Run speaker segmentation.
   - Align transcript segments to speakers.
   - Output speaker turns with timestamps.

4. Conversation event detector
   - Detect silence using audio energy and timestamp gaps.
   - Detect interruptions using overlapping diarization segments or rapid speaker changes.
   - Detect escalation using keyword rules plus classifier/LLM pass.
   - Detect compliance issues using configurable phrases and forbidden/required statements.

5. Scorecard generator
   - Compute agent metrics: greeting, verification, empathy cue, interruption count, hold/silence time, compliance flags, resolution status.
   - Generate a structured scorecard with evidence timestamps.

6. Summary generator
   - Generate call summary, issue, resolution, action items, and risk flags.
   - Include transcript references for important claims.

7. Evaluation and dashboard
   - Compare providers and model configs.
   - Show transcript timeline, speaker turns, event markers, and scorecard.
   - Export metrics tables and per-call reports.

## Evaluation metrics

ASR metrics:

- WER.
- CER for noisy or accented audio.
- Named-entity preservation rate when references include entities.
- Punctuation and casing quality as secondary metrics.

Diarization metrics:

- DER when gold speaker labels exist.
- Speaker count accuracy.
- Turn-boundary error.
- Overlap handling accuracy on annotated samples.

Event metrics:

- Silence detection precision, recall, and F1.
- Interruption detection precision, recall, and F1.
- Compliance issue precision, recall, and F1.
- Escalation detection precision, recall, and F1.

Summary and scorecard metrics:

- ROUGE or BERTScore where reference summaries exist.
- Human rubric on 10-20 examples: factuality, completeness, usefulness, and unsupported claims.
- Scorecard agreement with manual labels.

Systems metrics:

- Latency per audio minute.
- Cost per audio hour.
- Failure rate by provider.
- GPU/CPU runtime for local ASR.
- p50 and p95 processing time.

## Milestones

Week 1: Dataset and evaluation definition

- Select initial public/synthetic audio sources.
- Build audio registry schema.
- Create 5 manually reviewed examples.
- Define scorecard rubric.

Week 2: Whisper transcription baseline

- Implement local Whisper transcription.
- Normalize transcripts.
- Compute WER on samples with references.
- Save transcript artifacts.

Week 3: Diarization and alignment

- Add diarization pipeline.
- Align transcript segments with speakers.
- Produce speaker-attributed transcript.
- Add basic diarization metrics where labels exist.

Week 4: Event detection

- Implement silence detection.
- Implement interruption/overlap detection.
- Add first compliance and escalation rules.
- Manually label a small validation set.

Week 5: Scorecards and summaries

- Generate structured scorecards.
- Generate summaries with evidence timestamps.
- Evaluate summary factuality on a small sample.

Week 6: Provider comparison

- Add optional Deepgram and AssemblyAI adapters.
- Compare WER, latency, and cost when keys are available.
- Keep local Whisper as the default reproducible baseline.

Week 7: Dashboard

- Build Streamlit call timeline view.
- Show transcript, speaker turns, event markers, scorecard, and summary.
- Add provider comparison tables.

Week 8: Final report and portfolio polish

- Write results report.
- Add screenshots.
- Document privacy constraints and dataset limitations.
- Write resume bullets and GitHub summary.

Weeks 9-10, optional:

- Add multilingual call samples.
- Add sentiment and resolution classifiers.
- Add Docker support.
- Add human annotation UI for events.

## Risks and mitigations

- Risk: Real customer calls are private and risky.
  - Mitigation: Use public corpora and synthetic support scripts only.

- Risk: Meeting datasets are not exactly customer support calls.
  - Mitigation: Treat them as speech pipeline surrogates and clearly separate synthetic support-call labels.

- Risk: API provider costs can grow.
  - Mitigation: Make paid providers optional and cache all provider outputs.

- Risk: Diarization quality may be weak on telephone audio.
  - Mitigation: Report diarization failure cases separately and avoid hiding errors behind summaries.

- Risk: Compliance detection can look superficial.
  - Mitigation: Use timestamped evidence and a small manually labeled evaluation set.

## Resume/GitHub positioning

Suggested resume bullet:

- Built CallQA, an evaluation framework for support-call intelligence that benchmarks ASR, diarization, interruption/silence detection, compliance checks, agent scorecards, and summaries across WER, DER, event F1, latency, and cost-per-audio-hour.

GitHub positioning:

- Title: CallQA - Evaluation-first support call intelligence.
- One-line hook: Compare transcription, diarization, event detection, and summaries for call QA pipelines.
- Include a dashboard screenshot with audio timeline, speaker turns, event markers, and scorecard.
- Include a "privacy-first" note explaining that no private customer calls are used.

## First 7 days of work

Day 1:

- Create repository structure and README thesis.
- Choose initial datasets and confirm license/access notes.
- Define audio registry schema.

Day 2:

- Download or register 5-10 public audio examples.
- Add duration and transcript metadata.
- Create one manually reviewed sample transcript.

Day 3:

- Implement local Whisper transcription script.
- Save output in a common JSON format.
- Run first transcription on 3 samples.

Day 4:

- Add WER calculation with jiwer.
- Compare raw vs normalized transcript scoring.
- Write first ASR failure notes.

Day 5:

- Add diarization experiment with pyannote.audio or selected open pipeline.
- Store speaker turns.
- Manually inspect diarization output.

Day 6:

- Add silence detection using timestamp gaps and audio energy.
- Add first interruption heuristic.
- Mark events on one sample timeline.

Day 7:

- Define agent scorecard fields.
- Produce one end-to-end JSON report for a single call.
- Update README with current capability and next-week priorities.

## Definition of done for early plan

This planning folder is complete when this `PLAN.md` exists and can be used to start implementation without deciding the project direction again.

