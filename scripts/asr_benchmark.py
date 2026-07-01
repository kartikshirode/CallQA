"""ASR benchmark sweep and report.

Transcribes every registry call with a set of local Whisper sizes, scores WER
and CER per tier against the gold transcripts, and records latency and a
processing-time cost model. Transcripts are cached to disk so re-runs are free
and only the first sweep touches the GPU.

Whisper checkpoints are not gated, so there is no token to read here. Missing
audio is expected: the wav files are gitignored and may not be on disk, so a
call with no audio is skipped and counted, not treated as a crash. That way a
run without regenerated audio degrades to a coverage note instead of throwing.

This script needs a GPU for any fresh transcription. It does not print or
persist any secret. Run it from the repo root:

    python scripts/asr_benchmark.py
    python scripts/asr_benchmark.py --limit 3 --source harpervalley
    python scripts/asr_benchmark.py --models tiny.en,base.en --no-cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.asr.metrics import (  # noqa: E402
    aggregate_score,
    character_error_rate,
    word_error_rate,
)
from callqa.asr.transcript import Transcript  # noqa: E402
from callqa.asr.whisper_adapter import DEFAULT_SIZES, transcribe  # noqa: E402

REGISTRY_PATH = Path("data/registry/registry.jsonl")
HV_AUDIO_DIR = Path("data/harpervalley/audio")
ASR_CACHE_DIR = Path("data/asr")
REPORT_PATH = Path("docs/asr_benchmark_report.md")

# Local Whisper has no per-call dollar cost. The processing-time-per-audio-hour
# figure is always meaningful; the dollar figure is only a modelling assumption.
# Set this to a cloud-GPU hourly rate to estimate rental cost. Kept at zero here
# because the Week 2 run is local. Dollar-vs-provider comparisons land in Week 6.
ASSUMED_GPU_USD_PER_HOUR = 0.0


def load_registry() -> list[dict]:
    rows = []
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def tier_for(row: dict) -> str:
    """Tier label for a registry row."""
    return "synthetic" if row["source"] == "synthetic-kokoro-v1" else "harpervalley"


def wav_for(row: dict) -> Path:
    """Resolve the audio file for a registry row, same rule as diarization_gate."""
    if row["source"] == "harpervalley":
        sid = row["call_id"][len("hv-"):]
        return HV_AUDIO_DIR / f"{sid}.wav"
    return Path(row["audio_path"])


def cache_path(model_size: str, call_id: str) -> Path:
    """Where the cached transcript for one (model, call) lives on disk.

    The model size string doubles as the directory name. The sweep sizes
    ("tiny.en", "base.en", "small.en") have no slash, so they are already
    filesystem-safe; any slash in a custom size is swapped for an underscore.
    """
    model_dir = model_size.replace("/", "_")
    return ASR_CACHE_DIR / model_dir / f"{call_id}.json"


def transcribe_or_cache(
    row: dict, model_size: str, *, use_cache: bool
) -> tuple[Transcript, bool]:
    """Return the transcript for one call, from cache if possible.

    On a cache hit (and cache enabled) the JSON is loaded and the GPU is
    skipped. On a miss the wav is transcribed and the result is written back.
    Returns (transcript, cache_hit).
    """
    path = cache_path(model_size, row["call_id"])
    if use_cache and path.exists():
        return Transcript.from_json(path.read_text(encoding="utf-8")), True

    wav = wav_for(row)
    transcript = transcribe(wav, model_size, call_id=row["call_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.to_json(), encoding="utf-8")
    return transcript, False


def score_call(row: dict, transcript: Transcript) -> dict:
    """WER and CER for one transcribed call against its gold reference."""
    reference = row.get("reference_transcript") or ""
    hypothesis = transcript.text
    return {
        "call_id": row["call_id"],
        "tier": tier_for(row),
        "reference": reference,
        "hypothesis": hypothesis,
        "wer": word_error_rate(reference, hypothesis, normalize=True),
        "cer": character_error_rate(reference, hypothesis, normalize=True),
        "latency_seconds": transcript.latency_seconds,
        "audio_seconds": transcript.audio_seconds,
        "rtf": transcript.real_time_factor,
    }


def build_tables(results: list[dict]) -> tuple[str, list[dict]]:
    """Build the per-model per-tier results table and the summary rows.

    Rows in `results` carry a "model" key. For each (model, tier) the WER and
    CER are pooled with aggregate_score across that group's calls, and latency
    is aggregated as total processing time over total audio. Returns the
    markdown table and the list of summary dicts for reuse in the cost section.
    """
    header = (
        "| model | tier | calls scored | WER | CER | latency / audio min | avg RTF |\n"
        "|-------|------|--------------|-----|-----|---------------------|---------|"
    )
    lines = [header]
    summaries: list[dict] = []

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        groups.setdefault((r["model"], r["tier"]), []).append(r)

    for (model, tier), rows in sorted(groups.items()):
        pairs = [(r["reference"], r["hypothesis"]) for r in rows]
        score = aggregate_score(pairs, normalize=True)

        total_latency = sum(r["latency_seconds"] for r in rows)
        total_audio = sum(r["audio_seconds"] for r in rows)
        audio_minutes = total_audio / 60.0 if total_audio > 0 else 0.0
        latency_per_min = total_latency / audio_minutes if audio_minutes > 0 else 0.0
        avg_rtf = sum(r["rtf"] for r in rows) / len(rows) if rows else 0.0

        summaries.append(
            {
                "model": model,
                "tier": tier,
                "calls": len(rows),
                "wer": score.wer,
                "cer": score.cer,
                "total_latency": total_latency,
                "total_audio": total_audio,
                "latency_per_min": latency_per_min,
                "avg_rtf": avg_rtf,
            }
        )
        lines.append(
            f"| {model} | {tier} | {len(rows)} | {score.wer:.3f} | "
            f"{score.cer:.3f} | {latency_per_min:.2f}s | {avg_rtf:.2f} |"
        )

    return "\n".join(lines), summaries


def build_cost_table(summaries: list[dict]) -> str:
    """Processing-time and dollar cost per audio hour, per model per tier.

    processing-time-per-audio-hour is total_latency scaled to one hour of
    audio; it is always meaningful. The dollar figure multiplies that by
    ASSUMED_GPU_USD_PER_HOUR and is zero for the local run.
    """
    header = (
        "| model | tier | processing time / audio hour | assumed $/audio hour |\n"
        "|-------|------|------------------------------|----------------------|"
    )
    lines = [header]
    for s in summaries:
        if s["total_audio"] > 0:
            proc_hours_per_audio_hour = s["total_latency"] / s["total_audio"]
        else:
            proc_hours_per_audio_hour = 0.0
        # total_latency / total_audio is processing-seconds per audio-second,
        # which equals processing-hours per audio-hour. Report it as minutes for
        # readability, and the dollar figure from the assumed GPU rate.
        proc_minutes = proc_hours_per_audio_hour * 60.0
        dollars = proc_hours_per_audio_hour * ASSUMED_GPU_USD_PER_HOUR
        lines.append(
            f"| {s['model']} | {s['tier']} | {proc_minutes:.2f} min | "
            f"${dollars:.4f} |"
        )
    return "\n".join(lines)


def write_report(
    table: str,
    cost_table: str,
    covered: int,
    total: int,
    models: list[str],
) -> None:
    """Write the markdown benchmark report to REPORT_PATH."""
    model_list = ", ".join(models)
    body = f"""# ASR benchmark report

This is the Week 2 ASR sweep. It transcribes the registry calls with local
faster-whisper across {model_list}, scores WER and CER per tier against the gold
transcripts, and records latency and a processing-time cost model. Transcripts
are cached to data/asr, so this table can be rebuilt without re-running the GPU.

## Results

{table}

WER and CER are pooled across the calls in each group, not averaged per call, so
a longer call carries more weight. Both sides pass through the normalizer first,
which lowercases, strips punctuation and drops the HarperValley bracketed markers
so real WER is not inflated by markup.

## Cost model

Local Whisper has no per-call dollar cost. The processing-time-per-audio-hour
column is always meaningful; it is the wall-clock time the GPU spends per hour of
audio. The dollar column is an assumption only: it multiplies that by an assumed
GPU hourly rate, currently set to ${ASSUMED_GPU_USD_PER_HOUR:.2f} because this run
is local and free. Set ASSUMED_GPU_USD_PER_HOUR in the script to a cloud-GPU rate
to model rental cost. Real dollar-vs-provider comparisons land in Week 6 when the
Deepgram and AssemblyAI adapters come online.

{cost_table}

## Coverage

Calls with audio on disk: {covered} of {total}. Audio is gitignored and may not
be present, so any call missing its wav is skipped and counted here rather than
crashing the run. A low coverage number means the audio was not regenerated, not
that the models failed.

## Reading the numbers

Synthetic WER stays near zero because Kokoro speech is clean, so it is not a real
ASR stress test. The credible real-world WER comes from HarperValley, the primary
tier, which carries real support audio with overlap and channel noise. That gap
is expected and matches docs/DATASET_CARD.md. What the sweep adds is the accuracy
vs latency tradeoff: a larger model lowers WER but costs more processing time per
audio minute, and now there is a number on both sides of that choice.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(body, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Whisper ASR sweep over the registry and "
        "write a WER/CER benchmark report."
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_SIZES),
        help="Comma-separated Whisper sizes to sweep. Default: "
        + ",".join(DEFAULT_SIZES),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap calls per tier, for a quick smoke run before the full sweep.",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic-kokoro-v1", "harpervalley"],
        default=None,
        help="Only run calls from this source. Default: both tiers.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-transcribe even when a cached transcript exists.",
    )
    parser.add_argument(
        "--report",
        default=str(REPORT_PATH),
        help=f"Report output path. Default: {REPORT_PATH}",
    )
    return parser.parse_args(argv)


def select_rows(rows: list[dict], source: str | None, limit: int | None) -> list[dict]:
    """Filter rows by source and cap per tier, keeping registry order."""
    if source is not None:
        rows = [r for r in rows if r["source"] == source]
    if limit is None:
        return rows
    capped: list[dict] = []
    seen: dict[str, int] = {}
    for r in rows:
        tier = tier_for(r)
        if seen.get(tier, 0) >= limit:
            continue
        seen[tier] = seen.get(tier, 0) + 1
        capped.append(r)
    return capped


def main() -> None:
    global REPORT_PATH
    args = parse_args()
    REPORT_PATH = Path(args.report)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    use_cache = not args.no_cache

    rows = load_registry()
    rows = select_rows(rows, args.source, args.limit)
    total = len(rows)

    # Coverage: how many selected calls actually have their wav on disk. This is
    # source-only (independent of model), so count it once up front.
    present = [r for r in rows if wav_for(r).exists()]
    covered = len(present)
    print(f"Selected {total} calls, {covered} have audio on disk.")
    if covered == 0:
        print("No audio found. Regenerate the wav files before the sweep.")

    results: list[dict] = []
    for model_size in models:
        hits = 0
        fresh = 0
        for row in present:
            transcript, was_hit = transcribe_or_cache(
                row, model_size, use_cache=use_cache
            )
            if was_hit:
                hits += 1
            else:
                fresh += 1
            results.append({"model": model_size, **score_call(row, transcript)})
        print(
            f"{model_size}: scored {hits + fresh} calls "
            f"({hits} cache hits, {fresh} fresh)."
        )

    table, summaries = build_tables(results)
    cost_table = build_cost_table(summaries)

    print()
    print(table)
    print()
    print(cost_table)
    print()
    print(f"Coverage: {covered} of {total} selected calls had audio.")

    write_report(table, cost_table, covered, total, models)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
