"""Diarization gate and voice-diversity check.

Runs pyannote on the first 3 synthetic calls (the gate's real purpose) and on
2 HarperValley calls (to show real DER on the primary tier). For each call it
prints the predicted speaker count, the gold speaker count and the DER against
the gold speaker_segments, then prints a go/no-go verdict and writes a short
report.

This script needs the HF token and a GPU. It does not print or persist the
token. Run it from the repo root:

    python scripts/diarization_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.diarization.metrics import diarization_error_rate, speaker_count  # noqa: E402
from callqa.diarization.pyannote_adapter import diarize  # noqa: E402

REGISTRY_PATH = Path("data/registry/registry.jsonl")
HV_AUDIO_DIR = Path("data/harpervalley/audio")
REPORT_PATH = Path("docs/diarization_gate_report.md")

# DER thresholds for the synthetic voice-diversity gate.
TOO_EASY_DER = 0.05
SANE_HIGH_DER = 0.6


def load_registry() -> list[dict]:
    rows = []
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def wav_for(row: dict) -> Path:
    """Resolve the audio file for a registry row."""
    if row["source"] == "harpervalley":
        sid = row["call_id"][len("hv-"):]
        return HV_AUDIO_DIR / f"{sid}.wav"
    return Path(row["audio_path"])


def evaluate(row: dict) -> dict:
    """Run diarization on one call and compute its metrics."""
    wav = wav_for(row)
    gold = row.get("speaker_segments") or []
    predicted = diarize(wav)
    return {
        "call_id": row["call_id"],
        "tier": "synthetic" if row["source"] == "synthetic-kokoro-v1" else "harpervalley",
        "gold_speakers": speaker_count(gold),
        "pred_speakers": speaker_count(predicted),
        "der": diarization_error_rate(gold, predicted),
    }


def synthetic_verdict(r: dict) -> str:
    """Voice-diversity verdict for one synthetic call."""
    if r["pred_speakers"] != 2:
        return "WARN collapse"
    if r["der"] < TOO_EASY_DER:
        return "WARN too easy"
    if r["der"] <= SANE_HIGH_DER:
        return "PASS"
    return "WARN too hard"


def build_table(results: list[dict]) -> str:
    header = (
        "| call_id | tier | gold spk | pred spk | DER | verdict |\n"
        "|---------|------|----------|----------|-----|---------|"
    )
    lines = [header]
    for r in results:
        verdict = synthetic_verdict(r) if r["tier"] == "synthetic" else "-"
        lines.append(
            f"| {r['call_id']} | {r['tier']} | {r['gold_speakers']} | "
            f"{r['pred_speakers']} | {r['der']:.3f} | {verdict} |"
        )
    return "\n".join(lines)


def go_no_go(syn_results: list[dict]) -> tuple[str, str]:
    """Final gate decision from the synthetic calls only.

    A collapse (wrong speaker count) is a real failure. A 'too easy' warning is
    informational, since HarperValley owns real DER, not the synthetic tier.
    """
    verdicts = [synthetic_verdict(r) for r in syn_results]
    collapsed = [v for v in verdicts if v == "WARN collapse"]
    too_hard = [v for v in verdicts if v == "WARN too hard"]
    too_easy = [v for v in verdicts if v == "WARN too easy"]

    if collapsed or too_hard:
        decision = "NO-GO"
        note = (
            "At least one synthetic call did not give a clean 2-speaker split "
            "in a sane DER range. Retune voice spacing before generating the "
            "remaining calls."
        )
    elif too_easy:
        decision = "GO (with note)"
        note = (
            "Synthetic DER sits near zero on some calls, so the voices may be "
            "easy to separate. This is informational, not a failure: real DER "
            "credibility comes from HarperValley, and the synthetic tier owns "
            "the event labels, not the DER number."
        )
    else:
        decision = "GO"
        note = (
            "Every synthetic call gave a clean 2-speaker split with DER in a "
            "sane middle range. The voice spacing is a reasonable challenge."
        )
    return decision, note


def write_report(results: list[dict], decision: str, note: str, hv_der: list[float]) -> None:
    table = build_table(results)
    hv_line = (
        ", ".join(f"{d:.3f}" for d in hv_der) if hv_der else "none run"
    )
    body = f"""# Diarization gate report

This is the voice-diversity go/no-go checkpoint from the dataset design. It runs
pyannote speaker-diarization-3.1 on the first 3 synthetic calls and on 2
HarperValley calls, then compares predicted speaker turns to the gold
speaker_segments in the registry.

## Results

{table}

## Verdict: {decision}

{note}

## Reading the numbers

The synthetic tier is a voice-diversity check, not an ASR or DER stress test.
What we want there is a clean 2-speaker split and a DER that is neither near
zero (voices too distinct) nor sky high (speakers collapsing). The current
"cross" voice pair mixes a male agent and a female customer, so some separation
is expected to be easy.

Real DER credibility comes from HarperValley, the primary tier for diarization.
The HarperValley DER values on the 2 sampled calls were: {hv_line}. Those are
the numbers that stand in any benchmark reporting, since they run on real
support audio with real speaker overlap and channel noise.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(body, encoding="utf-8")


def main() -> None:
    rows = load_registry()
    syn = [r for r in rows if r["source"] == "synthetic-kokoro-v1"][:3]
    hv = [r for r in rows if r["source"] == "harpervalley"][:2]

    print("Running diarization on 3 synthetic + 2 HarperValley calls. This can")
    print("take a minute or two on the GPU.\n")

    syn_results = [evaluate(r) for r in syn]
    hv_results = [evaluate(r) for r in hv]
    results = syn_results + hv_results

    table = build_table(results)
    print(table)
    print()

    decision, note = go_no_go(syn_results)
    print(f"GATE DECISION: {decision}")
    print(note)

    hv_der = [r["der"] for r in hv_results]
    write_report(results, decision, note, hv_der)
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
