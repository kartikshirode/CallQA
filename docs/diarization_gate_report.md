# Diarization gate report

This is the voice-diversity go/no-go checkpoint from the dataset design. It runs
pyannote speaker-diarization-3.1 on the first 3 synthetic calls and on 2
HarperValley calls, then compares predicted speaker turns to the gold
speaker_segments in the registry.

## Results

| call_id | tier | gold spk | pred spk | DER | verdict |
|---------|------|----------|----------|-----|---------|
| syn-billing-01 | synthetic | 2 | 2 | 0.277 | PASS |
| syn-billing-02 | synthetic | 2 | 2 | 0.280 | PASS |
| syn-billing-03 | synthetic | 2 | 2 | 0.274 | PASS |
| hv-a3980352013548b8 | harpervalley | 2 | 2 | 0.120 | - |
| hv-29cfc984729c4f4c | harpervalley | 2 | 1 | 0.559 | - |

## Verdict: GO

Every synthetic call gave a clean 2-speaker split with DER in a sane middle range. The voice spacing is a reasonable challenge.

## Reading the numbers

The synthetic tier is a voice-diversity check, not an ASR or DER stress test.
What we want there is a clean 2-speaker split and a DER that is neither near
zero (voices too distinct) nor sky high (speakers collapsing). The current
"cross" voice pair mixes a male agent and a female customer, so some separation
is expected to be easy.

Real DER credibility comes from HarperValley, the primary tier for diarization.
The HarperValley DER values on the 2 sampled calls were: 0.120, 0.559. Those are
the numbers that stand in any benchmark reporting, since they run on real
support audio with real speaker overlap and channel noise.
