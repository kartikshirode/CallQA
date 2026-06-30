"""Deterministic synthetic call-script generator (Phase 1).

Given a seed, this builds call scripts by picking lines from the domain banks
with a seeded RNG, then inserting pauses, the occasional interruption, and event
tags that match exactly what was injected. No network, no LLM, no audio. Same
seed in means byte-identical scripts out.

Negative-class policy: across the batch of 20 calls, each event type
(silence, interruption, escalation, compliance) is a negative in roughly 40%
of calls (8 of 20). The split is assigned by construction, not left to the RNG,
so the ratio lands on target. A compliance negative means a required phrase
(recording_disclosure or identity_verification) is deliberately omitted and the
call carries a negative tag on that subtype, giving the detector a labeled miss.
"""
from __future__ import annotations

import random

from callqa.registry.schema import ComplianceSubtype, Domain, EventType, Polarity
from callqa.synth.banks import BANKS
from callqa.synth.script_schema import CallScript, EventTag, ScriptTurn

# A pause at or above this many seconds counts as a labeled silence.
SILENCE_PAUSE_THRESHOLD = 2.0

# Domains in a fixed order so batch output is stable.
DOMAINS = ("billing", "refund", "cancellation", "tech_support")
CALLS_PER_DOMAIN = 5
BATCH_SIZE = len(DOMAINS) * CALLS_PER_DOMAIN  # 20

# Target negatives per event type across the batch. 8 of 20 is 40%.
TARGET_NEGATIVES = 8

# Required compliance phrases. One of these is the one omitted on a
# compliance-negative call.
REQUIRED_SUBTYPES = (
    ComplianceSubtype.recording_disclosure,
    ComplianceSubtype.identity_verification,
)


def _pick(rng: random.Random, options: list[str]) -> str:
    """Pick one line deterministically from a slot's alternatives."""
    return options[rng.randrange(len(options))]


def _negative_flags() -> dict[str, list[bool]]:
    """Decide, per event type, which of the 20 call indices are negatives.

    Returns a dict event-name -> list[bool] of length BATCH_SIZE. We space the
    8 negatives across the 20 slots with a fixed stride and a per-event offset,
    so no single call lands on every negative at once and the ratio is exact.
    """
    flags: dict[str, list[bool]] = {}
    event_names = ["silence", "interruption", "escalation", "compliance"]
    for offset, name in enumerate(event_names):
        is_negative = [False] * BATCH_SIZE
        # Evenly spread TARGET_NEGATIVES picks over the batch.
        for k in range(TARGET_NEGATIVES):
            idx = (offset + k * BATCH_SIZE // TARGET_NEGATIVES) % BATCH_SIZE
            # Resolve rare collisions by stepping forward to the next free slot.
            while is_negative[idx]:
                idx = (idx + 1) % BATCH_SIZE
            is_negative[idx] = True
        flags[name] = is_negative
    return flags


def generate_call(
    domain: str,
    call_id: str,
    seed: int,
    *,
    silence_negative: bool = False,
    interruption_negative: bool = False,
    escalation_negative: bool = False,
    compliance_negative: bool = False,
    omitted_subtype: ComplianceSubtype = ComplianceSubtype.recording_disclosure,
) -> CallScript:
    """Build one call script.

    The *_negative flags say which events to leave out of this call. When an
    event is not a negative, it is injected as a positive with a matching tag.
    For a compliance negative, the omitted_subtype required phrase is dropped
    and tagged negative so there is a labeled miss.
    """
    rng = random.Random(seed)
    bank = BANKS[domain]
    turns: list[ScriptTurn] = []
    notes_bits: list[str] = []

    disclosure_omitted = compliance_negative and omitted_subtype is ComplianceSubtype.recording_disclosure
    verification_omitted = compliance_negative and omitted_subtype is ComplianceSubtype.identity_verification

    # 1. Opening.
    turns.append(ScriptTurn(speaker="agent", text=_pick(rng, bank["opening_agent"])))

    # 2. Recording disclosure (required compliance phrase).
    if disclosure_omitted:
        # Omit the line. Tag the miss on the opening turn so the label has an anchor.
        turns[-1].event_tags.append(
            EventTag(
                event_type=EventType.compliance,
                subtype=ComplianceSubtype.recording_disclosure,
                polarity=Polarity.negative,
            )
        )
        notes_bits.append("recording_disclosure omitted")
    else:
        turns.append(
            ScriptTurn(
                speaker="agent",
                text=_pick(rng, bank["recording_disclosure"]),
                event_tags=[
                    EventTag(
                        event_type=EventType.compliance,
                        subtype=ComplianceSubtype.recording_disclosure,
                        polarity=Polarity.positive,
                    )
                ],
            )
        )

    # 3. Identity verification (required compliance phrase).
    if verification_omitted:
        # Skip the verify exchange and tag the miss on the most recent turn.
        turns[-1].event_tags.append(
            EventTag(
                event_type=EventType.compliance,
                subtype=ComplianceSubtype.identity_verification,
                polarity=Polarity.negative,
            )
        )
        notes_bits.append("identity_verification omitted")
    else:
        turns.append(
            ScriptTurn(
                speaker="agent",
                text=_pick(rng, bank["verify_request"]),
                event_tags=[
                    EventTag(
                        event_type=EventType.compliance,
                        subtype=ComplianceSubtype.identity_verification,
                        polarity=Polarity.positive,
                    )
                ],
            )
        )
        turns.append(ScriptTurn(speaker="customer", text=_pick(rng, bank["verify_response"])))

    # 4. Core issue.
    turns.append(ScriptTurn(speaker="customer", text=_pick(rng, bank["issue_customer"])))
    turns.append(ScriptTurn(speaker="agent", text=_pick(rng, bank["issue_agent"])))

    # 5. Back and forth, with an optional silence and an optional interruption.
    if silence_negative:
        # Clean stretch: a back-and-forth with no labeled gap.
        turns.append(ScriptTurn(speaker="agent", text=_pick(rng, bank["back_and_forth_agent"])))
        turns.append(ScriptTurn(speaker="customer", text=_pick(rng, bank["back_and_forth_cust"])))
        notes_bits.append("no silence")
    else:
        # Insert a deliberate dead-air gap before the agent's working line.
        pause = round(rng.uniform(2.4, 4.5), 1)
        turns.append(
            ScriptTurn(
                speaker="agent",
                text=_pick(rng, bank["back_and_forth_agent"]),
                pause_before=pause,
                event_tags=[EventTag(event_type=EventType.silence)],
            )
        )
        turns.append(ScriptTurn(speaker="customer", text=_pick(rng, bank["back_and_forth_cust"])))

    if interruption_negative:
        notes_bits.append("no interruption")
    else:
        # Customer cuts in over the agent: overlap becomes an interruption.
        turns.append(
            ScriptTurn(
                speaker="customer",
                text=_pick(rng, bank["back_and_forth_cust"]),
                interrupt_prev=True,
                event_tags=[EventTag(event_type=EventType.interruption)],
            )
        )

    # 6. Escalation.
    if escalation_negative:
        notes_bits.append("no escalation")
    else:
        turns.append(
            ScriptTurn(
                speaker="customer",
                text=_pick(rng, bank["escalation_customer"]),
                event_tags=[EventTag(event_type=EventType.escalation)],
            )
        )
        turns.append(ScriptTurn(speaker="agent", text=_pick(rng, bank["escalation_agent"])))

    # 7. Closing.
    turns.append(ScriptTurn(speaker="agent", text=_pick(rng, bank["closing_agent"])))
    turns.append(ScriptTurn(speaker="customer", text=_pick(rng, bank["closing_customer"])))

    notes = "; ".join(notes_bits) if notes_bits else "all events present"
    return CallScript(
        call_id=call_id,
        domain=Domain(domain),
        seed=seed,
        turns=turns,
        notes=notes,
    )


def batch_event_summary(batch: list[CallScript]) -> dict[EventType, dict[str, float]]:
    """Count positives and negatives per event type across a batch.

    A call is a positive for an event when present_event_types reports it, and
    a negative otherwise. Used to eyeball that each event sits near 40% negative.
    """
    summary: dict[EventType, dict[str, float]] = {}
    total = len(batch)
    for event_type in (
        EventType.silence,
        EventType.interruption,
        EventType.escalation,
        EventType.compliance,
    ):
        positive = sum(1 for s in batch if event_type in s.present_event_types())
        negative = total - positive
        ratio = negative / total if total else 0.0
        summary[event_type] = {
            "positive": positive,
            "negative": negative,
            "negative_ratio": ratio,
        }
    return summary


def generate_batch(seed: int) -> list[CallScript]:
    """Produce the 20 MVP scripts (4 domains x 5), enforcing the negative split.

    Each call gets a derived seed so the batch is reproducible yet calls differ.
    The negative flags per event type are assigned by construction to hit 40%.
    """
    neg = _negative_flags()
    scripts: list[CallScript] = []
    index = 0
    for domain in DOMAINS:
        for n in range(1, CALLS_PER_DOMAIN + 1):
            call_id = f"syn-{domain}-{n:02d}"
            call_seed = seed + index * 1000 + 1

            compliance_negative = neg["compliance"][index]
            # Alternate which required phrase is the omitted one, deterministically.
            omitted = REQUIRED_SUBTYPES[index % len(REQUIRED_SUBTYPES)]

            script = generate_call(
                domain,
                call_id,
                call_seed,
                silence_negative=neg["silence"][index],
                interruption_negative=neg["interruption"][index],
                escalation_negative=neg["escalation"][index],
                compliance_negative=compliance_negative,
                omitted_subtype=omitted,
            )
            scripts.append(script)
            index += 1
    return scripts
