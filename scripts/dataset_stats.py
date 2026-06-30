"""Print a compact summary of the registry, per tier.

Counts, total and average duration for each tier, plus the synthetic event-type
and polarity breakdown. Reads straight from the registry so the numbers stay
true to what is actually registered.

Run it:
    python scripts/dataset_stats.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from callqa.registry.schema import CallRecord, EventType
from callqa.registry.store import Registry


def tier_of(record: CallRecord) -> str:
    return "synthetic" if record.source.lower().startswith("synthetic") else "real-anchor"


def main() -> int:
    registry = Registry()
    records = registry.list_all()

    by_tier: dict[str, list[CallRecord]] = {}
    for r in records:
        by_tier.setdefault(tier_of(r), []).append(r)

    print("CallQA registry stats")
    print("=" * 60)
    print(f"total calls: {len(records)}")
    print()

    for tier in sorted(by_tier):
        rows = by_tier[tier]
        total_dur = sum(r.duration_seconds for r in rows)
        avg_dur = total_dur / len(rows) if rows else 0.0
        domains = Counter(r.domain.value for r in rows)
        sources = Counter(r.source for r in rows)

        print(f"[{tier}]")
        print(f"  calls:        {len(rows)}")
        print(f"  total dur:    {total_dur:.1f}s  ({total_dur / 60:.1f} min)")
        print(f"  avg dur:      {avg_dur:.2f}s")
        print(f"  sources:      {dict(sources)}")
        print(f"  domains:      {dict(domains)}")

        if tier == "synthetic":
            ev = Counter()
            pol = Counter()
            for r in rows:
                for e in r.event_labels or []:
                    ev[e.event_type.value] += 1
                    if e.event_type == EventType.compliance:
                        p = (e.metadata or {}).get("polarity", "unknown")
                        pol[p] += 1
            print(f"  event types:  {dict(ev)}")
            print(f"  compliance polarity: {dict(pol)}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
