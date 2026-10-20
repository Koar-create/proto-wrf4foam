#!/usr/bin/env python3
"""Style-word and repetition audit for the Section 3.3 intermediate draft."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MD = REPO / "docs" / "scs-wrf-of-manuscript" / "02-rough-writing-3.3-only-medium.md"

SKIP = ("#", "---", "![", "|", "Table 5.", "**(", "$$")

NG = (
    "delve underscores underscore showcase meticulous pivotal unveil realm "
    "commendable boast groundbreaking bolster foster nuance interplay heighten "
    "harness illuminate navigate elevate enhance crucial invaluable multifaceted "
    "tapestry notably particularly primarily predominantly additionally moreover "
    "furthermore importantly specifically ultimately undoubtedly strategically "
    "effectively thorough robust comprehensive significant dynamic potential "
    "intricate methodical holistic paradigm cornerstone synergy insights capabilities "
    "pronounced substantial remarkable notable noteworthy demonstrate exhibit "
    "reveal highlight emphasize comprehensive"
).split()

PHRASES = [
    "mesoscale field",
    "not present in the mesoscale",
    "not inherited from the mesoscale",
    "Λ = 1",
    "route-mean Λ",
    "flight level",
    "exceedance",
    "within unity",
    "below unity",
    "under both weather regimes",
    "smooth rather than constant",
]


def main() -> int:
    txt = MD.read_text(encoding="utf-8")
    body = [
        ln for ln in txt.splitlines()
        if ln.strip() and not ln.startswith(SKIP)
    ]
    print("== paragraph word counts ==")
    for i, ln in enumerate(body, 1):
        print(f"  para {i:>2}: {len(ln.split()):>4} words  | {ln[:58]}...")
    print("total body words:", sum(len(ln.split()) for ln in body))

    low = txt.lower()
    print("\n== NG / style-word hits ==")
    for w in sorted(set(NG)):
        n = len(re.findall(r"\b" + re.escape(w) + r"\w{0,3}\b", low))
        if n:
            print(f"  {w:<14} {n}")

    print("\n== repeated phrase scan ==")
    for ph in PHRASES:
        print(f"  {ph!r:<36} {low.count(ph.lower())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
