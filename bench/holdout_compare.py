# ruff: noqa: E501
"""Hold-out check (review R1-11): compare shift distributions with and without the benchmark molecules.

    uv run python bench/nmrshiftdb2_shift_distributions.py > all.txt
    uv run python bench/nmrshiftdb2_shift_distributions.py --exclude-selection bench/results/selection.json \
        --exclude-selection bench/results_main/selection.json --exclude-selection bench/results_unreported/selection.json > held_out.txt
    uv run python bench/holdout_compare.py all.txt held_out.txt

A range counts as moved when its p5 or p95 changes by more than 0.05 ppm (1H) or 0.5 ppm (13C).
"""

import re
import sys

LINE = re.compile(
    r"^\s*(?P<label>.+?):\s+n=(?P<n>\d+) min=(?P<min>\S+) p5=(?P<p5>\S+) median=(?P<med>\S+) p95=(?P<p95>\S+)"
)


def parse(path: str) -> dict[str, tuple[int, float, float]]:
    out = {}
    for line in open(path):
        m = LINE.match(line)
        if m:
            out[m["label"].strip()] = (int(m["n"]), float(m["p5"]), float(m["p95"]))
    return out


a, b = parse(sys.argv[1]), parse(sys.argv[2])
worst = 0.0
moved = 0
for label, (n, p5, p95) in a.items():
    n2, q5, q95 = b[label]
    tol = 0.5 if "13C" in label else 0.05
    d = max(abs(p5 - q5), abs(p95 - q95)) if n2 else float("nan")
    flag = "MOVED" if n2 and d > tol else ("n=0" if not n2 else "ok")
    moved += flag == "MOVED"
    if n2:
        worst = max(worst, d / tol)
    if flag != "ok" or d > 0:
        print(
            f"{flag:6} {label}: n {n} -> {n2}, p5 {p5} -> {q5}, p95 {p95} -> {q95} (change {d:.2f}, tolerance {tol})"
        )
print(
    f"\n{len(a)} distributions compared; {moved} moved beyond tolerance; largest change is {worst:.2f} of its tolerance"
)
