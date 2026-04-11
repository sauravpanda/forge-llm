#!/usr/bin/env python3
"""Compare criterion benchmark results between two cargo bench runs.

Reads /tmp/base-bench.txt and /tmp/pr-bench.txt (criterion human-readable
output captured via `cargo bench 2>&1 | tee`), computes percentage change
per benchmark, writes a markdown summary to /tmp/bench-summary.txt, and
sets GITHUB_OUTPUT variables `regression=true/false`.

Criterion text output format:
    graph_build/tiny    time:   [1.234 µs  1.256 µs  1.278 µs]
The three values are lower_bound, point_estimate, upper_bound.
"""
import re
import sys
import os

REGRESSION_THRESHOLD = 0.05  # 5%

# Map criterion unit strings to nanoseconds
UNIT_TO_NS = {
    "ps": 1e-3,
    "ns": 1.0,
    "us": 1e3,
    "\xb5s": 1e3,  # µs as latin-1
    "ms": 1e6,
    "s": 1e9,
}


def parse_unit(unit_str):
    unit_str = unit_str.replace("\xb5", "u")  # normalize µ -> u
    return UNIT_TO_NS.get(unit_str, 1.0)


def parse_bench_txt(path):
    """Return dict of {benchmark_name: point_estimate_ns}."""
    results = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # e.g.: "graph_build/tiny    time:   [1.234 µs  1.256 µs  1.278 µs]"
                m = re.search(
                    r"^(.+?)\s{2,}time:\s+\["
                    r"[\d.]+ \S+\s+"      # lower bound (skip)
                    r"([\d.]+) (\S+)\s+"  # point estimate
                    r"[\d.]+ \S+\s*\]",
                    line,
                )
                if m:
                    name = m.group(1).strip()
                    val = float(m.group(2))
                    unit = m.group(3)
                    ns = val * parse_unit(unit)
                    results[name] = ns
    except FileNotFoundError:
        pass
    return results


def main():
    base = parse_bench_txt("/tmp/base-bench.txt")
    pr = parse_bench_txt("/tmp/pr-bench.txt")

    github_output = os.environ.get("GITHUB_OUTPUT", "/dev/stderr")

    if not base or not pr:
        print("No benchmark data found — skipping comparison.")
        with open(github_output, "a") as f:
            f.write("regression=false\n")
        with open("/tmp/bench-summary.txt", "w") as f:
            f.write("No benchmark data available (benches may not have run).\n")
        return

    rows = []
    regression_found = False
    for name in sorted(set(base) | set(pr)):
        b = base.get(name, 0)
        p = pr.get(name, 0)
        if b == 0 or p == 0:
            rows.append("| `{}` | N/A | N/A | N/A |".format(name))
            continue
        ratio = (p - b) / b
        pct = "{:+.1f}%".format(ratio * 100)
        flag = ""
        if ratio > REGRESSION_THRESHOLD:
            flag = " REGRESSION"
            regression_found = True
        elif ratio < -0.05:
            flag = " improvement"
        rows.append(
            "| `{}` | {:.3f} ms | {:.3f} ms | {}{} |".format(
                name, b / 1e6, p / 1e6, pct, flag
            )
        )

    threshold_pct = int(REGRESSION_THRESHOLD * 100)
    with open("/tmp/bench-summary.txt", "w") as f:
        f.write("### Benchmark comparison (base vs PR)\n\n")
        f.write("| Benchmark | Base (ms) | PR (ms) | Change |\n")
        f.write("|-----------|-----------|---------|--------|\n")
        for row in rows:
            f.write(row + "\n")
        f.write("\n> Threshold: >{}% slowdown flagged as regression.\n".format(threshold_pct))

    with open(github_output, "a") as f:
        f.write("regression={}\n".format("true" if regression_found else "false"))

    if regression_found:
        print("Performance regression detected!")
    else:
        print("No regressions detected.")


if __name__ == "__main__":
    main()
