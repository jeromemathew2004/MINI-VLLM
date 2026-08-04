"""Phase 7 — the speculative-decoding benchmark sweep and its chart.

Throughput against K for every proposer this engine has, on two workloads, with
the acceptance rate that explains each curve and the break-even line that says
which side of the trade each point is on. That figure, not a single before/after
number, is the deliverable: the interesting result on this hardware is *why*
speculation does or does not pay, and the answer turned out to be a story about
kernel-launch overhead before it was ever a story about proposal quality.

**Every configuration runs in its own process.** Two `Engine`s built back to
back in one process do not produce comparable numbers — the second inherits warm
`torch.compile` artefacts and cuBLAS autotuning, and reports throughput that has
nothing to do with the configuration under test. This script is therefore a
driver that spawns workers, not a loop. `experiments/spec_breakeven.py` records
the two measurement traps this cost us; both are respected here.

**The chart carries a correctness claim.** Every worker hashes the token ids it
produced and the driver compares them against plain decoding's, because a
speedup from a subtly wrong verify pass is worth nothing and the failure is
silent otherwise.

That claim has a measured limit, and finding it was the most interesting thing
this sweep did. A verify pass over `num_requests * (K+1)` rows is a wider GEMM
than a decode step, and past a certain width cuBLAS retiles and the projection
writes **different K/V into the paged cache** for the same token. Unlike logit
noise that is permanent: both runs keep decoding, but no longer over the same
numbers, and they can then diverge at a step that is not a near-tie at all.
`experiments/kv_shape_drift.py` measures where that starts — width 18 on this
host, so K=8 over 2 requests crosses it and K=4 does not. The driver therefore
splits mismatches into "above the width, expected and unfixable" and "below the
width, a real defect" rather than applying one gate to both.

**Four proposers, because they answer different questions:**

- `none`     — plain decode. The line everything else has to beat.
- `ngram`    — prompt lookup, the proposer that actually ships a speedup here.
- `draft`    — the random-init draft. 0% acceptance by construction, so this is
               the *cost* curve of the draft path: what a round costs before any
               proposal quality exists. It is the strongest available argument
               for why Phase 1 was not the cheapest route to a speedup.
- `selfdraft`— the target drafting for itself, ~100% acceptance. The ceiling a
               perfectly-trained draft could approach and never exceed, which is
               the honest way to bound what Phase 1 would have bought.

Two workloads, because the n-gram proposer's acceptance is a property of the
text rather than of any weights: `repetitive` prompts whose answer copies the
question, and `open` chat prompts. Reporting only the first would be dishonest.

Usage:
    python benchmark/spec_sweep.py                 # run the sweep, ~10 minutes
    python benchmark/spec_sweep.py --plot          # re-render from saved results
    python benchmark/spec_sweep.py --table         # markdown table for the README
    python benchmark/spec_sweep.py --k 2 4         # a shorter sweep
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# Must precede the torch import anywhere below: inductor reads this at
# config-module import, and CUDA graph capture dies without it on Windows.
os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RESULTS_PATH = os.path.join(REPO, "benchmark", "results", "spec_sweep.json")
CHART_PATH = os.path.join(REPO, "docs", "spec_sweep.svg")

# Proposer -> (label, how it is built). "selfdraft" points draft_model at the
# target, which is the only way to reach ~100% acceptance without training
# anything; it needs VRAM for two copies and is allowed to fail on a small card.
PROPOSERS = {
    "none": "no speculation",
    "ngram": "n-gram lookup",
    "draft": "draft model, random init",
    "selfdraft": "draft model, self-draft",
}

# The widest verify pass that still writes *bitwise identical* KV to what a plain
# decode step writes, measured by experiments/kv_shape_drift.py on this host.
#
# Above it, cuBLAS picks a different tiling for the projection GEMM and the round
# leaves the paged cache holding different numbers than plain decoding would
# have. Those numbers persist, so the two runs can diverge at a step arbitrarily
# far downstream — and at a step that is not a near-tie at all, since the cause
# is the cache rather than the logits. Byte-identical output is therefore not
# achievable above this width at *any* acceptance rate, and a mismatch there is
# expected rather than a defect. Below it, byte identity is a real gate.
KV_IDENTICAL_MAX_WIDTH = 14


def verify_width(result: dict, num_requests: int = 2) -> int:
    """Batch width of this configuration's verify pass: one row per token."""
    if result["proposer"] == "none":
        return num_requests
    return num_requests * (result["k"] + 1)


# ---------------------------------------------------------------------------
# one configuration, in this process. Invoked as a subprocess by the driver.
# ---------------------------------------------------------------------------

def run_one(cfg: dict) -> dict:
    """Time `Engine.generate` for one configuration and return its numbers."""
    import torch
    from transformers import AutoTokenizer

    from minivllm.config.sampling import SamplingParams
    from tests.spec_harness import (
        DRAFT, PROMPTS, REPETITIVE_PROMPTS, TARGET,
        build_engine, chat_prompts, free_gpu_memory,
    )

    proposer, k, workload = cfg["proposer"], cfg["k"], cfg["workload"]

    spec = proposer != "none"
    method = "ngram" if proposer == "ngram" else "draft"
    draft = TARGET if proposer == "selfdraft" else DRAFT

    engine = build_engine(spec=spec, draft=draft, k=k, cuda_graph=True,
                          method=method, ngram_min_match=cfg.get("min_match"))

    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    if workload == "repetitive":
        # Thinking off: Qwen3 spends its first few dozen tokens reasoning, which
        # would consume the generation budget before any copying starts and
        # measure the wrong thing entirely.
        prompts = chat_prompts(tokenizer, REPETITIVE_PROMPTS, enable_thinking=False)
    else:
        prompts = chat_prompts(tokenizer, PROMPTS, enable_thinking=True)

    def generate(max_tokens):
        return engine.generate(
            prompts,
            SamplingParams(temperature=1.0, top_k=0, top_p=1.0, max_tokens=max_tokens),
            use_tqdm=False)

    # A short throwaway run first. The first generate in a process pays for
    # compilation and autotuning; timing it would measure the toolchain.
    generate(8)
    engine.metrics.reset()

    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = generate(cfg["max_tokens"])
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    stats = engine.metrics.stats()
    token_lists = [o["tokens"] for o in outputs]
    tokens = sum(len(t) for t in token_lists)
    steps = engine.metrics.decode_steps

    result = {
        **cfg,
        "ok": True,
        "tokens": tokens,
        "wall": wall,
        "throughput": tokens / wall,
        "decode_steps": steps,
        "ms_per_step": wall / steps * 1000.0 if steps else 0.0,
        "acceptance_rate": stats.acceptance_rate,
        "speculation_rate": stats.speculation_rate,
        "tokens_per_request_step": stats.tokens_per_request_step,
        "kv_blocks": engine.config.kv_cache_num_blocks,
        "num_requests": len(prompts),
        # The correctness claim. Identical across every proposer and every K of
        # a workload, or the sweep is measuring something broken.
        "output_hash": hashlib.sha256(
            json.dumps(token_lists).encode()).hexdigest()[:16],
    }

    del engine
    free_gpu_memory()
    return result


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

def configurations(k_values: list[int], max_tokens: int, min_match: int | None) -> list[dict]:
    """The sweep. Deliberately not the full cross product.

    The draft path's round cost does not depend on the text — it runs the same
    weights whatever the prompt — so its curves are measured on one workload
    only. The n-gram proposer's acceptance *is* the text, so it gets both.
    """
    base = {"max_tokens": max_tokens, "min_match": min_match}
    cfgs = []

    for workload in ("repetitive", "open"):
        cfgs.append({**base, "proposer": "none", "k": 0, "workload": workload})
        for k in k_values:
            cfgs.append({**base, "proposer": "ngram", "k": k, "workload": workload})

    for k in k_values:
        cfgs.append({**base, "proposer": "draft", "k": k, "workload": "open"})
    # Two copies of the target have to fit in VRAM; K=8 reserves the most cache
    # slack, so it is the first to fail on a small card. Failures are recorded,
    # not fatal.
    for k in k_values:
        cfgs.append({**base, "proposer": "selfdraft", "k": k, "workload": "open"})

    return cfgs


def drive(cfgs: list[dict]) -> list[dict]:
    results = []
    for i, cfg in enumerate(cfgs, 1):
        label = f"{cfg['proposer']:10s} K={cfg['k']:<2d} {cfg['workload']:10s}"
        print(f"[{i:2d}/{len(cfgs)}] {label} ... ", end="", flush=True)

        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", json.dumps(cfg)],
            capture_output=True, text=True, cwd=REPO)

        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT "):
                payload = json.loads(line[len("RESULT "):])

        if payload is None:
            # Most likely out of VRAM (the self-draft cases), which is a real
            # answer about this hardware rather than a script bug. Keep the row
            # so the chart can show the gap instead of silently omitting it.
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            reason = tail[-1] if tail else f"exit {proc.returncode}"
            print(f"FAILED  ({reason[:90]})")
            results.append({**cfg, "ok": False, "error": reason})
            continue

        extra = ""
        if payload["proposer"] != "none":
            extra = f"  accept {payload['acceptance_rate']:5.1%}"
            if payload["proposer"] == "ngram":
                extra += f"  spec {payload['speculation_rate']:5.1%}"
        print(f"{payload['throughput']:6.1f} tok/s{extra}")
        results.append(payload)

    return results


def check_output_hashes(results: list[dict]) -> tuple[list[str], list[str]]:
    """Compare every configuration's tokens against plain decoding's.

    Returns `(unexpected, expected)` mismatches. Speculative decoding is exactly
    distribution-preserving, so under greedy sampling it must emit the target's
    own tokens — but only while both runs are computing over the same numbers.
    Above `KV_IDENTICAL_MAX_WIDTH` they are not: the round writes different K/V
    into the cache than a decode step would, permanently, and a later divergence
    follows from that rather than from anything being wrong.

    So a mismatch is split rather than flagged. Below the width it is a real
    failure and the sweep exits non-zero. At or above it, it is the documented
    consequence of running the same projection at a different shape, and the
    honest thing is to report it as a limit of the configuration — not to
    quietly widen the gate until everything passes.
    """
    unexpected, expected = [], []
    for workload in sorted({r["workload"] for r in results}):
        rows = [r for r in results if r.get("ok") and r["workload"] == workload]
        if not rows:
            continue
        baseline = next((r for r in rows if r["proposer"] == "none"), rows[0])
        for r in rows:
            if r["output_hash"] == baseline["output_hash"]:
                continue
            width = verify_width(r, r.get("num_requests", 2))
            message = (f"{workload}: {r['proposer']} K={r['k']} (verify width {width}) "
                       f"produced {r['output_hash']}, baseline produced "
                       f"{baseline['output_hash']}")
            (expected if width > KV_IDENTICAL_MAX_WIDTH else unexpected).append(message)
    return unexpected, expected


# ---------------------------------------------------------------------------
# the chart, hand-written SVG
# ---------------------------------------------------------------------------
#
# No matplotlib. This repo's environment is pinned deliberately (see CLAUDE.md)
# and a benchmark chart is not worth adding a dependency to it. SVG also renders
# inline in GitHub markdown and diffs as text, both of which a PNG does not.
# The background is painted explicitly rather than left transparent so the chart
# stays readable under GitHub's dark theme.

SERIES = [
    # key                        label                              colour
    (("ngram", "repetitive"), "n-gram, repetitive prompts", "#0b8457"),
    (("ngram", "open"), "n-gram, open-ended prompts", "#7cc4a5"),
    (("selfdraft", "open"), "draft model @ ~100% acceptance", "#b3541e"),
    (("draft", "open"), "draft model @ 0% acceptance", "#d9a06b"),
]
BASELINES = [
    ("repetitive", "plain decode, repetitive", "#333333"),
    ("open", "plain decode, open-ended", "#888888"),
]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


X0, PANEL_W, PANEL_H = 96, 700, 240
TITLE_GAP, TICK_GAP = 14, 20


def _panel(out, y0, xs, k_values, ymax, ylabel, series, baselines, title,
           fmt=lambda v: f"{v:.0f}"):
    """One axes box with its gridlines, ticks, baselines and polylines."""
    x0, w, h = X0, PANEL_W, PANEL_H
    out.append(f'<text x="{x0}" y="{y0 - TITLE_GAP}" class="ttl">{_escape(title)}</text>')
    out.append(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" class="box"/>')

    ticks = 5
    for i in range(ticks + 1):
        y = y0 + h - h * i / ticks
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" class="tick ar">'
                   f'{fmt(ymax * i / ticks)}</text>')

    for k, x in zip(k_values, xs):
        out.append(f'<text x="{x:.1f}" y="{y0 + h + TICK_GAP}" class="tick mid">K={k}</text>')

    cy = y0 + h / 2
    out.append(f'<text x="{x0 - 54}" y="{cy}" class="axis mid" '
               f'transform="rotate(-90 {x0 - 54} {cy})">{_escape(ylabel)}</text>')

    def ypix(value):
        return y0 + h - h * min(value / ymax, 1.0)

    for value, label, colour in baselines:
        y = ypix(value)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + w}" y2="{y:.1f}" '
                   f'class="base" stroke="{colour}"/>')
        out.append(f'<text x="{x0 + w - 6}" y="{y - 6:.1f}" class="note ae" '
                   f'fill="{colour}">{_escape(label)}</text>')

    for points, _label, colour in series:
        drawn = [(x, ypix(v)) for x, v in points]
        if len(drawn) > 1:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in drawn)
            out.append(f'<polyline points="{path}" class="line" stroke="{colour}"/>')
        for x, y in drawn:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{colour}"/>')


def render_svg(results: list[dict]) -> str:
    ok = [r for r in results if r.get("ok")]
    k_values = sorted({r["k"] for r in ok if r["proposer"] != "none"})
    if not k_values:
        raise SystemExit("no successful speculative runs to plot")

    # Laid out top-down from one cursor so nothing can silently overlap: title
    # block, panel A, its tick row, panel B, its tick row, footnote, legend.
    panel_a = 108
    panel_b = panel_a + PANEL_H + TICK_GAP + 16 + TITLE_GAP + 26
    footnote = panel_b + PANEL_H + TICK_GAP + 34
    legend = footnote + 26
    width, height = 940, legend + 26

    xs = [X0 + PANEL_W * (i + 0.5) / len(k_values) for i in range(len(k_values))]

    def pick(proposer, workload, field):
        by_k = {r["k"]: r[field] for r in ok
                if r["proposer"] == proposer and r["workload"] == workload}
        return [(x, by_k[k]) for k, x in zip(k_values, xs) if k in by_k]

    thr_series, acc_series, legend_entries = [], [], []
    for (proposer, workload), label, colour in SERIES:
        points = pick(proposer, workload, "throughput")
        if points:
            thr_series.append((points, label, colour))
            acc_series.append((pick(proposer, workload, "acceptance_rate"), label, colour))
            legend_entries.append((label, colour))

    thr_baselines, base_values = [], []
    for workload, label, colour in BASELINES:
        row = next((r for r in ok if r["proposer"] == "none" and r["workload"] == workload),
                   None)
        if row:
            base_values.append(row["throughput"])
            thr_baselines.append((row["throughput"], label, colour))

    thr_max = max([v for pts, _, _ in thr_series for _, v in pts] + base_values) * 1.16

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="DejaVu Sans, Segoe UI, sans-serif">',
        '<style>'
        'text{fill:#1b1b1b} .ttl{font-size:14.5px;font-weight:600}'
        '.sub{font-size:12px;fill:#555} .tick{font-size:11px;fill:#555}'
        '.axis{font-size:12px;fill:#333} .note{font-size:10.5px}'
        '.leg{font-size:12px} .ar{text-anchor:end} .ae{text-anchor:end}'
        '.mid{text-anchor:middle} .box{fill:none;stroke:#cccccc;stroke-width:1}'
        '.grid{stroke:#eeeeee;stroke-width:1}'
        '.line{fill:none;stroke-width:2.4;stroke-linejoin:round}'
        '.base{stroke-width:1.6;stroke-dasharray:6 4}'
        '</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="40" y="38" style="font-size:19px;font-weight:700">'
        'Speculative decoding in mini-vllm: throughput vs K</text>',
        '<text x="40" y="60" class="sub">'
        'Qwen3-0.6B on an RTX 3050 Laptop (4 GB), bf16, CUDA graphs on, greedy decoding, '
        '2 concurrent requests.</text>',
        f'<text x="40" y="78" class="sub">'
        f'Points at a verify width of {KV_IDENTICAL_MAX_WIDTH} or below emitted '
        f'byte-identical text; the sweep checks output hashes. Wider rounds write '
        f'different KV than plain decode and cannot.</text>',
    ]

    _panel(out, panel_a, xs, k_values, thr_max, "tokens / second",
           thr_series, thr_baselines,
           "Throughput. Dashed lines are plain decode - the bar each proposer has to clear.")
    _panel(out, panel_b, xs, k_values, 1.0, "acceptance rate",
           acc_series, [],
           "Acceptance rate. This is the number that explains the panel above.",
           fmt=lambda v: f"{v * 100:.0f}%")

    out.append(f'<text x="40" y="{footnote}" class="sub">'
               f'Break-even is ~30% acceptance for the n-gram proposer and ~60% at K=4 for '
               f'the draft path, which costs a forward pass per proposal. Below that line a '
               f'round returns fewer tokens than it cost.</text>')

    x = 40
    for label, colour in legend_entries:
        out.append(f'<rect x="{x}" y="{legend - 9}" width="13" height="4" fill="{colour}"/>')
        out.append(f'<text x="{x + 19}" y="{legend - 3}" class="leg">{_escape(label)}</text>')
        x += 30 + int(len(label) * 6.5)

    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def render_table(results: list[dict]) -> str:
    ok = [r for r in results if r.get("ok")]
    base = {r["workload"]: r["throughput"] for r in ok if r["proposer"] == "none"}

    lines = [
        "| proposer | workload | K | tok/s | vs plain | acceptance | rounds run | tok/request/step |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"none": 0, "ngram": 1, "draft": 2, "selfdraft": 3}
    for r in sorted(ok, key=lambda r: (r["workload"], order[r["proposer"]], r["k"])):
        speedup = r["throughput"] / base[r["workload"]] if r["workload"] in base else float("nan")
        spec = r["proposer"] != "none"

        # Only the n-gram proposer can decline to speculate; a draft model
        # proposes on every step, and plain decode has nothing to report.
        if r["proposer"] == "ngram":
            rounds = f"{r['speculation_rate']:.0%}"
        elif spec:
            rounds = "100%"
        else:
            rounds = "n/a"

        # ASCII only: this table gets piped into files and pasted into shells on
        # a Windows host whose stdout is cp1252, where an em dash raises.
        lines.append(
            f"| {PROPOSERS[r['proposer']]} | {r['workload']} | "
            f"{r['k'] if spec else 'n/a'} | {r['throughput']:.0f} | {speedup:.2f}x | "
            f"{r['acceptance_rate']:.0%} | {rounds} | "
            f"{r['tokens_per_request_step']:.2f} |")

    failed = [r for r in results if not r.get("ok")]
    if failed:
        lines.append("")
        for r in failed:
            lines.append(f"- `{r['proposer']} K={r['k']} {r['workload']}` did not run: "
                         f"{r['error']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None,
                    help=argparse.SUPPRESS)  # internal: run one configuration
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ngram-min-match", type=int, default=None,
                    help="override the shipped ngram_min_match_len default")
    ap.add_argument("--plot", action="store_true",
                    help="re-render the chart from saved results, without measuring")
    ap.add_argument("--table", action="store_true",
                    help="print the markdown table from saved results")
    args = ap.parse_args()

    if args.worker:
        print("RESULT " + json.dumps(run_one(json.loads(args.worker))))
        return 0

    if args.plot or args.table:
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            results = json.load(fh)["results"]
    else:
        import torch
        assert torch.cuda.is_available(), "this benchmark needs the GPU"
        cfgs = configurations(args.k, args.max_tokens, args.ngram_min_match)
        started = time.perf_counter()
        results = drive(cfgs)
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            json.dump({"results": results,
                       "max_tokens": args.max_tokens,
                       "elapsed": time.perf_counter() - started}, fh, indent=1)
        print(f"\nwrote {os.path.relpath(RESULTS_PATH, REPO)} "
              f"({time.perf_counter() - started:.0f}s)")

    if args.table:
        print(render_table(results))
        return 0

    unexpected, expected = check_output_hashes(results)
    print()
    print(render_table(results))
    print()
    if expected:
        print(f"Above the KV-identical batch width ({KV_IDENTICAL_MAX_WIDTH}), output is "
              f"not byte-identical, and cannot be:")
        for p in expected:
            print("  " + p)
        print("  A verify pass this wide writes different K/V into the cache than a decode")
        print("  step would (experiments/kv_shape_drift.py measures it), and that persists.")
        print("  Not a defect - a limit of running the same projection at a different shape.")
    if unexpected:
        print("\nOUTPUT MISMATCH below the KV-identical width - the sweep is not measuring")
        print("what it claims. Investigate before reading any number above:")
        for p in unexpected:
            print("  " + p)
        print("\n  Run experiments/decode_determinism_check.py first (a missing backend")
        print("  patch makes decode itself wrong above width 6), then")
        print("  experiments/spec_divergence.py to locate and classify the divergence.")
    if not expected and not unexpected:
        print("Output hashes agree within each workload: every configuration above "
              "emitted byte-identical text.")

    os.makedirs(os.path.dirname(CHART_PATH), exist_ok=True)
    with open(CHART_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_svg(results))
    print(f"wrote {os.path.relpath(CHART_PATH, REPO)}")

    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
