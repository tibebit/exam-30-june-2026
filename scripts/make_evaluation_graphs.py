#!/usr/bin/env python3
"""Generate presentation-ready graphs from one evaluation report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


SCENARIO_COLORS = [
    "#2563eb",
    "#f97316",
    "#16a34a",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
]
TEXT = "#0f172a"
MUTED = "#64748b"
GRID = "#dbe4ee"
AXIS = "#334155"
BACKGROUND = "#ffffff"


def read_json(path: Path) -> dict:
    """Read one JSON evaluation report."""

    return json.loads(path.read_text(encoding="utf-8"))


def human_name(name: str) -> str:
    """Make scenario names readable in chart labels."""

    cleaned = name.removesuffix("_eval").replace("_", " ")
    replacements = {
        "random partner heuristic opponents": "random partner + heuristic opponents",
        "advanced partner perfect heuristic opponents": (
            "advanced partner + perfect heuristic opponents"
        ),
    }
    return replacements.get(cleaned, cleaned)


def wrap_label(label: str, max_chars: int = 18) -> list[str]:
    """Wrap long scenario labels without depending on external plotting libs."""

    lines: list[str] = []
    current = ""
    for word in label.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:4]


def svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 13,
    anchor: str = "start",
    weight: str = "400",
    color: str = TEXT,
    extra: str = "",
) -> str:
    """Render escaped SVG text."""

    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}" {extra}>{html.escape(text)}</text>'
    )


def fmt_tick(value: float) -> str:
    """Format chart ticks compactly."""

    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def bar_chart(
    path: Path,
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    y_label: str,
    y_domain: tuple[float, float],
    colors: list[str],
    error_ranges: list[tuple[float, float]] | None = None,
    value_format: str = ".3f",
) -> None:
    """Write a vertical bar chart as SVG."""

    width, height = 1320, 760
    left, right, top, bottom = 100, 56, 106, 186
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = y_domain

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        svg_text(left, 42, title, size=27, weight="700"),
        svg_text(left, 70, subtitle, size=13, color=MUTED),
    ]

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 14, y + 4, fmt_tick(value), size=12, anchor="end", color=MUTED))

    zero_y = sy(0)
    if top <= zero_y <= height - bottom:
        parts.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" '
            f'y2="{zero_y:.2f}" stroke="{AXIS}" stroke-width="1.3"/>'
        )

    slot = plot_w / len(values)
    bar_w = slot * 0.56
    for index, (label, value) in enumerate(zip(labels, values)):
        x_center = left + slot * (index + 0.5)
        y0 = sy(0)
        y1 = sy(value)
        rect_y = min(y0, y1)
        rect_h = abs(y0 - y1)
        color = colors[index % len(colors)]
        parts.append(
            f'<rect x="{x_center - bar_w / 2:.2f}" y="{rect_y:.2f}" '
            f'width="{bar_w:.2f}" height="{rect_h:.2f}" rx="7" fill="{color}"/>'
        )

        if error_ranges:
            lo, hi = error_ranges[index]
            err_top = sy(hi)
            err_bottom = sy(lo)
            parts.append(
                f'<line x1="{x_center:.2f}" y1="{err_top:.2f}" '
                f'x2="{x_center:.2f}" y2="{err_bottom:.2f}" '
                f'stroke="{AXIS}" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{x_center - 10:.2f}" y1="{err_top:.2f}" '
                f'x2="{x_center + 10:.2f}" y2="{err_top:.2f}" '
                f'stroke="{AXIS}" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{x_center - 10:.2f}" y1="{err_bottom:.2f}" '
                f'x2="{x_center + 10:.2f}" y2="{err_bottom:.2f}" '
                f'stroke="{AXIS}" stroke-width="2"/>'
            )

        value_y = y1 - 12 if value >= 0 else y1 + 22
        parts.append(
            svg_text(
                x_center,
                value_y,
                format(value, value_format),
                size=13,
                anchor="middle",
                weight="700",
            )
        )
        for line_index, line in enumerate(wrap_label(label)):
            parts.append(
                svg_text(
                    x_center,
                    height - bottom + 36 + line_index * 17,
                    line,
                    size=12,
                    anchor="middle",
                    color=MUTED,
                )
            )

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        f'stroke="{AXIS}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="{AXIS}" stroke-width="1.4"/>'
    )
    parts.append(
        svg_text(
            24,
            top + plot_h / 2,
            y_label,
            size=13,
            anchor="middle",
            color=AXIS,
            extra=f'transform="rotate(-90 24 {top + plot_h / 2:.2f})"',
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def stacked_outcome_chart(path: Path, *, labels: list[str], scenarios: list[dict]) -> None:
    """Write a stacked win/draw/loss composition chart as SVG."""

    width, height = 1320, 760
    left, right, top, bottom = 100, 190, 106, 186
    plot_w = width - left - right
    plot_h = height - top - bottom
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        svg_text(left, 42, "Evaluation outcomes", size=27, weight="700"),
        svg_text(left, 70, "Win, draw and loss rates; each bar sums to 1.0", size=13, color=MUTED),
    ]

    def sy(value: float) -> float:
        return top + (1 - value) * plot_h

    for i in range(6):
        value = i / 5
        y = sy(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
            f'y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 14, y + 4, f"{value:.1f}", size=12, anchor="end", color=MUTED))

    outcome_specs = [
        ("win_rate", "win", "#16a34a"),
        ("draw_rate", "draw", "#f59e0b"),
        ("loss_rate", "loss", "#dc2626"),
    ]
    slot = plot_w / len(scenarios)
    bar_w = slot * 0.58
    for index, scenario in enumerate(scenarios):
        x = left + slot * (index + 0.5) - bar_w / 2
        base = 0.0
        for key, _, color in outcome_specs:
            value = float(scenario["metrics"][key])
            y_top = sy(base + value)
            y_bottom = sy(base)
            parts.append(
                f'<rect x="{x:.2f}" y="{y_top:.2f}" width="{bar_w:.2f}" '
                f'height="{y_bottom-y_top:.2f}" rx="5" fill="{color}"/>'
            )
            if value >= 0.055:
                parts.append(
                    svg_text(
                        x + bar_w / 2,
                        y_top + (y_bottom - y_top) / 2 + 4,
                        f"{value:.3f}",
                        size=11,
                        anchor="middle",
                        weight="700",
                        color="#ffffff",
                    )
                )
            base += value

        x_center = left + slot * (index + 0.5)
        for line_index, line in enumerate(wrap_label(labels[index])):
            parts.append(
                svg_text(
                    x_center,
                    height - bottom + 36 + line_index * 17,
                    line,
                    size=12,
                    anchor="middle",
                    color=MUTED,
                )
            )

    legend_x = width - right + 28
    for index, (_, label, color) in enumerate(outcome_specs):
        y = top + index * 28
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="18" height="18" rx="4" fill="{color}"/>')
        parts.append(svg_text(legend_x + 28, y + 2, label, size=13, color=MUTED))

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
        f'stroke="{AXIS}" stroke-width="1.4"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
        f'y2="{height-bottom}" stroke="{AXIS}" stroke-width="1.4"/>'
    )
    parts.append(
        svg_text(
            24,
            top + plot_h / 2,
            "rate",
            size=13,
            anchor="middle",
            color=AXIS,
            extra=f'transform="rotate(-90 24 {top + plot_h / 2:.2f})"',
        )
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_csv(path: Path, scenarios: list[dict]) -> None:
    """Write a compact CSV summary next to the SVGs."""

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "games",
                "win_rate",
                "draw_rate",
                "loss_rate",
                "mean_point_difference",
                "standard_error",
                "ci95_low",
                "ci95_high",
            ]
        )
        for scenario in scenarios:
            metrics = scenario["metrics"]
            ci_low, ci_high = metrics["confidence_interval_95"]
            writer.writerow(
                [
                    scenario["name"],
                    metrics["games"],
                    metrics["win_rate"],
                    metrics["draw_rate"],
                    metrics["loss_rate"],
                    metrics["mean_point_difference"],
                    metrics["standard_error"],
                    ci_low,
                    ci_high,
                ]
            )


def write_index(path: Path, files: list[Path], source_report: Path) -> None:
    """Write a small HTML page to browse the generated SVGs."""

    body = "\n".join(
        (
            f'<section><h2>{html.escape(file.stem.replace("_", " ").title())}</h2>'
            f'<img src="{html.escape(file.name)}" alt="{html.escape(file.stem)}"></section>'
        )
        for file in files
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation graphs</title>
  <style>
    body {{ margin: 0; font-family: Inter, Arial, sans-serif; color: #0f172a; background: #f8fafc; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 32px 24px 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    p {{ margin: 0 0 24px; color: #475569; font-size: 14px; }}
    section {{ margin: 0 0 28px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
    h2 {{ margin: 0; padding: 14px 18px; font-size: 15px; border-bottom: 1px solid #e2e8f0; }}
    img {{ display: block; width: 100%; height: auto; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>Evaluation graphs</h1>
    <p>Source report: <code>{html.escape(str(source_report))}</code></p>
    {body}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Read command-line graph generation options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.report.exists():
        raise FileNotFoundError(args.report)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = read_json(args.report)
    scenarios = list(report["scenarios"])
    labels = [human_name(scenario["name"]) for scenario in scenarios]
    games = report["games_per_scenario"]

    files: list[Path] = []
    win_file = args.out_dir / "hist_win_rate.svg"
    bar_chart(
        win_file,
        title="Win rate by scenario",
        subtitle=f"{games} greedy evaluation games per scenario",
        labels=labels,
        values=[float(s["metrics"]["win_rate"]) for s in scenarios],
        y_label="win rate",
        y_domain=(0.0, 1.0),
        colors=SCENARIO_COLORS,
    )
    files.append(win_file)

    diff_file = args.out_dir / "hist_mean_point_difference.svg"
    ci_values = [
        tuple(map(float, s["metrics"]["confidence_interval_95"]))
        for s in scenarios
    ]
    all_ci_values = [value for pair in ci_values for value in pair]
    lo = min(0.0, min(all_ci_values))
    hi = max(0.0, max(all_ci_values))
    pad = (hi - lo) * 0.12 if not math.isclose(lo, hi) else 1.0
    bar_chart(
        diff_file,
        title="Mean point difference by scenario",
        subtitle="Bars show mean point difference; whiskers show 95% confidence interval",
        labels=labels,
        values=[float(s["metrics"]["mean_point_difference"]) for s in scenarios],
        y_label="mean point difference",
        y_domain=(lo - pad, hi + pad),
        colors=SCENARIO_COLORS,
        error_ranges=ci_values,
        value_format=".1f",
    )
    files.append(diff_file)

    outcome_file = args.out_dir / "hist_outcome_composition.svg"
    stacked_outcome_chart(outcome_file, labels=labels, scenarios=scenarios)
    files.append(outcome_file)

    write_csv(args.out_dir / "evaluation_summary.csv", scenarios)
    write_index(args.out_dir / "index.html", files, args.report)

    print(f"Wrote {len(files)} SVG graphs to {args.out_dir}")
    for file in files:
        print(file)
    print(args.out_dir / "evaluation_summary.csv")
    print(args.out_dir / "index.html")


if __name__ == "__main__":
    main()

