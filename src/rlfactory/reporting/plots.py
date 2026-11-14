"""Dependency-free SVG figures for run reports.

Design rules (validated palette, both modes): categorical slots 1-2 are blue/orange, used in fixed
order; single-series charts use slot 1 and no legend; bars are thin with 4 px rounded data-ends
anchored to the baseline and 2 px gaps; grid and axes are hairline and recessive; text uses ink
tokens, never series colours; every mark carries a hover tooltip on an enlarged hit target; colours
switch with ``prefers-color-scheme`` and an explicit ``data-theme`` attribute. The Markdown report
always contains the same numbers as tables (the accessible table view).
"""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

WIDTH, HEIGHT = 720, 340
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 56, 24, 64, 56
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

_LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "s1": "#2a78d6",
    "s2": "#eb6834",
}
_DARK = {
    "surface": "#1a1a19",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "s1": "#3987e5",
    "s2": "#d95926",
}


def _vars(palette: dict[str, str]) -> str:
    return " ".join(f"--{key}: {value};" for key, value in palette.items())


def _style() -> str:
    return (
        "<style>"
        f"svg.viz-root {{ color-scheme: light; {_vars(_LIGHT)} }}"
        "@media (prefers-color-scheme: dark) {"
        f"svg.viz-root:not([data-theme='light']) {{ color-scheme: dark; {_vars(_DARK)} }} }}"
        f"svg.viz-root[data-theme='dark'] {{ color-scheme: dark; {_vars(_DARK)} }}"
        f"text {{ font-family: {FONT}; }}"
        ".title { font-size: 16px; font-weight: 600; fill: var(--ink); }"
        ".subtitle { font-size: 12px; fill: var(--ink2); }"
        ".tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }"
        ".label { font-size: 12px; fill: var(--ink2); }"
        ".value { font-size: 11px; fill: var(--ink2); font-variant-numeric: tabular-nums; }"
        ".grid { stroke: var(--grid); stroke-width: 1; }"
        ".axis { stroke: var(--axis); stroke-width: 1; }"
        ".s1 { fill: var(--s1); } .s2 { fill: var(--s2); }"
        ".line-s1 { stroke: var(--s1); stroke-width: 2; fill: none; }"
        ".hit { fill: transparent; } .hit:hover + .mark { opacity: 0.85; }"
        "</style>"
    )


def _open(title: str, subtitle: str, description: str, *, height: int = HEIGHT) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="viz-root" viewBox="0 0 {WIDTH} {height}" '
            f'width="{WIDTH}" height="{height}" role="img" aria-labelledby="t d">'
        ),
        f'<title id="t">{html.escape(title)}</title><desc id="d">{html.escape(description)}</desc>',
        _style(),
        f'<rect width="{WIDTH}" height="{height}" fill="var(--surface)"/>',
        f'<text class="title" x="{MARGIN_LEFT}" y="26">{html.escape(title)}</text>',
        f'<text class="subtitle" x="{MARGIN_LEFT}" y="44">{html.escape(subtitle)}</text>',
    ]


def _top_rounded_bar(x: float, y: float, width: float, height: float) -> str:
    """Vertical bar whose data-end (top) is rounded by up to 4 px; the baseline end stays square."""
    radius = max(0.0, min(4.0, width / 2, height))
    bottom = y + height
    return (
        f"M{x:.2f},{bottom:.2f} L{x:.2f},{y + radius:.2f} "
        f"Q{x:.2f},{y:.2f} {x + radius:.2f},{y:.2f} L{x + width - radius:.2f},{y:.2f} "
        f"Q{x + width:.2f},{y:.2f} {x + width:.2f},{y + radius:.2f} L{x + width:.2f},{bottom:.2f} Z"
    )


def _right_rounded_bar(x: float, y: float, width: float, height: float) -> str:
    radius = max(0.0, min(4.0, height / 2, width))
    right = x + width
    return (
        f"M{x:.2f},{y:.2f} L{right - radius:.2f},{y:.2f} Q{right:.2f},{y:.2f} {right:.2f},{y + radius:.2f} "
        f"L{right:.2f},{y + height - radius:.2f} Q{right:.2f},{y + height:.2f} {right - radius:.2f},{y + height:.2f} "
        f"L{x:.2f},{y + height:.2f} Z"
    )


def _y_axis(lines: list[str], y_max: float, fmt: Callable[[float], str], ticks: int = 4) -> None:
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    for i in range(ticks + 1):
        value = y_max * i / ticks
        y = MARGIN_TOP + plot_height * (1 - i / ticks)
        css = "axis" if i == 0 else "grid"
        lines.append(
            f'<line class="{css}" x1="{MARGIN_LEFT}" x2="{WIDTH - MARGIN_RIGHT}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        lines.append(
            f'<text class="tick" x="{MARGIN_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">{html.escape(fmt(value))}</text>'
        )


@dataclass(frozen=True)
class Series:
    name: str
    values: Sequence[float | None]


def grouped_bar_chart(
    title: str,
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    subtitle: str = "",
    y_max: float = 1.0,
    fmt: Callable[[float], str] = lambda v: f"{v:.0%}",
) -> str:
    """Up to two series (validated slots 1-2) grouped per category; legend for >= 2 series."""
    if not 1 <= len(series) <= 2:
        raise ValueError("grouped_bar_chart supports one or two series (validated palette slots)")
    description = "; ".join(
        f"{s.name}: "
        + ", ".join(
            f"{c}={fmt(v) if v is not None else 'n/a'}"
            for c, v in zip(categories, s.values, strict=True)
        )
        for s in series
    )
    lines = _open(title, subtitle, description)
    _y_axis(lines, y_max, fmt)
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    band = plot_width / max(1, len(categories))
    group = min(band * 0.56, 44.0 * len(series))
    bar_width = (group - 2.0 * (len(series) - 1)) / len(series)
    baseline = MARGIN_TOP + plot_height
    for index, category in enumerate(categories):
        left = MARGIN_LEFT + band * index + (band - group) / 2
        for slot, entry in enumerate(series, start=1):
            value = entry.values[index]
            x = left + (slot - 1) * (bar_width + 2.0)
            label = f"{entry.name} · {category}: {fmt(value) if value is not None else 'n/a'}"
            lines.append(
                f'<g><rect class="hit" x="{x - 2:.1f}" y="{MARGIN_TOP}" width="{bar_width + 4:.1f}" '
                f'height="{plot_height}"><title>{html.escape(label)}</title></rect>'
            )
            if value is not None and value > 0:
                height = plot_height * min(value, y_max) / y_max
                lines.append(
                    f'<path class="mark s{slot}" d="{_top_rounded_bar(x, baseline - height, bar_width, height)}"/>'
                )
            lines.append("</g>")
        lines.append(
            f'<text class="label" x="{MARGIN_LEFT + band * index + band / 2:.1f}" y="{baseline + 20}" '
            f'text-anchor="middle">{html.escape(category)}</text>'
        )
    if len(series) > 1:
        x = WIDTH - MARGIN_RIGHT
        for slot, entry in reversed(list(enumerate(series, start=1))):
            width = 18 + 7 * len(entry.name)
            x -= width
            lines.append(
                f'<rect class="s{slot}" x="{x:.1f}" y="18" width="10" height="10" rx="2"/>'
            )
            lines.append(
                f'<text class="label" x="{x + 14:.1f}" y="27">{html.escape(entry.name)}</text>'
            )
            x -= 12
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def histogram_chart(
    title: str, bins: Sequence[tuple[float, float, int]], *, subtitle: str = "", x_label: str = ""
) -> str:
    """Single-series histogram (slot 1, no legend): adjacent bins separated by 2 px gaps."""
    counts = [count for _, _, count in bins]
    description = ", ".join(f"[{lo:.2f}, {hi:.2f}): {count}" for lo, hi, count in bins)
    lines = _open(title, subtitle, description or "no data")
    peak = max(counts) if counts else 1
    _y_axis(lines, float(peak), lambda v: f"{v:.0f}")
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    baseline = MARGIN_TOP + plot_height
    width = plot_width / max(1, len(bins))
    for index, (low, high, count) in enumerate(bins):
        x = MARGIN_LEFT + index * width
        lines.append(
            f'<g><rect class="hit" x="{x:.1f}" y="{MARGIN_TOP}" width="{width:.1f}" height="{plot_height}">'
            f"<title>{low:.2f} to {high:.2f}: {count}</title></rect>"
        )
        if count:
            height = plot_height * count / peak
            lines.append(
                f'<path class="mark s1" d="{_top_rounded_bar(x + 1, baseline - height, max(1.0, width - 2), height)}"/>'
            )
        lines.append("</g>")
    if bins:
        right = MARGIN_LEFT + len(bins) * width
        middle = len(bins) // 2
        for x_pos, value, anchor_side in (
            (MARGIN_LEFT, bins[0][0], "start"),
            (MARGIN_LEFT + middle * width, bins[middle][0], "middle"),
            (right, bins[-1][1], "end"),
        ):
            lines.append(
                f'<text class="tick" x="{x_pos:.1f}" y="{baseline + 16}" '
                f'text-anchor="{anchor_side}">{value:.2f}</text>'
            )
    if x_label:
        lines.append(
            f'<text class="label" x="{WIDTH / 2:.1f}" y="{HEIGHT - 12}" text-anchor="middle">{html.escape(x_label)}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def horizontal_bar_chart(
    title: str, items: Sequence[tuple[str, int]], *, subtitle: str = ""
) -> str:
    """Single-series ranked bars (slot 1) with value labels at the data end."""
    ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    description = ", ".join(f"{name}: {value}" for name, value in ordered)
    canvas_height = MARGIN_TOP + 32 * max(1, len(ordered)) + 24
    lines = _open(title, subtitle, description or "no data", height=canvas_height)
    left = 190
    plot_width = WIDTH - left - MARGIN_RIGHT - 40
    peak = max((value for _, value in ordered), default=1) or 1
    row = 32.0
    for index, (name, value) in enumerate(ordered):
        y = MARGIN_TOP + index * row
        height = row - 10
        width = plot_width * value / peak
        lines.append(
            f'<g><rect class="hit" x="{left}" y="{y:.1f}" width="{plot_width + 40}" height="{row:.1f}">'
            f"<title>{html.escape(name)}: {value}</title></rect>"
        )
        if value:
            lines.append(
                f'<path class="mark s1" d="{_right_rounded_bar(left, y, width, height)}"/>'
            )
        lines.append("</g>")
        lines.append(
            f'<text class="label" x="{left - 8}" y="{y + height / 2 + 4:.1f}" text-anchor="end">{html.escape(name)}</text>'
        )
        lines.append(
            f'<text class="value" x="{left + width + 6:.1f}" y="{y + height / 2 + 4:.1f}">{value}</text>'
        )
    lines.append(
        f'<line class="axis" x1="{left}" x2="{left}" y1="{MARGIN_TOP - 4}" y2="{MARGIN_TOP + row * len(ordered):.1f}"/>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def line_chart(
    title: str,
    labels: Sequence[str],
    values: Sequence[float | None],
    *,
    subtitle: str = "",
    y_max: float = 1.0,
    fmt: Callable[[float], str] = lambda v: f"{v:.0%}",
) -> str:
    """Single-series line (slot 1, 2 px) with 8 px markers ringed by the surface colour."""
    description = ", ".join(
        f"{label}: {fmt(v) if v is not None else 'n/a'}"
        for label, v in zip(labels, values, strict=True)
    )
    lines = _open(title, subtitle, description)
    _y_axis(lines, y_max, fmt)
    plot_width = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    inset = 28.0
    step = (plot_width - 2 * inset) / max(1, len(labels) - 1) if len(labels) > 1 else 0.0
    points = []
    for index, value in enumerate(values):
        x = MARGIN_LEFT + (inset + index * step if len(labels) > 1 else plot_width / 2)
        y = (
            MARGIN_TOP + plot_height * (1 - min(value, y_max) / y_max)
            if value is not None
            else None
        )
        points.append((x, y, value, labels[index]))
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}"
        for i, (x, y, _, _) in enumerate(p for p in points if p[1] is not None)
    )
    if path:
        lines.append(f'<path class="line-s1" d="{path}"/>')
    for x, y, value, label in points:
        lines.append(
            f'<text class="label" x="{x:.1f}" y="{MARGIN_TOP + plot_height + 20}" text-anchor="middle">{html.escape(label)}</text>'
        )
        if y is None or value is None:
            continue
        lines.append(
            f'<g><circle class="hit" cx="{x:.1f}" cy="{y:.1f}" r="14"><title>{html.escape(label)}: {fmt(value)}</title></circle>'
            f'<circle class="mark s1" cx="{x:.1f}" cy="{y:.1f}" r="4" stroke="var(--surface)" stroke-width="2"/></g>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_svg(path: Path, svg: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")
    return path
