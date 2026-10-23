"""
Shared page helpers for the post (inline-SVG line chart, legend, table, page wrapper).
Adapted from the build script of the companion post "Why Rotary Embeddings Stop Extrapolating".
"""
import html
import json
import os
import re

W, H = 680, 300
ML, MR, MT, MB = 52, 128, 14, 44          # margins; the right margin holds direct labels


def _x(v, lo, hi):
    return ML + (v - lo) / (hi - lo) * (W - ML - MR)


def _y(v, lo, hi):
    return MT + (1 - (v - lo) / (hi - lo)) * (H - MT - MB)


def line_chart(series, xdom, ydom, xticks, yticks, xlabel, ylabel, aria, shade=(), vline=None, ytickfmt=None,
               xtickfmt=None, hline=None):
    """series: dicts with name, xs, ys, var (css colour variable), dash ('' / '6 4' / '2 3'), width,
    optional lo/hi (min-max band), label (direct label text, at the last point)."""
    x0, x1 = xdom
    y0, y1 = ydom
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(aria)}">']
    for a, b, txt in shade:
        xa, xb = _x(max(a, x0), x0, x1), _x(min(b, x1), x0, x1)
        out.append(f'<rect x="{xa:.1f}" y="{MT}" width="{xb - xa:.1f}" height="{H - MT - MB}" class="shade"/>')
        if txt:
            out.append(f'<text x="{xa + 4:.1f}" y="{MT + 12}" class="tick shade-label">{html.escape(txt)}</text>')
    for t in yticks:
        y = _y(t, y0, y1)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{ML - 8}" y="{y + 4:.1f}" class="tick" text-anchor="end">{(ytickfmt or str)(t)}</text>')
    for t in xticks:
        x = _x(t, x0, x1)
        out.append(f'<text x="{x:.1f}" y="{H - MB + 18}" class="tick" text-anchor="middle">{(xtickfmt or str)(t)}</text>')
    out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{H - MB}" y2="{H - MB}" class="axis"/>')
    out.append(f'<text x="{(ML + W - MR) / 2:.1f}" y="{H - 6}" class="axis-label" text-anchor="middle">{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{(MT + H - MB) / 2:.1f}" class="axis-label" text-anchor="middle" '
               f'transform="rotate(-90 14 {(MT + H - MB) / 2:.1f})">{html.escape(ylabel)}</text>')
    if hline is not None:
        y = _y(hline, y0, y1)
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
    if vline is not None:
        x = _x(vline, x0, x1)
        out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{H - MB}" class="ref"/>')
    for s in series:
        if 'lo' in s:
            pts = [(_x(a, x0, x1), _y(b, y0, y1)) for a, b in zip(s['xs'], s['hi'])]
            pts += [(_x(a, x0, x1), _y(b, y0, y1)) for a, b in reversed(list(zip(s['xs'], s['lo'])))]
            out.append(f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" '
                       f'style="fill: var({s["var"]}); opacity: 0.12"/>')
    labels = []
    for s in series:
        pts = [(_x(a, x0, x1), _y(min(max(b, y0), y1), y0, y1)) for a, b in zip(s['xs'], s['ys'])]
        d = 'M' + ' L'.join(f'{a:.1f},{b:.1f}' for a, b in pts)
        dash = f'; stroke-dasharray: {s["dash"]}' if s.get('dash') else ''
        out.append(f'<path d="{d}" class="series" style="stroke: var({s["var"]}); stroke-width: {s.get("width", 2)}{dash}"/>')
        if s.get('markers'):
            for (a, b), xv, yv in zip(pts, s['xs'], s['ys']):
                out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" class="dot" style="fill: var({s["var"]})"/>')
        yfmt = ytickfmt or (lambda v: f'{v:.2f}')
        xfmt = xtickfmt or (lambda v: f'{v:g}')
        if len(pts) <= 40:                                    # hover targets with native tooltips
            for (a, b), xv, yv in zip(pts, s['xs'], s['ys']):
                out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="7" class="hit"><title>'
                           f'{html.escape(s["name"])}: {xfmt(xv)} &#8594; {yfmt(yv)}</title></circle>')
        if s.get('label'):
            labels.append([pts[-1][1], s['label'], s['var']])
    labels.sort()
    for i in range(1, len(labels)):                       # nudge direct labels apart (downwards) ...
        if labels[i][0] - labels[i - 1][0] < 14:
            labels[i][0] = labels[i - 1][0] + 14
    floor = H - MB - 2                                    # ... then shift the stack up if it ran past the axis
    if labels and labels[-1][0] > floor:
        labels[-1][0] = floor
        for i in range(len(labels) - 2, -1, -1):
            labels[i][0] = min(labels[i][0], labels[i + 1][0] - 14)
    for y, txt, var in labels:
        out.append(f'<text x="{W - MR + 8}" y="{y + 4:.1f}" class="direct">{html.escape(txt)}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def legend(items):
    """items: (label, css var, dash) -> small HTML legend with line swatches."""
    parts = []
    for label, var, dash in items:
        style = f'border-top: 2px {"dashed" if dash else "solid"} var({var})'
        parts.append(f'<span class="lg"><span class="sw" style="{style}"></span>{html.escape(label)}</span>')
    return '<div class="legend">' + ''.join(parts) + '</div>'


def table(caption, head, rows, cls=None):
    th = ''.join(f'<th>{h}</th>' for h in head)
    body = ''
    for r in rows:
        body += '<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>'
    return (f'<div class="table-wrap"><table{f" class=\"{cls}\"" if cls else ""}><caption>{caption}</caption>'
            f'<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>')


def pct(v):
    return f'{round(100 * v)}%'


def f2(v):
    """1.00 / .93 / 0 style, as in the result tables."""
    if v == 0:
        return '0'
    if v < 0.005:
        return '&lt;.01'
    s = f'{v:.2f}'
    return s[1:] if s.startswith('0.') else s


def parse_reach_log(path):
    """probe_length.py text output -> {run: {L: acc}} (later rows for the same run win)."""
    out, lengths = {}, None
    for line in open(path, encoding='utf-8'):
        if line.startswith('run '):
            lengths = [int(x) for x in line.split()[1:]]
        elif lengths and re.match(r'^\S+\s+[0-9.]+', line):
            parts = line.split()
            out.setdefault(parts[0], {}).update({L: float(v) for L, v in zip(lengths, parts[1:])})
    return out


EXTRA_CSS = '''
  :root { --s1: #00849c; --s2: #c4501c; --ctx: #8a8d92; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --s1: #1fa3ae; --s2: #d9703e; --ctx: #7d7f85; } }
  :root[data-theme="dark"] { --s1: #1fa3ae; --s2: #d9703e; --ctx: #7d7f85; }
  svg.chart { width: 100%; height: auto; display: block; margin: 0.4rem 0 1.4rem; overflow: visible; }
  svg.chart .grid { stroke: var(--line-soft); stroke-width: 1; }
  svg.chart .axis { stroke: var(--line); stroke-width: 1; }
  svg.chart .ref { stroke: var(--ink-faint); stroke-width: 1; stroke-dasharray: 3 3; }
  svg.chart .shade { fill: var(--ink-faint); opacity: 0.10; }
  svg.chart .tick { fill: var(--ink-faint); font: 11px 'JetBrains Mono', monospace; }
  svg.chart .shade-label { fill: var(--ink-soft); }
  svg.chart .axis-label { fill: var(--ink-soft); font: 12px 'Newsreader', serif; }
  svg.chart .direct { fill: var(--ink-soft); font: 12px 'Newsreader', serif; }
  svg.chart .series { fill: none; stroke-linejoin: round; stroke-linecap: round; }
  svg.chart .dot { stroke: var(--surface); stroke-width: 2; }
  svg.chart .hit { fill: transparent; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.85rem; color: var(--ink-soft); margin-top: 0.6rem; }
  .legend .lg { display: inline-flex; align-items: center; gap: 6px; }
  .legend .sw { display: inline-block; width: 18px; height: 0; }
  .fig-cap { font-size: 0.88rem; color: var(--ink-soft); margin: -0.6rem 0 1.6rem; }
  ol.steps > li { margin-bottom: 0.9rem; }
'''


def page(title, description, body, base_css):
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{html.escape(description)}">
<style>
{base_css}
{EXTRA_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
'''
