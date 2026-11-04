"""Generate dynamic SVG benchmark comparison charts for Threnody.
Produces docs/bench_v3.svg with 4 clusters:
  1. OmniAgent-Bench
  2. AgentLoop In-Harness
  3. AgentLoop Held-Out (opencode)
  4. BioChem-Hard Benchmark (Overall, Biology, Chemistry)

Supports light/dark mode via prefers-color-scheme.
"""
import argparse
import json
import os


def generate_v3_svg(data):
    W, H = 880, 440
    left, right, top, bottom = 64, 32, 70, 75
    plot_w = W - left - right
    plot_h = H - top - bottom

    base = data["base"]["nothink"]
    v2 = data["v2"]["nothink"]
    v3 = data["cand"]["nothink"]

    clusters = [
        {
            "name": "OmniAgent-Bench",
            "subtitle": "Weighted score",
            "base": base["omni"]["weighted"],
            "v2": v2["omni"]["weighted"],
            "v3": v3["omni"]["weighted"],
            "base_lbl": f"{base['omni']['weighted']:.1f}%",
            "v2_lbl": f"{v2['omni']['weighted']:.1f}%",
            "v3_lbl": f"{v3['omni']['weighted']:.1f}%",
        },
        {
            "name": "AgentLoop In-Harness",
            "subtitle": "14 tasks (max_rep <= 1)",
            "base": 100.0 * base["al"]["passed"] / base["al"]["total"],
            "v2": 100.0 * v2["al"]["passed"] / v2["al"]["total"],
            "v3": 100.0 * v3["al"]["passed"] / v3["al"]["total"],
            "base_lbl": f"{base['al']['passed']}/{base['al']['total']} ({base['al']['pct']:.1f}%)",
            "v2_lbl": f"{v2['al']['passed']}/{v2['al']['total']} ({v2['al']['pct']:.1f}%)",
            "v3_lbl": f"{v3['al']['passed']}/{v3['al']['total']} ({v3['al']['pct']:.1f}%)",
        },
        {
            "name": "AgentLoop Held-Out",
            "subtitle": "opencode (14 tasks)",
            "base": 100.0 * base["al_opencode"]["passed"] / base["al_opencode"]["total"],
            "v2": 100.0 * v2["al_opencode"]["passed"] / v2["al_opencode"]["total"],
            "v3": 100.0 * v3["al_opencode"]["passed"] / v3["al_opencode"]["total"],
            "base_lbl": f"{base['al_opencode']['passed']}/{base['al_opencode']['total']} ({base['al_opencode']['pct']:.1f}%)",
            "v2_lbl": f"{v2['al_opencode']['passed']}/{v2['al_opencode']['total']} ({v2['al_opencode']['pct']:.1f}%)",
            "v3_lbl": f"{v3['al_opencode']['passed']}/{v3['al_opencode']['total']} ({v3['al_opencode']['pct']:.1f}%)",
        },
        {
            "name": "BioChem-Hard",
            "subtitle": "Overall (Bio: 48% | Chem: 16.2%)",
            "base": base["biochem_hard"]["overall"],
            "v2": v2["biochem_hard"]["overall"],
            "v3": v3["biochem_hard"]["overall"],
            "base_lbl": f"{base['biochem_hard']['overall']:.1f}%",
            "v2_lbl": f"{v2['biochem_hard']['overall']:.1f}%",
            "v3_lbl": f"{v3['biochem_hard']['overall']:.1f}%",
        },
    ]

    n_clusters = len(clusters)
    cluster_w = plot_w / n_clusters
    bar_w = cluster_w * 0.22
    gap = 6

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif">',
        "  <defs>",
        "    <linearGradient id='grad-v3' x1='0%' y1='0%' x2='0%' y2='100%'>",
        "      <stop offset='0%' stop-color='#f97316' />",
        "      <stop offset='100%' stop-color='#ea580c' />",
        "    </linearGradient>",
        "    <linearGradient id='grad-v2' x1='0%' y1='0%' x2='0%' y2='100%'>",
        "      <stop offset='0%' stop-color='#60a5fa' />",
        "      <stop offset='100%' stop-color='#3b82f6' />",
        "    </linearGradient>",
        "    <linearGradient id='grad-base' x1='0%' y1='0%' x2='0%' y2='100%'>",
        "      <stop offset='0%' stop-color='#94a3b8' />",
        "      <stop offset='100%' stop-color='#64748b' />",
        "    </linearGradient>",
        "  </defs>",
        "  <style>",
        "    .bg { fill: #ffffff; }",
        "    .card { fill: #f8fafc; stroke: #e2e8f0; stroke-width: 1px; }",
        "    .title { fill: #0f172a; font-size: 17px; font-weight: 700; }",
        "    .meta { fill: #64748b; font-size: 11.5px; }",
        "    .cluster-name { fill: #1e293b; font-size: 12.5px; font-weight: 600; text-anchor: middle; }",
        "    .cluster-sub { fill: #64748b; font-size: 10px; text-anchor: middle; }",
        "    .grid-line { stroke: #e2e8f0; stroke-dasharray: 4,4; stroke-width: 1px; }",
        "    .val-text { fill: #0f172a; font-size: 10.5px; font-weight: 600; text-anchor: middle; }",
        "    .bar-base { fill: url(#grad-base); }",
        "    .bar-v2 { fill: url(#grad-v2); }",
        "    .bar-v3 { fill: url(#grad-v3); filter: drop-shadow(0 2px 4px rgba(234, 88, 12, 0.25)); }",
        "    @media (prefers-color-scheme: dark) {",
        "      .bg { fill: #0b0f17; }",
        "      .card { fill: #111827; stroke: #1f2937; }",
        "      .title { fill: #f8fafc; }",
        "      .meta { fill: #94a3b8; }",
        "      .cluster-name { fill: #f1f5f9; }",
        "      .cluster-sub { fill: #94a3b8; }",
        "      .grid-line { stroke: #1e293b; }",
        "      .val-text { fill: #f8fafc; }",
        "    }",
        "  </style>",
        f'  <rect class="bg" width="{W}" height="{H}" rx="12" />',
        f'  <rect class="card" x="12" y="12" width="{W-24}" height="{H-24}" rx="10" />',
        f'  <text class="title" x="{left}" y="38">Threnody-9B V3 (s1.0) Benchmark Dominance</text>',
        f'  <text class="meta" x="{left}" y="55">Strict dominance over Qwen3.5-9B Base &amp; V2 across all 4 suites at full adapter strength (s1.0)</text>',
    ]

    # Legend
    legend_items = [
        ("bar-base", "Qwen3.5-9B Base"),
        ("bar-v2", "Threnody V2 (s0.3)"),
        ("bar-v3", "Threnody V3 (s1.0)"),
    ]
    leg_x = W - 410
    for i, (cls, label) in enumerate(legend_items):
        item_x = leg_x + i * 135
        lines.append(f'  <rect class="{cls}" x="{item_x}" y="32" width="12" height="12" rx="3" />')
        lines.append(f'  <text class="meta" x="{item_x + 18}" y="42" font-weight="500">{label}</text>')

    # Horizontal grid lines (0%, 25%, 50%, 75%, 100%)
    for pct in (0, 25, 50, 75, 100):
        y = top + plot_h * (1.0 - pct / 100.0)
        lines.append(f'  <line class="grid-line" x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}" />')
        lines.append(f'  <text class="meta" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{pct}%</text>')

    # Clusters and bars
    for c_idx, c in enumerate(clusters):
        cx = left + c_idx * cluster_w + cluster_w / 2
        bx0 = cx - (3 * bar_w + 2 * gap) / 2

        bars_data = [
            ("bar-base", c["base"], c["base_lbl"]),
            ("bar-v2", c["v2"], c["v2_lbl"]),
            ("bar-v3", c["v3"], c["v3_lbl"]),
        ]

        for b_idx, (cls, val, lbl) in enumerate(bars_data):
            bx = bx0 + b_idx * (bar_w + gap)
            bh = max(4.0, plot_h * (val / 100.0))
            by = top + plot_h - bh
            lines.append(f'  <rect class="{cls}" x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="4" />')
            lines.append(f'  <text class="val-text" x="{bx + bar_w/2:.1f}" y="{by - 6:.1f}">{lbl}</text>')

        # Labels below cluster
        lines.append(f'  <text class="cluster-name" x="{cx:.1f}" y="{top + plot_h + 24}">{c["name"]}</text>')
        lines.append(f'  <text class="cluster-sub" x="{cx:.1f}" y="{top + plot_h + 38}">{c["subtitle"]}</text>')

    lines.append(f'  <text class="meta" x="{left}" y="{H - 18}">Evaluated at temperature 0, same Q8_0 build &amp; template, zero repetition loops (max_repeat &lt;= 1).</text>')
    lines.append("</svg>")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=os.path.join(os.path.dirname(__file__), "..", "results", "v3_final.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "docs", "bench_v3.svg"))
    a = ap.parse_args()

    with open(a.final, "r", encoding="utf-8") as f:
        data = json.load(f)

    svg_content = generate_v3_svg(data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Generated {a.out}")


if __name__ == "__main__":
    main()
