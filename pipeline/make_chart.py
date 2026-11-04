"""Grouped SVG bar chart (base vs Threnody V2) from eval_v2/final.json -> docs/bench_v2.svg.
All metrics shown as percentages; light/dark aware via prefers-color-scheme.

    python make_chart.py --final final.json --out docs/bench_v2.svg
"""
import argparse
import json


def metrics(r):
    out = [("AgentLoop", 100 * r["al"]["passed"] / r["al"]["total"])]
    if "al_opencode" in r:
        out.append(("Held-out harness", 100 * r["al_opencode"]["passed"] / r["al_opencode"]["total"]))
    if "hard" in r:
        out.append(("AgentLoop-Hard", 100 * r["hard"]["passed"] / r["hard"]["total"]))
    if "omni" in r:
        out.append(("OmniAgent-Bench", r["omni"]["weighted"]))
    out.append(("MMLU-Pro bio+chem", r["sci"]["overall"]))
    return out


def svg(base, cand, title):
    mc = dict(metrics(cand))
    mb = [(k, v) for k, v in metrics(base) if k in mc]  # only metrics measured for both models
    mc = [(k, mc[k]) for k, _ in mb]
    W, H, left, top, bottom = 760, 380, 56, 56, 64
    ph = H - top - bottom
    gw = (W - left - 24) / len(mb)
    bw = gw * 0.32
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui,-apple-system,Segoe UI,sans-serif">',
         "<style>.bg{fill:#fff}.t{fill:#1a1a1a}.m{fill:#666}.g{stroke:#e5e5e5}.b{fill:#9aa0a6}.c{fill:#c2410c}"
         "@media (prefers-color-scheme:dark){.bg{fill:#0d1117}.t{fill:#e6edf3}.m{fill:#9da7b3}.g{stroke:#30363d}.b{fill:#6e7681}.c{fill:#f97316}}</style>",
         f'<rect class="bg" width="{W}" height="{H}" rx="8"/>',
         f'<text class="t" x="{left}" y="28" font-size="16" font-weight="600">{title}</text>',
         f'<rect class="b" x="{W - 250}" y="16" width="12" height="12" rx="2"/><text class="m" x="{W - 232}" y="27" font-size="12">Qwen3.5-9B base</text>',
         f'<rect class="c" x="{W - 120}" y="16" width="12" height="12" rx="2"/><text class="m" x="{W - 102}" y="27" font-size="12">Threnody V2</text>']
    for v in (0, 25, 50, 75, 100):
        y = top + ph * (1 - v / 100)
        s.append(f'<line class="g" x1="{left}" x2="{W - 24}" y1="{y:.1f}" y2="{y:.1f}"/>'
                 f'<text class="m" x="{left - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end">{v}%</text>')
    for i, ((name, vb), (_, vc)) in enumerate(zip(mb, mc)):
        x0 = left + i * gw + gw * 0.16
        for j, (v, cls) in enumerate(((vb, "b"), (vc, "c"))):
            h = ph * v / 100
            x = x0 + j * (bw + 4)
            s.append(f'<rect class="{cls}" x="{x:.1f}" y="{top + ph - h:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2"/>'
                     f'<text class="t" x="{x + bw / 2:.1f}" y="{top + ph - h - 5:.1f}" font-size="11" text-anchor="middle">{v:.1f}</text>')
        s.append(f'<text class="m" x="{x0 + bw + 2:.1f}" y="{H - bottom + 20}" font-size="12" text-anchor="middle">{name}</text>')
    s.append(f'<text class="m" x="{left}" y="{H - 14}" font-size="11">Q8_0, same build and chat template for both models, temperature 0, non-thinking mode.</text>')
    s.append("</svg>")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = json.load(open(a.final))
    open(a.out, "w", encoding="utf-8").write(svg(f["base"]["nothink"], f["cand"]["nothink"], "Threnody-9B V2 vs. Qwen3.5-9B"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
