#!/usr/bin/env python3
"""Builds dist/index.html from docs/*.md, src/* and model/model.js.

Run with any Python that has the `markdown` package:  python build.py
"""
import html
import pathlib
import re

import markdown

ROOT = pathlib.Path(__file__).resolve().parent
DOCS = ROOT.parent / "docs"
SRC = ROOT / "src"

# id, doc file, short nav label, floorplan area, color group, section eyebrow
CHAPTERS = [
    ("insights", "00_insights", "Core insights", "ins", "g-r", "Start here"),
    ("frame", "01_frame", "The frame", "frm", "g-m", "Foundations"),
    ("physics", "02_physics", "Area, clock, power", "phy", "g-m", "Foundations"),
    ("systolic", "03_systolic", "Systolic arrays", "sys", "g-c", "The compute core"),
    ("shape", "04_shape", "Matmul shape", "shp", "g-c", "The compute core"),
    ("reduction", "05_reduction", "Reduction trees", "red", "g-c", "The compute core"),
    ("fabric", "06_fabric", "Crossbars, NoC", "fab", "g-f", "Feeding the array"),
    ("memory", "07_memory", "Memory", "mem", "g-f", "Feeding the array"),
    ("control", "08_control", "Control and feeding", "ctl", "g-f", "Feeding the array"),
    ("vector", "09_vector", "Vector units", "vec", "g-f", "Feeding the array"),
    ("workloads", "10_workloads", "Workloads", "wrk", "g-s", "Systems"),
    ("scale", "11_scale", "Scale-out", "scl", "g-s", "Systems"),
    ("generations", "12_generations", "Generations", "gen", "g-s", "History"),
    ("method", "13_method", "Method", "mth", "g-m", "Practice"),
    ("yourchip", "14_yourchip", "Your chip", "you", "g-r", "Practice"),
    ("nonlinear", "18_nonlinear", "LUTs and MoE", "non", "g-r", "Practice"),
    ("judgement", "15_judgement", "Judgement", "jud", "g-m", "Practice"),
    ("atlas", "16_atlas", "Atlas", "atl", "g-m", "Reference"),
    ("audit", "17_audit", "Audit", "aud", "g-m", "Reference"),
]
TITLES = {
    "Systolic arrays": "systolic", "Matmul shape": "shape", "The frame": "frame", "Area, clock, power": "physics",
    "Reduction trees": "reduction", "Crossbars and NoC": "fabric", "Memory": "memory", "Control and feeding": "control",
    "Vector units": "vector", "Workloads": "workloads", "Scale-out": "scale", "Generations": "generations",
    "Crossbars, NoC": "fabric", "Your chip": "yourchip", "Nonlinear units and MoE": "nonlinear", "Judgement": "judgement", "Method": "method", "Source audit": "audit",
}
SOURCES = {
    "J": "Jouppi et al., Ironwood paper", "R": "Roune, Designing AI Chip Hardware and Software",
    "IA": "Irrational Analysis, unfinished draft", "E": "Component Encyclopedia", "F": "Your factbook in research/",
    "D": "Derived here and tested in model/", "O": "Outside the four PDFs: general engineering knowledge",
}
TAG = re.compile(r"\[((?:J|R|IA|E|F|D|O)(?:, (?:J|R|IA|E|F|D|O))*)\]")


def chips(match):
    return "".join(
        f'<span class="src src-{t}" title="{html.escape(SOURCES[t])}">{t}</span>' for t in match.group(1).split(", ")
    )


def render(cid, doc):
    text = (DOCS / f"{doc}.md").read_text()
    text = re.sub(r"^::viz (\w+)\s*$", lambda m: f'<div data-lab="{m.group(1)}"></div>', text, flags=re.M)
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    body = re.sub(r"<strong>\[((?:J|R|IA|E|F|D|O))\]</strong>", lambda m: f"[{m.group(1)}]", body)
    # Source tags stay out of code spans.
    parts = re.split(r"(<code>.*?</code>)", body, flags=re.S)
    body = "".join(p if p.startswith("<code>") else TAG.sub(chips, p) for p in parts)
    body = re.sub(r"→ <em>(.+?)</em>", lambda m: f'→ <a href="#{TITLES.get(m.group(1), "")}">{m.group(1)}</a>', body)
    body = re.sub(r"<em>([^<]+?)</em>", lambda m: f'<a href="#{TITLES[m.group(1)]}">{m.group(1)}</a>' if m.group(1) in TITLES else m.group(0), body)
    body = re.sub(r"<p><code>(.*?)</code></p>", r'<p class="formula"><code>\1</code></p>', body, flags=re.S)
    body = body.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
    return body


def main():
    arts, nav = [], []
    for i, (cid, doc, label, area, group, eyebrow) in enumerate(CHAPTERS):
        body = render(cid, doc)
        h1 = re.search(r"<h1>(.*?)</h1>", body).group(0)
        body = body.replace(h1, "", 1)
        hero = ""
        if cid == "insights":
            h1 = "<h1>Keep the array busy</h1>"
            hero = ('<div class="hero-wrap"><div class="stage"></div><div class="hero-cap">A 16×16 output-stationary array. '
                    'Blue and red operands enter skewed, the amber wavefront crosses the die, and green accumulators fill.</div></div>'
                    '<h2 style="margin-top:0">The core insights</h2>')
        prev = CHAPTERS[i - 1] if i else None
        nxt = CHAPTERS[i + 1] if i + 1 < len(CHAPTERS) else None
        pager = '<nav class="pager" aria-label="Chapter navigation">'
        if prev:
            pager += f'<a href="#{prev[0]}"><span>Previous</span>{prev[2]}</a>'
        if nxt:
            pager += f'<a class="next" href="#{nxt[0]}"><span>Next</span>{nxt[2]}</a>'
        pager += "</nav>"
        arts.append(f'<article class="chapter" id="{cid}"{" hidden" if i else ""}><p class="eyebrow">{eyebrow}</p>{h1}{hero}{body}{pager}</article>')
        nav.append(f'<a class="{group}" style="grid-area:{area}" href="#{cid}">{label}</a>')
    options = "".join(f'<option value="{c[0]}">{c[2]}</option>' for c in CHAPTERS)
    legend = "".join(f"<div><b>{k}</b> {v}</div>" for k, v in {
        "J": "Ironwood paper", "R": "Roune", "IA": "Irrational Analysis", "E": "Encyclopedia", "F": "Your factbook", "D": "Derived, tested", "O": "Outside the PDFs"}.items())
    style = (SRC / "style.css").read_text()
    scripts = "\n".join((p).read_text() for p in [ROOT.parent / "model" / "model.js", SRC / "core.js", SRC / "labs1.js", SRC / "labs2.js", SRC / "app.js"])
    page = f"""<meta charset="utf-8"><title>Keep the Array Busy</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@100..125,400..800&family=JetBrains+Mono:wght@500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,650;1,8..60,400&display=swap">
<style>{style}</style>
<div class="shell">
<aside class="rail">
<a class="brand" href="#insights">Keep the Array Busy<small>A deep dive on AI chip architecture</small></a>
<nav class="floorplan" aria-label="Chapters, laid out as a die floorplan">{''.join(nav)}</nav>
<div class="mobilebar"><label class="keyline" for="chapter-select">Chapter</label><select id="chapter-select">{options}</select></div>
<div class="legend">{legend}</div>
</aside>
<main>{''.join(arts)}</main>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>{scripts}</script>
"""
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    # index.html is the standalone copy. The artifact file is wrapped by the publish skeleton.
    head = '<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    split = page.index('<div class="shell">')
    (out / "index.html").write_text(head + page[:split] + "</head><body>" + page[split:] + "</body></html>")
    (out / "keep-the-array-busy.html").write_text(page)
    words = sum(len((DOCS / f"{c[1]}.md").read_text().split()) for c in CHAPTERS)
    print(f"built {out / 'index.html'}: {len(page) / 1024:.0f} KB, {len(CHAPTERS)} chapters, {words} words")


if __name__ == "__main__":
    main()
