#!/usr/bin/env python3
"""Convert the Chinese strategy guide markdown to a styled PDF."""

import re
from pathlib import Path

import markdown
from weasyprint import HTML

SRC = Path("/home/user/analysis/docs/strategy_guide_zh.md")
OUT = Path("/home/user/analysis/docs/strategy_guide_zh.pdf")

text = SRC.read_text(encoding="utf-8")

# Render markdown -> HTML with tables, fenced code, TOC, footnotes
md = markdown.Markdown(extensions=[
    "tables", "fenced_code", "codehilite", "toc", "sane_lists", "attr_list",
], extension_configs={
    "codehilite": {"guess_lang": False, "noclasses": True},
})
body = md.convert(text)

CSS = """
@page {
  size: A4;
  margin: 1.8cm 1.6cm 2cm 1.6cm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "WenQuanYi Zen Hei", sans-serif;
    font-size: 9px; color: #888;
  }
}
* { box-sizing: border-box; }
body {
  font-family: "WenQuanYi Zen Hei", "DejaVu Sans", sans-serif;
  font-size: 10.5px; line-height: 1.65; color: #1a1a1a;
}
h1 {
  font-size: 21px; color: #0b3d66; margin: 0 0 0.4em;
  padding-bottom: 0.25em; border-bottom: 3px solid #0b3d66;
  page-break-after: avoid;
}
h2 {
  font-size: 16px; color: #0b3d66; margin: 1.3em 0 0.5em;
  padding: 0.2em 0 0.2em 0.4em; border-left: 5px solid #2a7ab0;
  background: #eef4f9; page-break-after: avoid;
}
h3 {
  font-size: 13px; color: #134a73; margin: 1.1em 0 0.4em;
  page-break-after: avoid;
}
p { margin: 0.45em 0; text-align: justify; }
strong { color: #8a1f1f; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.4em 0; }
ul, ol { margin: 0.4em 0; padding-left: 1.5em; }
li { margin: 0.2em 0; }
code {
  font-family: "WenQuanYi Zen Hei Mono", "DejaVu Sans Mono", monospace;
  background: #f2f4f6; padding: 1px 4px; border-radius: 3px;
  font-size: 9.5px; color: #b5402a;
}
pre {
  background: #1e2733; color: #e6edf3; padding: 0.8em 1em;
  border-radius: 6px; overflow-x: auto; line-height: 1.45;
  page-break-inside: avoid; margin: 0.6em 0;
}
pre code {
  background: transparent; color: inherit; padding: 0;
  font-size: 9px; color: #e6edf3;
}
blockquote {
  margin: 0.6em 0; padding: 0.4em 0.9em;
  background: #fff7e6; border-left: 4px solid #e0a93b; color: #5c4612;
}
blockquote p { margin: 0.25em 0; }
table {
  border-collapse: collapse; width: 100%; margin: 0.7em 0;
  font-size: 9.5px; page-break-inside: avoid;
}
th {
  background: #0b3d66; color: #fff; padding: 6px 8px;
  text-align: left; border: 1px solid #0b3d66;
}
td {
  padding: 5px 8px; border: 1px solid #cdd6df; vertical-align: top;
}
tr:nth-child(even) td { background: #f4f7fa; }
"""

html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<style>{CSS}</style></head>
<body>{body}</body></html>"""

# keep a copy of the HTML for debugging / reuse
Path("/tmp/strategy_guide.html").write_text(html, encoding="utf-8")

HTML(string=html).write_pdf(str(OUT))
size_kb = OUT.stat().st_size / 1024
print(f"PDF written: {OUT} ({size_kb:.0f} KB)")
