#!/usr/bin/env python3
"""Render a notes .md file with LaTeX math ($...$, $$...$$) to a standalone .html (KaTeX in the browser).

usage: python md2html.py NOTE.md [NOTE2.md ...]   -> writes NOTE.html next to each input
Needs python-markdown (in the kl-nn env). The page loads KaTeX from cdn.jsdelivr.net, so open it online.
"""
import html
import re
import sys
from pathlib import Path

import markdown

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist"
PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{katex}/katex.min.css">
<script defer src="{katex}/katex.min.js"></script>
<script defer src="{katex}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}},
  {{left: '$', right: '$', display: false}}], throwOnError: false}});"></script>
<style>
body {{ max-width: 860px; margin: 2em auto; padding: 0 16px; font: 16px/1.6 Georgia, serif; color: #1d2630; }}
code {{ font: 0.9em Menlo, monospace; background: #f1f3f5; padding: 1px 4px; border-radius: 3px; }}
table {{ border-collapse: collapse; }} th, td {{ border: 1px solid #ccd3da; padding: 4px 8px; }}
h1, h2, h3 {{ font-family: Helvetica, Arial, sans-serif; }}
.katex-display {{ overflow-x: auto; overflow-y: hidden; }}
</style></head><body>
{body}
</body></html>
"""


def convert(path: Path) -> Path:
    text = path.read_text()
    stash = []

    def keep(m):
        stash.append(m.group(0))
        return f"MATHSTASH{len(stash) - 1}X"

    # protect code spans first, then display and inline math, from the markdown parser
    code = []
    text = re.sub(r"`[^`\n]+`", lambda m: (code.append(m.group(0)), f"CODESTASH{len(code) - 1}X")[1], text)
    text = re.sub(r"\$\$.+?\$\$", keep, text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", keep, text)
    text = re.sub(r"CODESTASH(\d+)X", lambda m: code[int(m.group(1))], text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    body = re.sub(r"MATHSTASH(\d+)X", lambda m: html.escape(stash[int(m.group(1))], quote=False), body)
    title = next((l.lstrip("# ").strip() for l in path.read_text().splitlines() if l.startswith("# ")), path.stem)
    out = path.with_suffix(".html")
    out.write_text(PAGE.format(title=html.escape(title), katex=KATEX, body=body))
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(convert(Path(arg)))
