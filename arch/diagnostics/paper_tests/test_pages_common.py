"""Shared helpers for the explanatory per-test pages of the paper-tests campaign.

Each content module (pages_npe.py, pages_shear.py, pages_stencil_stack.py)
defines a list PAGES of dicts built with ``page(...)``. ``write_test_pages.py``
imports them and calls ``render`` for each page.

Rules for content modules:
- Every number shown must come from a JSON lookup via ``J(...)`` (raises if a
  key is missing) and be formatted with ``f(...)`` / ``pm(...)``; never retype
  numbers by hand.
- Figures are embedded with ``fig(path, caption)`` (base64 data URI).
- Math: MathJax, inline ``\\( ... \\)``, display ``\\[ ... \\]``. In Python
  strings use raw strings (r"...") so backslashes survive.
- Headings use descriptive names; the old internal code goes in ``code=``.
"""
from __future__ import annotations

import base64
import html
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path("/ocean/projects/phy250048p/shared/reports/paper-tests")
TESTS_DIR = ROOT / "tests"
SCRIPTS = "/jet/home/xwang30/kl-nn/arch/diagnostics/paper_tests"
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"

JSON_FILES = {
    "npe": ROOT / "A_npe/report.json",
    "shear": ROOT / "A_shear/results.json",
    "shear_report": ROOT / "A_shear/report.json",
    "supp": ROOT / "A_shear/supplement.json",
    "supp2": ROOT / "A_shear/supplement2.json",
    "stencil": ROOT / "B_inference/report.json",
    "stencil_xu3": ROOT / "B_inference/t2b_xu3like.json",
    "stencil_prop": ROOT / "B_inference/t2b_fibrepair.json",
    "stack": ROOT / "C_stacking/report.json",
}


@lru_cache(maxsize=None)
def _load(name: str):
    return json.loads(JSON_FILES[name].read_text())


def J(name: str, *keys):
    """Look up a value in one of the campaign JSON files; raise on a missing key."""
    node = _load(name)
    path = [name]
    for key in keys:
        path.append(str(key))
        if isinstance(node, dict):
            if key not in node:
                raise KeyError(f"missing key {'/'.join(path)}")
            node = node[key]
        elif isinstance(node, list):
            node = node[int(key)]
        else:
            raise KeyError(f"cannot index {'/'.join(path)}")
    return node


def f(x, nd=3, sign=False):
    """Format a number (fixed decimals, or scientific for tiny/huge values)."""
    x = float(x)
    if x != 0 and (abs(x) < 10 ** (-nd) or abs(x) >= 1e5):
        s = f"{x:.{max(nd - 1, 1)}e}"
    else:
        s = f"{x:.{nd}f}"
    if sign and x >= 0:
        s = "+" + s
    return s.replace("-", "−")


def pm(x, e, nd=3, sign=False):
    return f"{f(x, nd, sign)} ± {f(e, nd)}"


def pct(x, nd=1):
    return f"{100 * float(x):.{nd}f}%".replace("-", "−")


def fig(path, caption: str, width="100%") -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    data = base64.b64encode(p.read_bytes()).decode()
    return (
        f"<figure><img src='data:image/png;base64,{data}' style='width:{width}'>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def table(header, rows) -> str:
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def page(*, slug, title, code, section, answer, question, why, theory,
         procedure, results, interpretation, caveats, provenance):
    """All text fields are HTML strings. ``answer`` is a one-line plain-text summary for the index."""
    return dict(slug=slug, title=title, code=code, section=section, answer=answer,
                question=question, why=why, theory=theory, procedure=procedure,
                results=results, interpretation=interpretation, caveats=caveats,
                provenance=provenance)


CONVENTIONS = f"""<div class='conv'><b>Common conventions.</b>
Model <code>{MODEL}</code> (trained on 100k galaxies, CCL-pretrained frozen feature
extractor). Test set: the Xu3 (DESI BGS Any) test set of 100,000 simulated galaxies.
Shear estimator: the posterior mean of \\((g_1, g_2)\\), with posterior samples reweighted
by the Tully–Fisher prior. Per-galaxy shape noise (Xu et al. 2024 definition):
\\(\\sigma_{{\\epsilon,i}} = \\sqrt{{\\sigma_{{g_1,i}}^2 + \\sigma_{{g_2,i}}^2}}\\), from posterior
standard deviations. Bias: \\(\\hat g - g = m\\,g + c\\), a linear fit over galaxies with
\\(|g| < 0.02\\) unless stated otherwise.</div>"""

STYLE = """
body { font: 16px/1.55 Palatino, 'Palatino Linotype', serif; margin: 2rem auto; max-width: 980px; padding: 0 1rem; color: #222; }
h1 { margin-bottom: .2rem; } .code { color: #777; font: 13px ui-sans-serif, sans-serif; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: .2rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font: 14px/1.4 ui-sans-serif, sans-serif; margin: .6rem 0; }
th, td { border-bottom: 1px solid #ccc; padding: .3rem .45rem; text-align: left; vertical-align: top; }
figure { margin: 1rem 0; } figcaption { font-size: 14px; color: #444; }
.conv { background: #f5f5f0; border-left: 4px solid #999; padding: .6rem .9rem; font-size: 14px; margin: 1rem 0; }
.answer { background: #eef4ee; border-left: 4px solid #5a5; padding: .6rem .9rem; }
nav { font: 14px ui-sans-serif, sans-serif; margin-bottom: 1rem; }
code { font-size: 90%; }
"""

MATHJAX = """<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}};</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>"""

SECTIONS = [
    ("question", "What this test asks"),
    ("why", "Why it matters for the paper"),
    ("theory", "Theory and justification"),
    ("procedure", "Procedure"),
    ("results", "Results"),
    ("interpretation", "What it means"),
    ("caveats", "Caveats and open issues"),
    ("provenance", "Provenance"),
]


def render(p: dict, prev_next=(None, None)) -> str:
    prev, nxt = prev_next
    nav = "<a href='../index.html'>← campaign index</a>"
    if prev:
        nav += f" · <a href='{prev['slug']}.html'>previous: {html.escape(prev['title'])}</a>"
    if nxt:
        nav += f" · <a href='{nxt['slug']}.html'>next: {html.escape(nxt['title'])}</a>"
    body = "".join(f"<h2>{h}</h2>{p[k]}" for k, h in SECTIONS)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(p['title'])}</title>
<style>{STYLE}</style>
{MATHJAX}
</head><body>
<nav>{nav}</nav>
<h1>{html.escape(p['title'])}</h1>
<div class='code'>{html.escape(p['section'])} · internal code in the group reports: {html.escape(p['code'])}</div>
<p class='answer'><b>Short answer.</b> {html.escape(p['answer'])}</p>
{CONVENTIONS}
{body}
</body></html>"""
