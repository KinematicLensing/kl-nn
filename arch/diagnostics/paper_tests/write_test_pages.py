"""Render the explanatory per-test pages into paper-tests/tests/ and rebuild the index.

python write_test_pages.py
"""
import html
from html.parser import HTMLParser

from test_pages_common import ROOT, TESTS_DIR, render

import pages_npe
import pages_shear
import pages_stencil_stack
import pages_localcal

ORDER = [
    "training-convergence", "posterior-calibration", "tf-weight-health", "example-posteriors",
    "raw-shear-bias", "gaussian-shrinkage", "bias-fit-linearity", "error-bars-bootstrap",
    "width-vs-true-shear", "weighting-and-width-cuts", "width-cut-selection-bias",
    "response-calibration", "width-convention", "npe-likelihood-stacking",
    "local-calibration",
]


class _Check(HTMLParser):
    def __init__(self):
        super().__init__()
        self.bad_img = 0

    def handle_starttag(self, tag, attrs):
        if tag == "img" and not dict(attrs).get("src", "").startswith("data:image"):
            self.bad_img += 1


def main():
    pages = {p["slug"]: p for mod in (pages_npe, pages_shear, pages_stencil_stack, pages_localcal) for p in mod.PAGES}
    missing = [s for s in ORDER if s not in pages]
    extra = [s for s in pages if s not in ORDER]
    if missing or extra:
        raise SystemExit(f"missing pages {missing}; unexpected pages {extra}")
    ordered = [pages[s] for s in ORDER]
    TESTS_DIR.mkdir(exist_ok=True)
    for i, p in enumerate(ordered):
        prev = ordered[i - 1] if i > 0 else None
        nxt = ordered[i + 1] if i + 1 < len(ordered) else None
        text = render(p, (prev, nxt))
        checker = _Check()
        checker.feed(text)
        if checker.bad_img:
            raise SystemExit(f"{p['slug']}: {checker.bad_img} non-embedded images")
        (TESTS_DIR / f"{p['slug']}.html").write_text(text)
        print("wrote", p["slug"])
    write_index(ordered)


GROUP_REPORTS = [
    ("A_npe/report.html", "NPE validation (training, calibration, TF weights, corner plots)"),
    ("A_shear/report.html", "Shear estimators on the Xu3 cache"),
    ("B_inference/report.html", "New inference: SNR contours, 9D corners, matched-shear stencil"),
    ("C_stacking/report.html", "NPE likelihood stacking at constant shear"),
    ("D_local_calibration/report.html", "Coverage conditional on the true shear"),
]


def write_index(ordered):
    sections = {}
    for p in ordered:
        sections.setdefault(p["section"], []).append(p)
    parts = []
    for sec, ps in sections.items():
        rows = "".join(
            f"<tr><td><a href='tests/{p['slug']}.html'>{html.escape(p['title'])}</a></td>"
            f"<td>{html.escape(p['answer'])}</td></tr>" for p in ps)
        parts.append(f"<h2>{html.escape(sec)}</h2><table><thead><tr><th>Test</th>"
                     f"<th>Short answer</th></tr></thead><tbody>{rows}</tbody></table>")
    glossary = "".join(
        f"<tr><td>{html.escape(p['code'])}</td><td><a href='tests/{p['slug']}.html'>"
        f"{html.escape(p['title'])}</a></td></tr>" for p in ordered)
    groups = "".join(f"<li><a href='{u}'>{html.escape(t)}</a></li>" for u, t in GROUP_REPORTS)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>paper-tests campaign</title>
<style>
body {{ font: 16px/1.5 Palatino, serif; margin: 2rem auto; max-width: 1000px; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; font: 14px/1.4 ui-sans-serif, sans-serif; }}
th, td {{ border-bottom: 1px solid #ccc; padding: .35rem .45rem; text-align: left; vertical-align: top; }}
</style></head><body>
<h1>paper-tests campaign</h1>
<p>Tests run before writing Sec. 5 (validating the NPE) and Sec. 6 (shear estimation)
of the KL-NN paper, on the current 100k-trained model and the Xu3 test set. Each
page below explains what the test asks, the theory behind it, the procedure, and the
results. Plan: <code>kl-nn/repo_reports/PAPER_TESTS_PLAN.md</code>.</p>
{''.join(parts)}
<h2>Detailed group reports</h2>
<p>These are the raw reports written by each agent. They use internal codes; the
glossary below maps them to the pages above.</p>
<ul>{groups}</ul>
<table><thead><tr><th>Internal code</th><th>Page</th></tr></thead><tbody>{glossary}</tbody></table>
</body></html>"""
    (ROOT / "index.html").write_text(page)
    print("wrote index.html")


if __name__ == "__main__":
    main()
