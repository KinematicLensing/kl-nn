#!/usr/bin/env python3
"""Render the Markdown methods manuscript as standalone, Overleaf-ready TeX.

The Markdown remains the editable source of truth. Run this file after editing
it, or use --check to verify that the committed TeX rendering is current.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = ROOT / "repo_reports" / "METHODS_MANUSCRIPT.md"
TEX_PATH = ROOT / "repo_reports" / "METHODS_MANUSCRIPT.tex"
FINGERPRINT_RE = re.compile(
    r"<!-- klnn-methods-source-sha256: ([0-9a-f]{64}|PENDING) -->"
)
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)", re.DOTALL)
INLINE_MATH_RE = re.compile(r"\\\(.*?\\\)")
CODE_RE = re.compile(r"\x60([^\x60]+)\x60")
REFERENCE_RE = re.compile(r'^<a id="([^"]+)"></a>')


PREAMBLE = r"""\documentclass[11pt]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,bm}
\usepackage{array,booktabs,tabularx}
\usepackage{geometry}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{url}
\usepackage{ragged2e}

\geometry{margin=1in}
\hypersetup{
  colorlinks=true,
  linkcolor=blue!50!black,
  citecolor=blue!50!black,
  urlcolor=blue!60!black,
  pdfauthor={},
  pdftitle={KL-NN simulator-v3 methods manuscript}
}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.65em}
\setlength{\emergencystretch}{2em}
\newcommand{\code}[1]{\nolinkurl{#1}}
"""


def escape_tex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


class InlineRenderer:
    """Convert the small inline-Markdown subset used by this manuscript."""

    def __init__(self) -> None:
        self.tokens: list[str] = []

    def protect(self, rendered: str) -> str:
        token = f"ZZKLNNTOKEN{len(self.tokens)}ZZ"
        self.tokens.append(rendered)
        return token

    def restore(self, text: str) -> str:
        for index in range(len(self.tokens) - 1, -1, -1):
            text = text.replace(f"ZZKLNNTOKEN{index}ZZ", self.tokens[index])
        return text

    def fragment(self, text: str) -> str:
        text = CODE_RE.sub(
            lambda match: self.protect(r"\code{" + match.group(1) + "}"), text
        )
        return self.restore(escape_tex(text))

    def render(self, text: str) -> str:
        self.tokens = []
        text = INLINE_MATH_RE.sub(lambda match: self.protect(match.group(0)), text)
        text = CODE_RE.sub(
            lambda match: self.protect(r"\code{" + match.group(1) + "}"), text
        )

        def link(match: re.Match[str]) -> str:
            label = self.fragment(match.group(1))
            target = match.group(2)
            if target.startswith("#"):
                rendered = rf"\hyperlink{{ref:{target[1:]}}}{{{label}}}"
            else:
                rendered = rf"\href{{{target}}}{{{label}}}"
            return self.protect(rendered)

        text = LINK_RE.sub(link, text)
        text = re.sub(
            r"\*\*([^*]+)\*\*",
            lambda match: self.protect(
                r"\textbf{" + self.fragment(match.group(1)) + "}"
            ),
            text,
        )
        text = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            lambda match: self.protect(
                r"\emph{" + self.fragment(match.group(1)) + "}"
            ),
            text,
        )
        return self.restore(escape_tex(text))


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_spec(columns: int) -> str:
    ragged = r">{\RaggedRight\arraybackslash}"
    if columns == 2:
        return ragged + r"p{0.27\linewidth} " + ragged + "X"
    if columns == 3:
        return (
            ragged
            + r"p{0.27\linewidth} "
            + ragged
            + r"p{0.30\linewidth} "
            + ragged
            + "X"
        )
    if columns == 4:
        return (
            ragged
            + r"p{0.13\linewidth} "
            + ragged
            + r"p{0.17\linewidth} "
            + ragged
            + r"p{0.18\linewidth} "
            + ragged
            + "X"
        )
    return " ".join(ragged + "X" for _ in range(columns))


def render_table(lines: list[str], inline: InlineRenderer) -> str:
    rows = [split_table_row(line) for line in lines]
    header, body = rows[0], rows[2:]
    columns = len(header)
    if any(len(row) != columns for row in body):
        raise ValueError("inconsistent Markdown table width")

    def render_row(row: list[str], bold: bool = False) -> str:
        cells = [inline.render(cell) for cell in row]
        if bold:
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        return " & ".join(cells) + r" \\"

    output = [
        r"\begin{center}",
        r"\small",
        rf"\begin{{tabularx}}{{\linewidth}}{{@{{}}{table_spec(columns)}@{{}}}}",
        r"\toprule",
        render_row(header, bold=True),
        r"\midrule",
    ]
    output.extend(render_row(row) for row in body)
    output.extend([r"\bottomrule", r"\end{tabularx}", r"\end{center}"])
    return "\n".join(output)


def render_references(lines: list[str], inline: InlineRenderer) -> str:
    entries: list[tuple[str, str]] = []
    key: str | None = None
    body: list[str] = []

    def finish() -> None:
        nonlocal key, body
        if key is not None:
            entries.append((key, " ".join(part.strip() for part in body).strip()))
        key, body = None, []

    for line in lines:
        match = REFERENCE_RE.match(line)
        if match:
            finish()
            key = match.group(1)
            body = [line[match.end():]]
        elif key is not None and line.strip():
            body.append(line)
    finish()

    output = [r"\section*{References}", r"\addcontentsline{toc}{section}{References}"]
    for reference_key, reference_text in entries:
        output.extend(
            [
                rf"\hypertarget{{ref:{reference_key}}}{{}}%",
                inline.render(reference_text) + r"\par",
            ]
        )
    return "\n".join(output)


def render_document(markdown: str) -> str:
    fingerprint_match = FINGERPRINT_RE.search(markdown)
    if fingerprint_match is None:
        raise ValueError("source manuscript has no methods fingerprint")
    fingerprint = fingerprint_match.group(1)
    markdown = LINK_RE.sub(
        lambda match: f"[{' '.join(match.group(1).split())}]({match.group(2)})",
        markdown,
    )
    lines = markdown.splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError("source manuscript must begin with a level-one title")
    title = lines[0][2:].strip()
    try:
        references_index = lines.index("## References")
    except ValueError as error:
        raise ValueError("source manuscript has no References section") from error

    body_lines = lines[1:references_index]
    reference_lines = lines[references_index + 1:]
    date_match = re.search(r"working tree on (\d{4}-\d{2}-\d{2})", markdown)
    snapshot_date = date_match.group(1) if date_match else ""
    inline = InlineRenderer()
    output = [
        PREAMBLE.rstrip(),
        rf"\title{{{escape_tex(title)}}}",
        r"\author{}",
        rf"\date{{Working-tree snapshot: {escape_tex(snapshot_date)}}}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{center}",
        r"\small\textbf{Monitored-source fingerprint:} "
        + rf"\code{{{fingerprint}}}",
        r"\end{center}",
    ]

    index = 0
    in_math = False
    while index < len(body_lines):
        line = body_lines[index]
        stripped = line.strip()
        if FINGERPRINT_RE.fullmatch(stripped):
            index += 1
            continue
        if stripped == r"\[":
            in_math = True
            output.append(r"\[")
            index += 1
            continue
        if stripped == r"\]":
            in_math = False
            output.append(r"\]")
            index += 1
            continue
        if in_math:
            output.append(line)
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(body_lines):
            separator = body_lines[index + 1].strip()
            if re.fullmatch(r"\|(?:\s*:?-+:?\s*\|)+", separator):
                end = index + 2
                while end < len(body_lines) and body_lines[end].strip().startswith("|"):
                    end += 1
                output.append(render_table(body_lines[index:end], inline))
                index = end
                continue
        if line.startswith(">"):
            quote: list[str] = []
            while index < len(body_lines) and body_lines[index].startswith(">"):
                quote.append(body_lines[index].removeprefix("> "))
                index += 1
            output.extend([r"\begin{quote}", r"\small"])
            output.extend(inline.render(quote_line) for quote_line in quote)
            output.append(r"\end{quote}")
            continue
        if line.startswith("## "):
            output.append(rf"\section{{{inline.render(line[3:].strip())}}}")
        elif line.startswith("### "):
            output.append(rf"\subsection{{{inline.render(line[4:].strip())}}}")
        elif line.startswith("#### "):
            output.append(rf"\subsubsection{{{inline.render(line[5:].strip())}}}")
        elif stripped:
            output.append(inline.render(line))
        else:
            output.append("")
        index += 1

    if in_math:
        raise ValueError("unclosed display-math block")
    output.extend(
        [render_references(reference_lines, inline), r"\end{document}", ""]
    )
    return "\n".join(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    rendered = render_document(MARKDOWN_PATH.read_text(encoding="utf-8"))
    if args.check:
        if not TEX_PATH.is_file() or TEX_PATH.read_text(encoding="utf-8") != rendered:
            print(
                "METHODS_MANUSCRIPT.tex is stale; regenerate it with "
                "python repo_reports/render_methods_tex.py",
                file=sys.stderr,
            )
            return 1
        print("METHODS_MANUSCRIPT.tex matches METHODS_MANUSCRIPT.md")
        return 0
    TEX_PATH.write_text(rendered, encoding="utf-8")
    print(TEX_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
