#!/bin/bash
# Build Secs. 5, 6 and the appendix of the KL-NN paper in a stub article (REVTeX is not installed on Bridges-2).
# Usage: build_paper_stub.sh [build_dir]; prints errors, undefined references and overfull boxes.
set -u
P=/jet/home/xwang30/Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks
B=${1:-/tmp/klnn_paper_stub_$USER}
mkdir -p "$B" && cd "$B" || exit 1
ln -sfn "$P/sections" sections; ln -sfn "$P/figs" figs; ln -sf "$P/main.bib" main.bib
E=$(grep -n '^\\begin{document}' "$P/main.tex" | cut -d: -f1)
{ echo '\documentclass[twocolumn]{article}'
  sed -n "22,$((E-1))p" "$P/main.tex" | grep -v '^%' | grep -v 'orcidlink\|siunitx\|lipsum\|gensymb\|subfiles\|documentclass\|multirow'
  echo '\usepackage{booktabs}'; echo '\providecommand{\tableline}{\hline}'; echo '\usepackage[numbers]{natbib}'; echo '\begin{document}'
  printf '\\input{sections/data-generate}\n\\input{sections/validation}\n\\input{sections/shear}\n\\input{sections/appendix}\n\\bibliographystyle{plain}\n\\bibliography{main}\n\\end{document}\n'
} > stub.tex
rm -f stub.aux stub.bbl
pdflatex -interaction=nonstopmode stub >/dev/null; bibtex stub >/dev/null 2>&1
pdflatex -interaction=nonstopmode stub >/dev/null; pdflatex -interaction=nonstopmode stub >/dev/null
echo "errors: $(grep -c '^!' stub.log)"; grep -A2 '^!' stub.log | head -20
echo "undefined references (expected: sections outside the stub):"
grep -o "Reference \`[^']*' on page [0-9]* undefined" stub.log | sed 's/ on page.*//' | sort -u
echo "overfull boxes: $(grep -c Overfull stub.log)"; grep -A1 Overfull stub.log | head -10
echo "pdf: $B/stub.pdf"
