#!/bin/bash

# copy files to new dir but without the stuff we process separately
rm -rf output/content-tex
mkdir -p output/content-tex
(
    cd content;
    for f in *; do
        ln -s ../../content/$f ../output/content-tex/
    done
)

rm output/content-tex/{00.front-matter.md,0_abstract.md}

export SVG2PDF_OUT_DIR=output/latex-intermediaries

mkdir -p "$SVG2PDF_OUT_DIR"

manubot process \
  --content-directory=output/content-tex \
  --output-directory=output/output-tex \
  --cache-directory=ci/cache \
  --skip-citations \
  --log-level=INFO

PANDOC_DATA_DIR=build/pandoc
pandoc \
    --data-dir="$PANDOC_DATA_DIR" \
    --defaults=latex.yaml \
    -M url2cite-output-bib=output/latex-intermediaries/thesis.bib \
    --verbose

args=(--shift-heading-level-by=-1 --top-level-division=chapter)
pandoc "${args[@]}" content/0_abstract.md -o output/output-tex/abstract.tex
pandoc "${args[@]}" content/abstract_de.md -o output/output-tex/abstract_de.tex

TEXINPUTS=.::build/assets/latex::output/output-tex latexmk -silent -outdir=output/latex-intermediaries -xelatex build/assets/latex/thesis.tex
rubber-info --into=output/latex-intermediaries build/assets/latex/thesis.tex
cp output/latex-intermediaries/thesis.pdf output
