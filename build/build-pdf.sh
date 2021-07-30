#!/bin/bash

# hacky script to build a latex PDF with a latex template. does not (currently) run in github CI

ln -s content/images

# copy files to new dir but without the stuff we process separately
rm -rf output/content-tex
mkdir -p output/content-tex
(
    cd content;
    for f in *; do
        ln -s ../../content/$f ../output/content-tex/
    done
)

rm output/content-tex/{0a_front-matter.md,0b_abstract.md} # we don't want these files in our PDF in the main section

export SVG2PDF_OUT_DIR=output/latex-intermediaries

mkdir -p "$SVG2PDF_OUT_DIR"

# basically same as build.sh but using a subset of the input files
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

# convert these files separately since they are included without the main section
pandoc "${args[@]}" content/0b_abstract.md -o output/output-tex/abstract.tex
pandoc "${args[@]}" content/abstract_de.md -o output/output-tex/abstract_de.tex

TEXINPUTS=.::build/assets/latex::output/output-tex latexmk -silent -outdir=output/latex-intermediaries -xelatex build/assets/latex/thesis.tex
rubber-info --into=output/latex-intermediaries build/assets/latex/thesis.tex
cp output/latex-intermediaries/thesis.pdf output/manuscript.pdf
