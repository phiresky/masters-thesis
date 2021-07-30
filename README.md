# Bayesian and Attentive Aggregation for Cooperative Multi-Agent Deep Reinforcement Learning

My master's thesis. Written in Pandoc-Markdown.

The hosted version is here:

https://phiresky.github.io/masters-thesis/

The LaTeX PDF is here:

https://phiresky.github.io/masters-thesis/manuscript.pdf

This repository is based on Manubot: https://github.com/manubot/rootstock with
the following changes:

1. Add the KIT ALR LaTeX thesis template and the `build/build-pdf.sh` script to
   build the thesis using LaTeX in an indistinguishable manner from if it had
   been written with LaTeX. This (sadly) does not run in GitHub CI due to
   lazyness.
2. Minor styling changes to the html template in `build/themes/default.html`
3. A pandoc filter that automatically converts all headings to title-case (that
   is a great idea -> That is a Great Idea)
4. A pandoc filter that automatically converts svgs to pdf (including complex
   ones that inkscape / the normal latex svg package can't handle)
5. Switches `pandoc-xnos` pandoc filter to `pandoc-crossref` mostly because I'm
   more familiar with that syntax
6. Switches `pandoc-manubot-cite` pandoc filter to `pandoc-url2cite` because
   it's my own
