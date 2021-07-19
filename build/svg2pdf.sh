#!/bin/bash
# https://gist.github.com/guillermo/3258662554c6afa2128492ca9a1a116c
# Convert an SVG file to a PDF file by using headless Chrome.
#

if [ $# -ne 2 ]; then
  echo "Usage: ./svg2pdf.sh input.svg output.pdf" 1>&2
  exit 1
fi

INPUT=$1
OUTPUT=$2

if [[ "$INPUT" == *.drawio.svg ]]; then
    draw.io "$INPUT" --transparent --crop --export --output "$OUTPUT"
    exit 0
fi

HTML="
<html>
  <head>
    <style>
body {
  margin: 0;
}
    </style>
    <script>
function init() {
  const element = document.getElementById('targetsvg');
  const positionInfo = element.getBoundingClientRect();
  const height = positionInfo.height;
  const width = positionInfo.width;
  const style = document.createElement('style');
  style.innerHTML = \`@page {margin: 0; size: \${width}px \${height}px}\`;
  document.head.appendChild(style);
}
window.onload = init;
    </script>
  </head>
  <body>
    <img id=\"targetsvg\" src=\"${INPUT}\">
  </body>
</html>
"

tmpfile=$(mktemp XXXXXX.html)
trap "rm -f $tmpfile" EXIT
echo $HTML > $tmpfile

chromium --headless --disable-gpu --print-to-pdf="$OUTPUT" --virtual-time-budget=10000 $tmpfile