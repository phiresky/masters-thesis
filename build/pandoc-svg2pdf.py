#! /usr/bin/env python
"""
Pandoc filter to convert svg files to pdf as suggested at:
https://github.com/jgm/pandoc/issues/265#issuecomment-27317316
"""

import subprocess
import os
import sys
from pandocfilters import toJSONFilter, Image
from pathlib import Path
from urllib.parse import unquote

def svg_to_any(key, value, fmt, meta):
    out_dir = os.environ.get("SVG2PDF_OUT_DIR")
    if key == 'Image':
        attrs, alt, [src, title] = value
        src = Path(unquote(src))
        if src.suffix == ".svg":
            eps_name = src.with_suffix(".pdf") if not out_dir else Path(out_dir) / src.with_suffix(".pdf").name
            try:
                mtime = os.path.getmtime(eps_name)
            except OSError:
                mtime = -1
            if mtime < os.path.getmtime(src):
                cmd_line = [Path(__file__).parent / 'svg2pdf.sh', src, eps_name]
                cmd_line = [str(a) for a in cmd_line]
                sys.stderr.write("Running %s\n" % " ".join(cmd_line))
                subprocess.call(cmd_line, stdout=sys.stderr.fileno())
            return Image(attrs, alt, [str(eps_name), title])

if __name__ == "__main__":
    toJSONFilter(svg_to_any)