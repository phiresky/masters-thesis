#! /usr/bin/env python
"""
Pandoc filter to convert svg files to pdf as suggested at:
https://github.com/jgm/pandoc/issues/265#issuecomment-27317316
"""

from pandocfilters import toJSONFilter, Header, Str, stringify
from titlecase import titlecase


def title_case(key, value, fmt, meta):
    if key == "Header":
        # print(value, file=sys.stderr)
        level, attrs, content = value
        return Header(level, attrs, [Str(titlecase(stringify(content)))])


if __name__ == "__main__":
    toJSONFilter(title_case)
