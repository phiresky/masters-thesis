#! /usr/bin/env python
"""
Pandoc filter to convert all headings to title case. Basically uppercase all words except stopwords (a, an, and, ...)
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
