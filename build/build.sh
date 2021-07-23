#!/usr/bin/env bash

## build.sh: compile manuscript outputs from content using Manubot and Pandoc

set -o errexit \
    -o nounset \
    -o pipefail

# Set timezone used by Python for setting the manuscript's date
export TZ=Etc/UTC
# Default Python to read/write text files using UTF-8 encoding
export LC_ALL=en_US.UTF-8

# Generate reference information
echo >&2 "Retrieving and processing reference metadata"
manubot process \
  --content-directory=content \
  --output-directory=output \
  --cache-directory=ci/cache \
  --skip-citations \
  --log-level=INFO

# Pandoc's configuration is specified via files of option defaults
# located in the $PANDOC_DATA_DIR/defaults directory.
PANDOC_DATA_DIR="${PANDOC_DATA_DIR:-build/pandoc}"

# Make output directory
mkdir -p output

# Create HTML output
# https://pandoc.org/MANUAL.html
echo >&2 "Exporting HTML manuscript"
pandoc --verbose \
  --data-dir="$PANDOC_DATA_DIR" \
  --defaults=common.yaml \
  --defaults=html.yaml

# Set DOCKER_RUNNING to a non-empty string if docker is running, otherwise null.
DOCKER_RUNNING="$(docker info &> /dev/null && echo "yes" || true)"

# Create DOCX output (if BUILD_DOCX environment variable equals "true")
if [ "${BUILD_DOCX:-}" = "true" ]; then
  echo >&2 "Exporting Word Docx manuscript"
  pandoc --verbose \
    --data-dir="$PANDOC_DATA_DIR" \
    --defaults=common.yaml \
    --defaults=docx.yaml
fi

# Create LaTeX output (if BUILD_LATEX environment variable equals "true")
if [ "${BUILD_LATEX:-}" = "true" ]; then
  echo >&2 "Exporting LaTeX manuscript"
  pandoc \
    --data-dir="$PANDOC_DATA_DIR" \
    --defaults=common.yaml \
    --defaults=latex.yaml
fi

# Spellcheck
if [ "${SPELLCHECK:-}" = "true" ]; then
  export ASPELL_CONF="add-extra-dicts $(pwd)/build/assets/custom-dictionary.txt; ignore-case true"

  # Identify and store spelling errors
  pandoc \
    --data-dir="$PANDOC_DATA_DIR" \
    --lua-filter spellcheck.lua \
    output/manuscript.md \
    | sort -fu > output/spelling-errors.txt
  echo >&2 "Potential spelling errors:"
  cat output/spelling-errors.txt

  # Add additional forms of punctuation that Pandoc converts so that the
  # locations can be detected
  # Create a new expanded spelling errors file so that the saved artifact
  # contains only the original misspelled words
  cp output/spelling-errors.txt output/expanded-spelling-errors.txt
  grep "’" output/spelling-errors.txt | sed "s/’/'/g" >> output/expanded-spelling-errors.txt || true

  # Find locations of spelling errors
  # Use "|| true" after grep because otherwise this step of the pipeline will
  # return exit code 1 if any of the markdown files do not contain a
  # misspelled word
  cat output/expanded-spelling-errors.txt | while read word; do grep -ion "\<$word\>" content/*.md; done | sort -h -t ":" -k 1b,1 -k2,2 > output/spelling-error-locations.txt || true
  echo >&2 "Filenames and line numbers with potential spelling errors:"
  cat output/spelling-error-locations.txt

  rm output/expanded-spelling-errors.txt
fi

echo >&2 "Build complete"
