#!/usr/bin/env bash
# Build the project in Docker: run the verification selftest and compile
# exact_shuffle.pdf from exact_shuffle.tex. Requires only Docker.
set -euo pipefail
cd "$(dirname "$0")"

IMAGE=recursive-radix-shuffle

docker build -t "$IMAGE" .

docker run --rm \
  -v "$PWD":/work -w /work \
  -e HOME=/tmp \
  --user "$(id -u):$(id -g)" \
  "$IMAGE" bash -c '
    set -euo pipefail
    echo "== verify_shuffle.py selftest =="
    python3 verify_shuffle.py selftest
    echo "== building exact_shuffle.pdf =="
    latexmk -pdf -interaction=nonstopmode -halt-on-error exact_shuffle.tex
    latexmk -c
  '

echo "Done: exact_shuffle.pdf"
