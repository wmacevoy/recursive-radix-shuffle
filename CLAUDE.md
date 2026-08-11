# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research notes on an *exact* (zero-deficit) manual card-shuffling procedure, directed by dice/coins,
written up as a short paper aimed at *Cryptologia*. There is no application code — just:

- `exact_shuffle.tex` / `exact_shuffle.pdf` — the paper ("An Exact Manual Shuffle").
- `verify_shuffle.py` — simulation/verification code that reproduces the numbers cited in the paper.
- `README.md` — project pointer, link to the originating Claude conversation, and build instructions.
- `Dockerfile` / `build.sh` / `requirements.txt` — containerized build so collaborators need only Docker.

The main open TODO is the physical rig itself: a printable/laser-cuttable apparatus (the 36-bin stand or
shoebox-lid version described in the paper's Apparatus section) that's actually been built and tested, not just
simulated.

## Commands

```bash
./build.sh                            # Docker: runs `verify_shuffle.py selftest`, then builds exact_shuffle.pdf
```

`build.sh` builds the `recursive-radix-shuffle` image (Python 3.12 + texlive + `requirements.txt`), then runs the
selftest and `latexmk -pdf` inside a container with the repo bind-mounted at `/work`, so `exact_shuffle.pdf`
lands directly in the working tree owned by the calling user (`--user "$(id -u):$(id -g)"`). This is the
easiest path for a collaborator with no local Python/LaTeX setup — only Docker is required.

To iterate on `verify_shuffle.py` outside Docker, `pip install -r requirements.txt` (just `numpy`) and run it
directly:

```bash
python3 verify_shuffle.py selftest    # vectorized-vs-scalar cross-check, must PASS before trusting other output
python3 verify_shuffle.py recursive   # reproduces the paper's "exactness sanity" + placement-count figures
python3 verify_shuffle.py stand       # reproduces the paper's Cost-section throw/placement table (stand/shoebox rigs)
python3 verify_shuffle.py adaptive    # reproduces the "one discipline" bias demo (Section 3, n=3 exact + n=52 sim)
python3 verify_shuffle.py sweep       # selftest + table + full test-A/B/C battery in one run
```

There is no test framework, lint config, or Makefile — `selftest` is the correctness check, and the other
subcommands are report generators. Run `python3 verify_shuffle.py` with no args (or any unrecognized subcommand)
to print the full usage docstring, which is the authoritative list of subcommands and their arguments.

## Architecture: two distinct shuffle models live in one file

`verify_shuffle.py` contains analysis for **two unrelated shuffle procedures**; don't conflate them when editing
either the paper or the script.

1. **The "d-stack reversal" shuffle** (top half of the file: `one_pass_scalar`, `one_pass_vec`, `exact_tv`,
   `collision`, `big_stats`, `rec_k`, `table`, `sweep`, `selftest`). This is a fixed-`k`-pass approximate shuffle
   (deal to `d` stacks, reverse odd-indexed stacks, restack, repeat `k` times) with a recommended pass count
   `k(d,n) = ceil(1.25 log_d n) + 1`. It converges to uniform but never reaches it exactly. **This is not the
   procedure described in the current paper** — it's retained as background/comparison material (the paper's
   abstract explicitly contrasts its own zero-deficit result against exactly this kind of fixed-pass approach).

2. **The exact recursive radix shuffle** (bottom half: `eulerian`, `tv_ashuffle`, `min_bins`, `radix_report`,
   `entropies`/`entropy_report`, `recursive_shuffle`/`recursive_report`, `stand_shuffle`/`stand_report`,
   `adaptive_demo`). This *is* the paper's procedure: deal into `d` bins, recurse only into bins holding ≥2 cards,
   concatenate in a fixed predetermined order. It is exactly uniform for any per-level radix (proved in the paper,
   not just simulated) — `recursive_shuffle`/`stand_shuffle` simulate placement/throw counts (cost, not
   correctness), while `tv_ashuffle`/`eulerian` give the exact Bayer-Diaconis total-variation formula for the
   classical *fixed-pass* `a`-shuffle, used here only as the "still has a deficit" foil to the exact recursive
   result (`radix_report`, `entropy_report`).

`stand_shuffle` further special-cases pile sizes 2, 3, and 4 with a single die roll each (see the "action table"
in the paper's Apparatus section) instead of recursing, matching the physical 36-bin stand apparatus described
there — `fourfix=False` disables the size-4 special case for comparison.

When changing either the paper's numeric claims (Cost table, exactness sanity ratios, bias-demo TV values) or the
corresponding script functions, keep them in sync — the paper cites `verify_shuffle.py`'s `recursive` and `stand`
commands directly as its source of empirical figures.
