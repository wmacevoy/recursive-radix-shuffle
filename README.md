# Exact Shuffle

This is a set of notes on creating an exact shuffle using fair coins/dice.

The missing part is a physical artifact - a printable rig that has been
tested in use and refined.

I hope Karl and or Jeremy likes this idea and will collaborate on a paper to
Cryptologia - https://www.tandfonline.com/journals/ucry20

Here is the reference conversation:

https://claude.ai/share/c092d4f5-5b38-4d04-a869-47253195a461

## Building

The only dependency is Docker.

```
./build.sh
```

This runs `verify_shuffle.py selftest` as a sanity check and compiles
`exact_shuffle.pdf` from `exact_shuffle.tex`, both inside a container, so
nobody needs a local LaTeX install or Python environment to reproduce
either.

To work on the verification script without Docker, install
`requirements.txt` (just `numpy`) and run `verify_shuffle.py` directly —
see its docstring (`python3 verify_shuffle.py`) for the full list of
subcommands.
