#!/usr/bin/env python3
"""Verification suite for the d-stack shuffle (coin d=2 / die d=6 versions).

Procedure under test (one pass, deck size n, d stacks):
  1. For each card taken from the top, draw a uniform value in {0..d-1}
     (coin flip for d=2, die roll for d=6) and deal the card to that stack.
  2. Reverse (flip) the odd-indexed stacks 1, 3, 5, ...
  3. Pick up stacks in fixed order 0..d-1 (stack 0 on top) to reform the deck.
Repeat k passes.  Recommended k(d, n) = ceil(1.25 * log_d(n)) + 1
(Chen-Ottolini shelf-shuffle cutoff (5/4) log_{2m} n, 2m = d, plus one
margin pass).

Tests, in decreasing strength / increasing feasible n:
  A. exact  : exact total-variation distance to uniform over S_n (n <= 8)
  B. coll   : collision-entropy (Renyi-2) test on sampled permutations (n <= 13)
  C. stats  : position-occupancy chi-square + pairwise-order aggregate (any n)
Tests B and C are one-sided: they detect non-uniformity, they cannot certify
its absence.  Test A is definitive for the n it can reach.

Usage:
  python3 verify_shuffle.py selftest          # vectorized-vs-scalar cross-check
  python3 verify_shuffle.py table             # k(d,n) table, d in {2,6}, n=2..70
  python3 verify_shuffle.py exact  d n kmax   # e.g. exact 6 6 4
  python3 verify_shuffle.py coll   d n k S    # e.g. coll  6 10 3 600000
  python3 verify_shuffle.py stats  d n k S    # e.g. stats 6 52 4 100000
  python3 verify_shuffle.py sweep             # spot checks across n=2..70 at
                                              # recommended k for d=2 and d=6
"""
import sys, math, itertools
from collections import defaultdict
import numpy as np


# ----------------------------------------------------------------------------
def rec_k(d, n):
    """Recommended pass count: cutoff (5/4) log_d n, ceiled, plus 1 margin."""
    if n < 2:
        return 1
    return math.ceil(1.25 * math.log(n) / math.log(d)) + 1


# ------------------------------------------------------------ scalar reference
def one_pass_scalar(deck, draws, d):
    stacks = [[] for _ in range(d)]
    for c, r in zip(deck, draws):
        stacks[r].append(c)
    out = []
    for s in range(d):
        seg = stacks[s]
        out += seg[::-1] if s % 2 == 1 else seg
    return tuple(out)


# --------------------------------------------------------- vectorized version
def one_pass_vec(D, rng, d):
    """D: (S, n) int array of decks; returns shuffled copies."""
    S, n = D.shape
    draw = rng.integers(0, d, size=(S, n))
    pos = np.arange(n)[None, :]
    key = draw * n + np.where(draw % 2 == 0, pos, n - 1 - pos)
    z = np.argsort(key, axis=1, kind="stable")
    return np.take_along_axis(D, z, axis=1)


def shuffle_many(n, d, k, S, seed, dtype=np.int16):
    rng = np.random.default_rng(seed)
    D = np.tile(np.arange(n, dtype=dtype), (S, 1))
    for _ in range(k):
        D = one_pass_vec(D, rng, d)
    return D


# ---------------------------------------------------------------- test A: TV
def exact_tv(n, d, passes):
    """Exact TV to uniform after each pass, via one-pass measure Q convolved.
    Cost: d^n to build Q, then |support(dist)| * |support(Q)| per pass."""
    ident = tuple(range(n))
    Q = defaultdict(float)
    w = 1.0 / d ** n
    for draws in itertools.product(range(d), repeat=n):
        Q[one_pass_scalar(ident, draws, d)] += w
    Qi = list(Q.items())
    uni = 1.0 / math.factorial(n)
    dist = {ident: 1.0}
    res = []
    for _ in range(passes):
        nd = defaultdict(float)
        for p, pp in dist.items():
            for g, qg in Qi:
                nd[tuple(p[i] for i in g)] += pp * qg
        dist = nd
        tv = 0.5 * (sum(abs(x - uni) for x in dist.values())
                    + (math.factorial(n) - len(dist)) * uni)
        res.append(tv)
    return res


# --------------------------------------------------- test B: collision entropy
def collision(n, d, k, S, seed=0):
    """Ratio of observed permutation collisions to the uniform expectation.
    ratio == 1 iff Renyi-2 entropy equals log2(n!).  Approx std dev of the
    ratio under uniformity ~ 1/sqrt(E[collisions])."""
    D = shuffle_many(n, d, k, S, seed, dtype=np.uint8)
    v = np.ascontiguousarray(D).view(np.dtype((np.void, n)))
    _, cnt = np.unique(v, return_counts=True)
    coll = float((cnt * (cnt - 1) // 2).sum())
    exp = S * (S - 1) / 2 / math.factorial(n)
    return coll / exp, 1.0 / math.sqrt(exp)


# ------------------------------------------------------- test C: distribution
def big_stats(n, d, k, S, seed=0):
    """(z_position, z_pairorder): standardized deviations of
    - chi-square of the n x n card/position occupancy matrix, df=(n-1)^2
    - sum over card pairs of squared z for P(i before j) = 1/2, df ~ n(n-1)/2
    The pairorder normalization ignores pair correlations; treat only
    |z| >~ 8 as decisive, and rely on z_position as the calibrated statistic."""
    D = shuffle_many(n, d, k, S, seed)
    pm = np.zeros((n, n))
    for p in range(n):
        pm[:, p] = np.bincount(D[:, p], minlength=n)
    e = S / n
    chi = ((pm - e) ** 2 / e).sum()
    df = (n - 1) ** 2
    zpos = (chi - df) / math.sqrt(2 * df)
    inv = np.argsort(D, axis=1)
    zsum, npairs = 0.0, 0
    for i in range(n):
        zz = ((inv[:, i][:, None] < inv[:, i + 1:]).mean(axis=0) - 0.5) \
             * 2 * math.sqrt(S)
        zsum += (zz ** 2).sum()
        npairs += zz.size
    zpair = (zsum - npairs) / math.sqrt(2 * npairs) if npairs else 0.0
    return zpos, zpair


# ------------------------------------------------------------------ self test
def selftest():
    rng = np.random.default_rng(0)
    for t in range(500):
        n = int(rng.integers(2, 12))
        d = int(rng.choice([2, 6]))
        deck = tuple(np.random.permutation(n))
        draws = list(np.random.randint(0, d, n))
        ref = one_pass_scalar(deck, draws, d)
        D = np.array([deck])
        draw = np.array([draws])
        pos = np.arange(n)[None, :]
        key = draw * n + np.where(draw % 2 == 0, pos, n - 1 - pos)
        z = np.argsort(key, axis=1, kind="stable")
        got = tuple(np.take_along_axis(D, z, axis=1)[0])
        assert got == ref, (t, n, d, deck, draws, got, ref)
    print("selftest: PASS (500 randomized scalar/vector cross-checks)")


# ---------------------------------------------------------------------- table
def table():
    print(" n range      k(d=2)   k(d=6)")
    prev = None
    for n in range(2, 71):
        cur = (rec_k(2, n), rec_k(6, n))
        if cur != prev:
            if prev is not None:
                print(f"  {lo}-{n-1:2d}        {prev[0]}        {prev[1]}")
            lo, prev = n, cur
    print(f"  {lo}-70        {prev[0]}        {prev[1]}")


# ---------------------------------------------------------------------- sweep
def sweep():
    print("A. exact TV at recommended k (definitive, small n):")
    for d, n in [(2, 5), (2, 6), (2, 7), (6, 5), (6, 6), (6, 7)]:
        k = rec_k(d, n)
        tv = exact_tv(n, d, k)
        print(f"   d={d} n={n} k={k}: TV after passes = "
              + ", ".join(f"{x:.5f}" for x in tv))
    print("\nB. collision-entropy ratio at recommended k "
          "(1.000 = uniform to Renyi-2):")
    for d, n, S in [(2, 8, 400000), (2, 10, 600000), (2, 12, 2000000),
                    (6, 8, 400000), (6, 10, 600000), (6, 12, 2000000)]:
        k = rec_k(d, n)
        r, sd = collision(n, d, k, S, seed=n * d)
        print(f"   d={d} n={n:2d} k={k}: ratio {r:.4f}  (sd {sd:.4f})")
    print("\nC. distribution statistics at recommended k "
          "(sigmas; |z|<~3 clean, pairorder tolerant to ~8):")
    for d, n, S in [(2, 20, 100000), (2, 40, 100000), (2, 52, 100000),
                    (2, 70, 60000),
                    (6, 20, 100000), (6, 40, 100000), (6, 52, 200000),
                    (6, 70, 100000)]:
        k = rec_k(d, n)
        zp, zq = big_stats(n, d, k, S, seed=n * d + k)
        print(f"   d={d} n={n:2d} k={k}: position z={zp:+6.2f}  "
              f"pairorder z={zq:+6.2f}")


# ============================================================================
# Exact a-shuffle machinery (radix-shuffle family: no reversals, fixed or any
# value-independent pickup order).  k passes of b bins == one b^k-shuffle,
# so exact TV at any n via the Bayer-Diaconis closed form.
# ============================================================================
from fractions import Fraction

def eulerian(n):
    """A[j] = number of permutations of n with j descents (big-int exact)."""
    A = [1]
    for m in range(1, n + 1):
        B = [0] * m
        for j in range(m):
            B[j] = (j + 1) * (A[j] if j < len(A) else 0) \
                 + (m - j) * (A[j - 1] if j - 1 >= 0 else 0)
        A = B
    return A

def tv_ashuffle(n, a):
    """Exact TV to uniform of a single a-shuffle of n cards (Bayer-Diaconis:
    P(pi) = C(a + n - r, n)/a^n, r = rising sequences of pi).  For the radix
    shuffle use a = bins**passes.  Validated against the published value
    tv_ashuffle(52, 2**7) = 0.3340609... and against brute-force enumeration
    at n = 6 (see paper)."""
    A = eulerian(n)
    fact = math.factorial(n)
    an = a ** n
    tot = 0
    for j in range(n):
        tot += A[j] * abs(math.comb(a + n - (j + 1), n) * fact - an)
    return Fraction(tot, 2 * an * fact)

def min_bins(n, k, eps):
    """Smallest b with exact TV(b^k-shuffle) <= eps (rule-of-thumb seeded,
    then corrected against the exact formula)."""
    c = math.sqrt(2 / math.pi) / (4 * math.sqrt(3))
    b = max(2, int((c * n ** 1.5 / eps) ** (1 / k)) - 2)
    while float(tv_ashuffle(n, b ** k)) > eps:
        b += 1
    return b

def radix_report():
    print("Exact TV of the radix shuffle at the two recommended designs:")
    print("  bins=36 (6x6 grid, two d6), 3 passes:")
    for n in (10, 40, 52, 70):
        print(f"    n={n:2d}: TV = {float(tv_ashuffle(n, 36**3)):.6f}")
    print("  bins=16 (4x4 grid, two d4 / d16 / four coins), 4 passes:")
    for n in (10, 40, 52, 70):
        print(f"    n={n:2d}: TV = {float(tv_ashuffle(n, 16**4)):.6f}")
    print("  sanity: tv(52, 2^7) =", float(tv_ashuffle(52, 2**7)),
          " (published: 0.3340609...)")



def adaptive_demo():
    """Demonstrate that outcome-dependent (largest-pile-first) pickup biases
    the shuffle: exact at n=3 (a=2) and simulated at n=52 (b=36, k=3)."""
    import itertools
    from collections import defaultdict
    Qf, Qb = defaultdict(int), defaultdict(int)
    for lab in itertools.product((0, 1), repeat=3):
        piles = [[], []]
        for c, l in enumerate(lab):
            piles[l].append(c)
        Qf[tuple(piles[0] + piles[1])] += 1
        order = sorted((0, 1), key=lambda s: (-len(piles[s]), s))
        Qb[tuple(piles[order[0]] + piles[order[1]])] += 1
    for name, Q in (("fixed", Qf), ("big-first", Qb)):
        tv = sum(abs(v / 8 - 1 / 6) for v in Q.values()) / 2              + (6 - len(Q)) / 12
        print(f"  n=3 a=2 {name:9s}: support {len(Q)}/6, TV = {tv:.4f}")
    for adaptive in (False, True):
        rng = np.random.default_rng(42)
        S, n, b = 200000, 52, 36
        D = np.tile(np.arange(n, dtype=np.int16), (S, 1))
        pos = np.arange(n)[None, :]
        for _ in range(3):
            die = rng.integers(0, b, size=(S, n))
            if adaptive:
                cnt = np.zeros((S, b), dtype=np.int32)
                np.add.at(cnt, (np.arange(S)[:, None], die), 1)
                order = np.argsort(-cnt, axis=1, kind="stable")
                rank = np.empty_like(order)
                np.put_along_axis(rank, order,
                                  np.tile(np.arange(b), (S, 1)), axis=1)
                cr = np.take_along_axis(rank, die, axis=1)
            else:
                cr = die
            z = np.argsort(cr * n + pos, axis=1, kind="stable")
            D = np.take_along_axis(D, z, axis=1)
        pm = np.zeros((n, n))
        for p in range(n):
            pm[:, p] = np.bincount(D[:, p], minlength=n)
        e = S / n
        chi = ((pm - e) ** 2 / e).sum()
        df = (n - 1) ** 2
        lab = "largest-first" if adaptive else "fixed order"
        print(f"  n=52 b=36 k=3 {lab:13s}: position z = "
              f"{(chi - df) / math.sqrt(2 * df):+6.2f}")


def entropies(n, a):
    """Exact entropy certificates for one a-shuffle (use a = bins**passes).
    Returns bit deficits vs uniform for Shannon (= KL divergence), Renyi-2,
    and min-entropy, plus exact TV.  All from the rising-sequence law
    p_r = C(a+n-r, n)/a^n with Eulerian multiplicities; heaviest atom is
    always the identity (r = 1), with deficit ~ n(n-1)*log2(e)/(2a)."""
    A = eulerian(n)
    lognfact = math.lgamma(n + 1) / math.log(2)
    logan = n * math.log2(a)
    H = 0.0; C2 = 0.0; logp1 = 0.0
    for j in range(n):
        r = j + 1
        logC = (math.lgamma(a + n - r + 1) - math.lgamma(a - r + 1)
                - math.lgamma(n + 1)) / math.log(2)
        logp = logC - logan
        if r == 1:
            logp1 = logp
        p = 2.0 ** logp
        Aj = float(A[j])
        H += -Aj * p * logp
        C2 += Aj * p * p
    return {"log2_nfact": lognfact,
            "shannon_deficit": lognfact - H,
            "renyi2_deficit": lognfact - (-math.log2(C2)),
            "minentropy_deficit": lognfact - (-logp1),
            "tv": float(tv_ashuffle(n, a))}

def entropy_report():
    for n in (40, 52):
        for name, a in (("6x6 k=3", 36 ** 3), ("4x4 k=4", 16 ** 4)):
            e = entropies(n, a)
            print(f"  n={n} {name}: TV={e['tv']:.2e}  deficits [bits]: "
                  f"Shannon {e['shannon_deficit']:.2e}, "
                  f"Renyi-2 {e['renyi2_deficit']:.2e}, "
                  f"min-entropy {e['minentropy_deficit']:.4f}"
                  f"  (log2 n! = {e['log2_nfact']:.3f})")


def recursive_shuffle(deck, rng, top_d=36, sub_d=6, _lvl=1, _st=None):
    """Las Vegas exact shuffle: deal into bins, recursively shuffle each bin,
    concatenate in predetermined order.  Exactly uniform (zero deficit) for
    any per-level radices, by conditioning on the size tree.  Returns
    (shuffled, stats)."""
    if _st is None:
        _st = {"rolls": 0, "deals": 0, "depth": 0}
    if len(deck) <= 1:
        return deck, _st
    d = top_d if _lvl == 1 else sub_d
    _st["rolls"] += len(deck)
    _st["deals"] += 1
    _st["depth"] = max(_st["depth"], _lvl)
    bins = [[] for _ in range(d)]
    for c in deck:
        bins[int(rng.integers(0, d))].append(c)
    out = []
    for b in bins:
        r, _ = recursive_shuffle(b, rng, top_d, sub_d, _lvl + 1, _st)
        out += r
    return out, _st

def recursive_report():
    from collections import Counter
    rng = np.random.default_rng(1)
    S, n = 400000, 10
    seen = Counter()
    for _ in range(S):
        r, _ = recursive_shuffle(list(range(n)), rng)
        seen[tuple(r)] += 1
    coll = sum(c * (c - 1) // 2 for c in seen.values())
    exp = S * (S - 1) / 2 / math.factorial(n)
    print(f"  exactness sanity n=10 (36 top / d6 subs): "
          f"collision ratio {coll/exp:.4f} (sd ~{1/math.sqrt(exp):.4f})")
    for n in (40, 52):
        for top, sub, lab in ((36, 6, "36 top / d6 subs"),
                              (36, 36, "36 throughout   ")):
            R = []
            rng = np.random.default_rng(n * top + sub)
            for _ in range(4000):
                _, st = recursive_shuffle(list(range(n)), rng, top, sub)
                R.append(st["rolls"])
            R = np.array(R)
            print(f"  n={n} {lab}: E[placements]={R.mean():6.1f}  "
                  f"95%<={np.percentile(R,95):.0f}  max seen {R.max()}"
                  f"   (fixed 6x6 k=3: {3*n})")


def stand_shuffle(deck, rng, top_d=36, sub_d=6, _st=None, _lvl=1,
                  fourfix=True):
    """Recursive exact shuffle with direct uniform resolution of 2- and
    3-card piles by a single d6 roll (2! and 3! both divide 6), as on the
    36-bin stand apparatus.  Exact by the theorem; piles of >= 4 recurse."""
    import itertools
    if _st is None:
        _st = {"throws": 0, "placed": 0, "res2": 0, "res3": 0,
               "redeals": 0, "depth": 1}
    n = len(deck)
    if n <= 1:
        return list(deck), _st
    if n == 2:
        # one die against the action table 1,1X,2,2X,3,3X: X (even faces)
        # means exchange -- exact, 3 faces each way.
        _st["throws"] += 1; _st["res2"] += 1
        face = int(rng.integers(0, 6))
        return (list(deck) if face % 2 == 0 else [deck[1], deck[0]]), _st
    if n == 3:
        # face f -> action (digit = f//2 + 1, X = f odd): move card <digit>
        # (1 = top) to the back; if X, exchange the remaining pair.
        # The six actions enumerate S3 exactly.
        _st["throws"] += 1; _st["res3"] += 1
        face = int(rng.integers(0, 6))
        digit, ex = face // 2, face % 2 == 1
        rest = [deck[i] for i in range(3) if i != digit]
        if ex:
            rest = [rest[1], rest[0]]
        return rest + [deck[digit]], _st
    if n == 4 and fourfix:
        # two dice mod 4 pick the first card (exact: 4 | 36), one die
        # permutes the remaining three (exact: 3! = 6) -- three throws,
        # deterministic, no recursion.
        _st["throws"] += 3; _st["res4"] = _st.get("res4", 0) + 1
        first = int(rng.integers(0, 4))
        rest = [deck[i] for i in range(4) if i != first]
        p = list(itertools.permutations(range(3)))[int(rng.integers(0, 6))]
        return [deck[first]] + [rest[i] for i in p], _st
    d = top_d if _lvl == 1 else sub_d
    _st["throws"] += n; _st["placed"] += n
    if _lvl > 1:
        _st["redeals"] += 1
    _st["depth"] = max(_st["depth"], _lvl)
    bins = [[] for _ in range(d)]
    for c in deck:
        bins[int(rng.integers(0, d))].append(c)
    out = []
    for b in bins:
        r, _ = stand_shuffle(b, rng, top_d, sub_d, _st, _lvl + 1,
                             fourfix)
        out += r
    return out, _st

def stand_report():
    for n in (40, 52):
        for top, sub, ff, lab in ((36, 6, True,  "stand + 4-fix   "),
                                  (36, 6, False, "stand, no 4-fix "),
                                  (12, 12, False, "shoebox d12     ")):
            T = []; P = []
            rng = np.random.default_rng(n * top)
            for _ in range(8000):
                _, st = stand_shuffle(list(range(n)), rng, top, sub,
                                      fourfix=ff)
                T.append(st["throws"]); P.append(st["placed"])
            print(f"  n={n} {lab}: E[throws]={np.mean(T):6.1f} "
                  f"(95%<={np.percentile(T,95):.0f})  "
                  f"E[deal-placements]={np.mean(P):6.1f}")

# ----------------------------------------------------------------------- main
if __name__ == "__main__":
    a = sys.argv[1:] or ["table"]
    cmd = a[0]
    if cmd == "selftest":
        selftest()
    elif cmd == "table":
        table()
    elif cmd == "exact":
        d, n, kmax = map(int, a[1:4])
        print(exact_tv(n, d, kmax))
    elif cmd == "coll":
        d, n, k, S = map(int, a[1:5])
        print(collision(n, d, k, S))
    elif cmd == "stats":
        d, n, k, S = map(int, a[1:5])
        print(big_stats(n, d, k, S))
    elif cmd == "atv":
        d, n, k = map(int, a[1:4])
        print(float(tv_ashuffle(n, d ** k)))
    elif cmd == "minbins":
        n, k = map(int, a[1:3]); eps = float(a[3])
        print(min_bins(n, k, eps))
    elif cmd == "radix":
        radix_report()
    elif cmd == "adaptive":
        adaptive_demo()
    elif cmd == "recursive":
        recursive_report()
    elif cmd == "stand":
        stand_report()
    elif cmd == "entropy":
        if len(a) >= 4:
            n, b, k = map(int, a[1:4])
            for kk, v in entropies(n, b ** k).items():
                print(f"  {kk}: {v:.6g}")
        else:
            entropy_report()
    elif cmd == "sweep":
        selftest()
        print()
        table()
        print()
        sweep()
    else:
        print(__doc__)


