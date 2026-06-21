"""Validation checks for the "learnability" of the generated data.

It (1) checks whether emitted samples are structurally *valid* (balanced
Dyck words after stripping the pad token; well-formed delimiter-wrapped
frames for NCA), and (2) estimates the irreducible next-token entropy of
the generating process, i.e., the best cross-entropy loss any model
could possibly reach on this data. If that oracle loss equals the uniform
baseline (ln of the alphabet), the task carries no learnable signal.

It also (3) reports, for every *simple* structural / counting / baseline
pattern, the minimum achievable next-token loss of the COMPOSED samples
that are actually written to disk. The reported floor therefore measures
only the irreducible cost of the genuinely unpredictable draws, with pad
positions excluded from the loss.

Usage:
    python tools/validate.py
"""

import gzip
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

import generators  # noqa: F401  (registers patterns + module flags)
from compose import compose_sample
from generators import counting as counting_mod
from generators import nca as nca_mod
from generators.dyck import PAD_ID, gen_dyck, gen_shuffle_dyck
from generators.nca import gen_nca
from registry import PATTERNS
from utils import get_vocab

# --------------------------------------------------------------------------- #
# Patterns whose generic value-membership floor is only a LOOSE lower bound:
#   noisy_palindrome -- random corruptions are credited as predictable when
#                       their value coincidentally reappears, so the true
#                       floor is higher.
#   mixer            -- a whole-context concatenation of sub-patterns; the
#                       induction credit ignores cross-segment uncertainty.
# NOTE: the counting_* patterns are handled by a dedicated EXACT oracle and
# are not in this set.
# --------------------------------------------------------------------------- #
_LOOSE_FLOOR_PATTERNS = frozenset({"noisy_palindrome", "mixer"})

# Match the production configuration set in generator.py.
generators.dyck._SHARED_IDS = True
nca_mod._SHARED_RULE = True

# --------------------------------------------------------------------------- #
# The "simple" structural / counting / baseline patterns whose generative law
# is "either a fresh uniform draw or a deterministic copy of an earlier token".
# (dyck / shuffle_dyck / nca carry per-position choice entropy and have their
# own dedicated oracles above; they are intentionally excluded here.)
# --------------------------------------------------------------------------- #
SIMPLE_PATTERNS = [
    "periodic",
    "palindrome",
    "copy",
    "reverse",
    "counting_anbn",
    "counting_anbncn",
    "nested",
    "interleaving",
    "permutation_cycle",
    "hierarchical",
    "noisy_palindrome",
    "random",
    "identity",
    "composite_mirror_repeat",
    "mixer",
]


def _strip_pad(seq):
    """Remove trailing pad tokens and assert pad only ever appears in the tail."""
    n = len(seq)
    while n > 0 and seq[n - 1] == PAD_ID:
        n -= 1
    body = seq[:n]
    # Pad must never appear inside the bracket body.
    assert PAD_ID not in body, "pad token found inside the expression body"
    return body


# Validators (IDs: pad=0, dyck open=1/close=2, shuffle openers=1..k closers=k+1..2k)
def is_valid_dyck1(seq, open_id=1, close_id=2):
    """True iff `seq` (pad already stripped) is a balanced Dyck-1 word."""
    depth = 0
    for t in seq:
        if t == open_id:
            depth += 1
        elif t == close_id:
            depth -= 1
            if depth < 0:  # closed with nothing open
                return False
        else:
            return False  # token outside the bracket alphabet
    return depth == 0


def is_valid_shuffle_dyck(seq, k=3):
    """True iff `seq` is a valid *shuffle* Dyck-k word.

    Each type must independently be balanced and never go negative, but
    types may interleave freely (no stack/nesting constraint).
    """
    openers = set(range(1, k + 1))  # ids 1..k
    closers = {c: c - k - 1 for c in range(k + 1, 2 * k + 1)}  # id -> type
    counts = [0] * k
    for t in seq:
        if t in openers:
            counts[t - 1] += 1
        elif t in closers:
            ty = closers[t]
            counts[ty] -= 1
            if counts[ty] < 0:
                return False
        else:
            return False
    return all(c == 0 for c in counts)


def is_valid_nested_dyck(seq, k=3):
    """True iff `seq` is a *nested* (stack-matched) Dyck-k word.

    This is the classic hierarchical Dyck-k: a closer must match the type
    on top of the stack. shuffle_dyck output will usually FAIL this.
    """
    openers = set(range(1, k + 1))
    closers = {c: c - k - 1 for c in range(k + 1, 2 * k + 1)}
    stack = []
    for t in seq:
        if t in openers:
            stack.append(t - 1)
        elif t in closers:
            if not stack or stack[-1] != closers[t]:
                return False
            stack.pop()
        else:
            return False
    return not stack


# Entropy estimator
def empirical_unigram_entropy(samples):
    """Cross-entropy (nats) of predicting each token from the global unigram."""
    c = Counter()
    for s in samples:
        c.update(s)
    total = sum(c.values())
    return -sum((n / total) * math.log(n / total) for n in c.values())


# --------------------------------------------------------------------------- #
# Oracle conditional entropy: the BEST cross-entropy any model can reach.
#
# We replay the exact generation rule against each emitted (non-pad) token and
# read off the probability the generator assigned to it. The mean of -log p is
# the irreducible loss of a model that has perfectly learned the rule. If this
# is well below ln(alphabet), the task carries real learnable signal.
# --------------------------------------------------------------------------- #
def oracle_entropy_shuffle(samples, L, k=3, p_open=0.5, max_depth=4):
    openers = set(range(1, k + 1))
    nll, n = 0.0, 0
    for s in samples:
        stack = []
        for i, t in enumerate(s):
            if t == PAD_ID:
                break  # tail is masked in the loss
            budget = L - i
            depth = len(stack)
            if depth == 0:
                p = 1.0 / k  # uniform over k openers
                stack.append(t - 1)
            else:
                can_open = budget >= depth + 2 and depth < max_depth
                if t in openers:
                    p = (p_open / k) if can_open else 0.0
                    stack.append(t - 1)
                else:  # forced/elected close (deterministic type)
                    p = (1.0 - p_open) if can_open else 1.0
                    stack.pop()
            nll += -math.log(p)
            n += 1
    return nll / n


def oracle_entropy_dyck1(samples, L):
    nll, n = 0.0, 0
    for s in samples:
        depth = 0
        for i, t in enumerate(s):
            if t == PAD_ID:
                break
            budget = L - i
            if depth == 0:
                p = 1.0  # only an opener is legal
                depth += 1
            elif budget >= depth + 2:
                p = 0.5  # free open/close choice
                depth += 1 if t == 1 else -1
            else:
                p = 1.0  # forced close
                depth -= 1
            nll += -math.log(p)
            n += 1
    return nll / n


# --------------------------------------------------------------------------- #
# Simple-pattern oracle: minimum achievable loss for the composed samples.
#
# Every sample is a SINGLE pattern instance filling the whole context, with
# the trailing slack set to the reserved pad token (ID 0). Pad positions are
# masked (excluded from the loss). For the remaining (content) tokens the
# structural / baseline generators only ever do two things to produce a token:
# draw a FRESH uniform symbol, or DETERMINISTICALLY reuse an earlier symbol
# (mirror, repeat, cycle, copy, ...). So the irreducible loss of a perfect model
# that has learned the rule is just the cost of the genuinely free draws:
#
#   * free draw      -> a fresh uniform symbol over V            -> ln(V)
#   * determined     -> a deterministic copy of an earlier token -> 0
#   * pad tail       -> masked, excluded from the loss            -> (skipped)
#
# The floor is therefore  (#free draws / #content tokens) * ln(V).  We must NOT
# count "free" by value-novelty on the production vocab, because two independent
# fresh draws collide ~1/V of the time and a collision would be miscounted as a
# predictable copy, badly UNDER-estimating the floor (e.g. a palindrome's free
# first half would look only ~80% novel instead of 100%). Instead we re-run each
# generator on a HUGE collision-free vocabulary so that a repeated value can
# only be a genuine structural copy; the free-draw fraction measured there is
# exact, and we then charge each free draw the REAL ln(V).
#
# This is EXACT for the deterministic copy / reflection / cycle / periodic
# patterns. It remains a LOOSE lower bound for noisy_palindrome (clean mirror
# positions still carry the 10% corruption uncertainty) and mixer (its counting
# sub-segments hide switch-point entropy); both are flagged via
# _LOOSE_FLOOR_PATTERNS. The counting_* patterns use their own exact oracle
# (`oracle_entropy_counting`) that additionally charges the switch-point entropy.
#
# Huge collision-free symbol pool: range objects cost O(1) memory and support
# rng.choice / rng.sample / len / indexing, so generators run unchanged. ID 0
# (PAD_ID) is excluded so pad never collides with content. mixer materialises a
# per-segment vocab (list comprehension), so it gets a smaller -- still large --
# concrete list to keep memory and time bounded.
# --------------------------------------------------------------------------- #
_HUGE_VOCAB = range(1, 1 << 30)
_MIXER_VOCAB = list(range(1, (1 << 18) + 1))


def _pad_fraction(sample):
    """Fraction of a sample that is the reserved masked pad token."""
    return sum(x == PAD_ID for x in sample) / len(sample)


def free_draw_floor(name, fn, L, n_samples, vocab_size, seed=0):
    """EXACT structural floor (nats): (#free draws / #content) * ln(V).

    Re-runs the generator on a collision-free vocabulary so a repeated value can
    only be a genuine structural copy. Each first-seen (non-pad) value is one
    free uniform draw worth ln(vocab_size); every repeat is a determined copy
    worth 0; pad tokens are masked out.
    """
    lnV = math.log(vocab_size)
    content = _MIXER_VOCAB if name == "mixer" else _HUGE_VOCAB
    rng = random.Random(seed)
    total_free = total_content = 0
    for _ in range(n_samples):
        sample = fn(content, L, rng)
        seen = set()
        for x in sample:
            if x == PAD_ID:
                continue  # masked tail: excluded from loss
            total_content += 1
            if x not in seen:
                seen.add(x)
                total_free += 1
    return (total_free / total_content) * lnV if total_content else 0.0


def oracle_entropy_counting(samples, target_len, k, vocab_size, run_min=None, run_max=None):
    """EXACT minimum loss (nats) for the counting_anbn / counting_anbncn data.

    Each sample is a tiling of segments a^n b^n (... c^n) whose run length `n`
    is drawn fresh per segment from Uniform[run_min, M], with
    M = clip(remaining // k, run_min, run_max). A perfect model pays:

      * k * ln(V) per sample      -- the one-off novel symbol identities
                                     (a, b, [c] are each first seen once), and
      * ln(M_seg - run_min + 1)   -- the switch-point entropy of each segment:
                                     the a-run uniquely encodes n, and n is
                                     uniform over (M - run_min + 1) values, so
                                     by the chain rule the a-run + switch tokens
                                     cost exactly H(n) = ln(#values). Once the
                                     switch is observed the remaining runs are
                                     fully determined (0 cost).

    Pad tokens are masked. Returns the mean loss over the unmasked tokens.
    """
    run_min = counting_mod._RUN_MIN if run_min is None else run_min
    run_max = counting_mod._RUN_MAX if run_max is None else run_max
    lnV = math.log(vocab_size)
    total_nll, total_n = 0.0, 0
    for s in samples:
        body = [t for t in s if t != PAD_ID]  # pad only ever in the tail
        L = len(body)
        if L == 0:
            continue
        a = body[0]
        nll = k * lnV  # k novel symbol identities
        i = 0
        while i < L:
            seg_start = i
            run = 0
            while i < L and body[i] == a:  # measure the a-run length n
                i += 1
                run += 1
            i += run * (k - 1)  # skip the remaining (b, [c]) runs
            remaining = target_len - seg_start
            M = max(run_min, min(run_max, remaining // k))
            nll += math.log(M - run_min + 1)  # switch-point entropy H(n)
        total_nll += nll
        total_n += L
    return total_nll / total_n if total_n else 0.0


def gzip_complexity(sample):
    """gzip compression ratio = compressed_bytes / raw_bytes (README metric)."""
    raw = bytes(t & 0xFF for t in sample)  # 1 byte/token (vocab<=256)
    comp = gzip.compress(raw, compresslevel=9)
    return len(comp) / len(raw)


def _pad_only_in_tail(sample):
    """True iff PAD_ID appears only as a contiguous trailing block (or not at all)."""
    seen_pad = False
    for x in sample:
        if x == PAD_ID:
            seen_pad = True
        elif seen_pad:
            return False  # content after pad => pad inside body
    return True


def collect_simple_pattern_data(vocab, L, n_samples, seed=0):
    """Collect learnability metrics for every simple structural/counting/baseline pattern.

    Returns a list of result dicts, one per pattern.
    """
    V = len(vocab)
    lnV = math.log(V)
    results = []
    for name in SIMPLE_PATTERNS:
        if name not in PATTERNS:
            continue
        _desc, fn = PATTERNS[name]
        # Production-vocab samples: used for pad%, gzip, unigram and the
        # pad-only-in-tail structural check (the data exactly as written).
        rng = random.Random(seed)
        pad_fracs = []
        gzips = []
        pad_ok = True
        all_samples = []
        for _ in range(n_samples):
            sample, _insertions = compose_sample(
                name,
                fn,
                vocab,
                L,
                rng=rng,
            )
            pad_fracs.append(_pad_fraction(sample))
            gzips.append(gzip_complexity(sample))
            pad_ok = pad_ok and _pad_only_in_tail(sample)
            all_samples.append(sample)
        unigram = empirical_unigram_entropy(all_samples)
        # Oracle (floor): `random` is iid uniform (exact ln(V)); counting_* use
        # the dedicated exact oracle that charges switch-point entropy;
        # everything else uses the collision-free free-draw floor (exact for
        # the deterministic patterns, a loose lower bound for the flagged ones).
        if name == "random":
            oracle = lnV
            loose = False
        elif name == "counting_anbn":
            oracle = oracle_entropy_counting(all_samples, L, k=2, vocab_size=V)
            loose = False
        elif name == "counting_anbncn":
            oracle = oracle_entropy_counting(all_samples, L, k=3, vocab_size=V)
            loose = False
        else:
            oracle = free_draw_floor(name, fn, L, n_samples, V, seed=seed)
            loose = name in _LOOSE_FLOOR_PATTERNS
        results.append(
            {
                "name": name,
                "family": "structural",
                "oracle_loss": oracle,
                "random_baseline": lnV,
                "gzip_ratio": _mean(gzips),
                "unigram": unigram,
                "pad_frac": _mean(pad_fracs),
                "pad_ok": pad_ok,
                "loose_bound": loose,
                "warnings": (["PAD_INSIDE_BODY"] if not pad_ok else []),
                "valid_label": None,
                "valid_count": None,
                "total_count": None,
            }
        )
    return results


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# Output formatting
def _print_primary_table(rows):
    """Print a unified table of the most important learnability metrics.

    All losses are cross-entropy in nats (natural log units).
    Lower values = less uncertainty.  Pad tokens excluded from all metrics.
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print("  PRIMARY METRICS — Intrinsic Learnability")
    print(sep)
    print()
    print("  Lower Oracle Loss -> more predictable.  Higher Improvement -> stronger signal.")
    print()
    hdr = f"  {'Pattern':<27} {'Oracle Loss':>12} {'Random':>9} {'Gzip':>7} {'Improvement':>13}"
    sub = f"  {'':27} {'(best possible)':>12} {'(uniform)':>9} {'Ratio':>7} {'(Random÷Oracle)':>13}"
    print(hdr)
    print(sub)
    print("  " + "─" * 70)
    for r in rows:
        name = r["name"]
        flags = ""
        if r.get("loose_bound"):
            flags += " †"
        if r.get("warnings"):
            flags += " ⚠"
        oracle = r["oracle_loss"]
        rand = r["random_baseline"]
        gz = r.get("gzip_ratio")
        impr = rand / oracle if oracle > 0 else float("inf")
        gz_str = f"{gz:>7.2f}" if gz is not None else "     —"
        print(f"  {name + flags:<27} {oracle:>12.4f} {rand:>9.4f} {gz_str} {impr:>11.2f} ×")
    print()
    print("  Oracle Loss    Best possible cross-entropy — a perfect model that learned the rule.")
    print("  Random         Cross-entropy of a uniform random guesser = ln(vocab size).")
    print("  Gzip Ratio     gzip compressed size ÷ raw size.  Lower = more structured / redundant.")
    print("  Improvement    Random ÷ Oracle.  How many times better than random guessing.")
    print("                 Values near 1.0 -> little or no learnable signal in the data.")
    print("  🔮             Oracle value is a LOOSE lower bound; true floor is higher.")
    print("  ⚠️             See Diagnostic Checks section below.")


def _print_diagnostics(dyck_rows, simple_rows, nca_blob):
    """Print all diagnostic / sanity-check information in a clearly separated section."""
    sep = "─" * 80
    print(f"\n{sep}")
    print("  DIAGNOSTIC CHECKS")
    print(sep)

    # Structural validity
    print("\n  🔹 Structural Validity")
    print(f"  {'Pattern':<27} {'Check':<34} {'Result':>12}")
    print("  " + "─" * 75)
    for r in dyck_rows:
        for check in r.get("validity_checks", []):
            ok = check["ok"]
            mark = " ✅" if ok else " ❌ FAIL"
            print(
                f"  {r['name']:<27} {check['label']:<34} "
                f"{check['count']:>4}/{check['total']:<4}{mark}"
            )
    for r in simple_rows:
        if r.get("valid_label") and r.get("total_count"):
            ok = r["valid_count"] == r["total_count"]
            mark = " ✅" if ok else " ❌ FAIL"
            print(
                f"  {r['name']:<27} {r['valid_label']:<34} "
                f"{r['valid_count']:>4}/{r['total_count']:<4}{mark}"
            )
    if nca_blob:
        bo, bc, bcell = nca_blob["bad_open"], nca_blob["bad_close"], nca_blob["bad_cell"]
        n_bad = bo + bc + bcell
        ok = n_bad == 0
        mark = " ✅" if ok else " ❌ FAIL"
        print(f"  {'nca':<27} {'frame open/close/cell malformed':<34} {n_bad:>4} issues{mark}")

    # Pad token sanity
    print("\n  🔹 Pad Token Sanity  (pad ID 0 should only appear as a trailing suffix)")
    print(f"  {'Pattern':<27} {'Pad Fraction':>12} {'Status':>12}")
    print("  " + "─" * 53)
    for r in dyck_rows:
        pf = r.get("pad_frac")
        if pf is not None:
            ok = r.get("pad_ok", True)
            print(
                f"  {r['name']:<27} {pf * 100:>8.0f}%     {'✅ OK' if ok else '⚠ PAD IN BODY':>12}"
            )
    for r in simple_rows:
        pf = r.get("pad_frac")
        if pf is not None:
            ok = r.get("pad_ok", True)
            print(
                f"  {r['name']:<27} {pf * 100:>8.0f}%     {'✅ OK' if ok else '⚠ PAD IN BODY':>12}"
            )
    if nca_blob:
        total_tail = nca_blob["tail_total"]
        nonpad = nca_blob["tail_nonpad"]
        ok = nonpad == 0
        print(f"  {'nca':<27} {total_tail:>8} tail    {'✅ OK' if ok else '⚠ NON-PAD IN TAIL':>12}")

    # Empirical unigram comparison
    print("\n  🔹 Empirical Unigram Loss  (naive token-frequency baseline for comparison)")
    print(f"  {'Pattern':<27} {'Unigram':>10} {'Oracle':>10} {'Excess':>10}")
    print("  " + "─" * 60)
    for r in dyck_rows:
        if r.get("unigram") is not None:
            excess = r["unigram"] - r["oracle_loss"]
            print(
                f"  {r['name']:<27} {r['unigram']:>10.4f} {r['oracle_loss']:>10.4f} {excess:>10.4f}"
            )
    for r in simple_rows:
        if r.get("unigram") is not None:
            excess = r["unigram"] - r["oracle_loss"]
            print(
                f"  {r['name']:<27} {r['unigram']:>10.4f} {r['oracle_loss']:>10.4f} {excess:>10.4f}"
            )

    # NCA regime ladder
    if nca_blob and nca_blob.get("regimes"):
        print(
            f"\n  🔹 NCA Regime Ladder  (cell oracle vs uniform baseline ln(d)={nca_blob['ln_d']:.4f})"
        )
        print(
            f"  {'Regime':<15} {'Temp':>6} {'Bias':>6} {'Oracle':>10} {'% of ln(d)':>12} {'Verdict':>12}"
        )
        print("  " + "─" * 65)
        for reg in nca_blob["regimes"]:
            verdict = "NO signal" if reg["pct"] > 90 else "learnable ✅"
            active = " ← active" if reg["active"] else ""
            print(
                f"  {reg['name']:<15} {reg['temp']:>6.1f} {reg['bias']:>6.1f} "
                f"{reg['oracle']:>10.4f} {reg['pct']:>11.1f}% {verdict:>12}{active}"
            )

    # Loose-bound notes
    print("\n  🔹 Notes on Oracle Loss Bounds")
    print("    noisy_palindrome  — random corruptions credited as predictable when value")
    print("                         coincidentally reappears; true floor is higher.")
    print("    mixer             — whole-context concatenation of sub-patterns; ignores")
    print("                         cross-segment uncertainty.")
    print("    counting_anbn(*)  — EXACT oracle used (charges switch-point entropy).")


# NCA validators
def validate_nca_structure(seq, frame_size, open_tok, close_tok, state_ids):
    """Structural checks for one serialized NCA sample.

    Returns (issues, tail) where `issues` counts malformed full frames and
    `tail` is the residual (non-frame) suffix so the caller can inspect how
    the leftover slack was padded.
    """
    state_set = set(state_ids)
    n_full = len(seq) // frame_size
    issues = Counter()
    for f in range(n_full):
        base = f * frame_size
        if seq[base] != open_tok:
            issues["bad_open"] += 1
        if seq[base + frame_size - 1] != close_tok:
            issues["bad_close"] += 1
        for c in seq[base + 1 : base + frame_size - 1]:
            if c not in state_set:
                issues["bad_cell"] += 1
    tail = seq[n_full * frame_size :]
    return issues, tail


def nca_oracle_entropy(n_roll=40, frames=15, seed0=0):
    """Mean per-cell conditional entropy (nats) of the NCA transition law.

    This is the best achievable cross-entropy on the *cell* tokens for a
    model that has perfectly learned the (shared) rule and tracks the grid.
    Replays the exact production rollout (shared rule, burn-in, temperature,
    identity bias). Compare against ln(d_state): values near that baseline
    mean the automaton is essentially a uniform RNG (no learnable signal).
    """
    d = nca_mod._D_STATE
    grid = nca_mod._GRID_SIZE
    device = torch.device("cpu")
    sg = torch.Generator(device="cpu")
    sg.manual_seed(nca_mod._SHARED_RULE_SEED)
    net = nca_mod._make_rule(d, sg, device)

    tot, cnt = 0.0, 0
    for r in range(n_roll):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed0 + r)
        p0 = torch.softmax(torch.empty(d).normal_(generator=g), dim=-1)
        state = torch.multinomial(p0, grid * grid, replacement=True, generator=g).reshape(
            grid, grid
        )
        for t in range(nca_mod._BURN_IN_STEPS + frames):
            oh = F.one_hot(state, d).float()
            logits = (net(oh) + oh * nca_mod._IDENTITY_BIAS) / nca_mod._TEMPERATURE
            probs = torch.softmax(logits, dim=-1).reshape(-1, d)
            if t >= nca_mod._BURN_IN_STEPS:
                H = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
                tot += H.sum().item()
                cnt += H.numel()
            state = torch.multinomial(probs, 1, generator=g).reshape(grid, grid)
    return tot / cnt, math.log(d)


class _Tee:
    """Duplicates writes to both *stdout* and an open file handle."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, text):
        self.stream.write(text)
        self.fh.write(text)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def main():
    # Redirect all print() output to both the terminal and the log file.
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate.logs")
    _old_stdout = sys.stdout
    with open(_log_path, "w") as _log_fh:
        sys.stdout = _Tee(_old_stdout, _log_fh)

        rng = random.Random(0)
        L = 4096
        N = 10000

        # ── shuffle_dyck ─────────────────────────────────────────────
        vocab = get_vocab(7)
        sd = [gen_shuffle_dyck(vocab, L, rng) for _ in range(N)]
        sd_bodies = [_strip_pad(s) for s in sd]
        sd_oracle = oracle_entropy_shuffle(sd, L)
        sd_gzip = _mean([gzip_complexity(s) for s in sd])
        sd_pad_ok = all(_pad_only_in_tail(s) for s in sd)

        dyck_rows = [
            {
                "name": "shuffle_dyck (k=3)",
                "family": "dyck",
                "oracle_loss": sd_oracle,
                "random_baseline": math.log(6),
                "gzip_ratio": sd_gzip,
                "unigram": empirical_unigram_entropy(sd),
                "pad_frac": _mean([_pad_fraction(s) for s in sd]),
                "pad_ok": sd_pad_ok,
                "loose_bound": False,
                "warnings": [],
                "validity_checks": [
                    {
                        "label": "exact length == L",
                        "count": sum(len(s) == L for s in sd),
                        "total": N,
                        "ok": all(len(s) == L for s in sd),
                    },
                    {
                        "label": "valid shuffle-Dyck (balanced)",
                        "count": sum(is_valid_shuffle_dyck(b) for b in sd_bodies),
                        "total": N,
                        "ok": all(is_valid_shuffle_dyck(b) for b in sd_bodies),
                    },
                    {
                        "label": "valid NESTED Dyck-k (stack-matched)",
                        "count": sum(is_valid_nested_dyck(b) for b in sd_bodies),
                        "total": N,
                        "ok": all(is_valid_nested_dyck(b) for b in sd_bodies),
                    },
                ],
            }
        ]

        # ── dyck-1 ───────────────────────────────────────────────────
        vocab = get_vocab(3)
        d1 = [gen_dyck(vocab, L, rng) for _ in range(N)]
        d1_bodies = [_strip_pad(s) for s in d1]
        d1_oracle = oracle_entropy_dyck1(d1, L)
        d1_gzip = _mean([gzip_complexity(s) for s in d1])
        d1_pad_ok = all(_pad_only_in_tail(s) for s in d1)

        dyck_rows.append(
            {
                "name": "dyck-1",
                "family": "dyck",
                "oracle_loss": d1_oracle,
                "random_baseline": math.log(2),
                "gzip_ratio": d1_gzip,
                "unigram": empirical_unigram_entropy(d1),
                "pad_frac": _mean([_pad_fraction(s) for s in d1]),
                "pad_ok": d1_pad_ok,
                "loose_bound": False,
                "warnings": [],
                "validity_checks": [
                    {
                        "label": "exact length == L",
                        "count": sum(len(s) == L for s in d1),
                        "total": N,
                        "ok": all(len(s) == L for s in d1),
                    },
                    {
                        "label": "valid Dyck-1 (balanced)",
                        "count": sum(is_valid_dyck1(b) for b in d1_bodies),
                        "total": N,
                        "ok": all(is_valid_dyck1(b) for b in d1_bodies),
                    },
                ],
            }
        )

        # ── Simple structural / counting / baseline patterns ─────────
        simple_rows = collect_simple_pattern_data(get_vocab(256), L, n_samples=200, seed=0)

        # ── NCA ──────────────────────────────────────────────────────
        vocab = get_vocab(11)
        n_nca = 100
        d_state = min(nca_mod._D_STATE, len(vocab) - nca_mod._N_RESERVED)
        frame_size = nca_mod._GRID_SIZE**2 + 2
        pad_tok = PAD_ID
        open_tok = vocab[nca_mod._OPEN_IDX]
        close_tok = vocab[nca_mod._CLOSE_IDX]
        state_ids = vocab[nca_mod._N_RESERVED : nca_mod._N_RESERVED + d_state]

        nca_samples = [gen_nca(vocab, L, rng) for _ in range(n_nca)]
        bad_frames = Counter()
        tail_total = tail_nonpad = 0
        for s in nca_samples:
            issues, tail = validate_nca_structure(s, frame_size, open_tok, close_tok, state_ids)
            bad_frames.update(issues)
            tail_total += len(tail)
            tail_nonpad += sum(t != pad_tok for t in tail)
        nca_gzip = _mean([gzip_complexity(s) for s in nca_samples])

        # Regime ladder
        saved = (nca_mod._TEMPERATURE, nca_mod._IDENTITY_BIAS)
        regimes = []
        for name, (temp, bias) in nca_mod._REGIMES.items():
            nca_mod._TEMPERATURE, nca_mod._IDENTITY_BIAS = temp, bias
            H_cell, ln_d = nca_oracle_entropy()
            regimes.append(
                {
                    "name": name,
                    "temp": temp,
                    "bias": bias,
                    "oracle": H_cell,
                    "pct": H_cell / ln_d * 100,
                    "active": name == nca_mod._REGIME,
                }
            )
        nca_mod._TEMPERATURE, nca_mod._IDENTITY_BIAS = saved

        # Active regime values for the primary table
        active_reg = next(r for r in regimes if r["active"])
        ln_d = math.log(nca_mod._D_STATE)

        nca_blob = {
            "bad_open": bad_frames["bad_open"],
            "bad_close": bad_frames["bad_close"],
            "bad_cell": bad_frames["bad_cell"],
            "tail_total": tail_total,
            "tail_nonpad": tail_nonpad,
            "ln_d": ln_d,
            "regimes": regimes,
        }

        nca_row = {
            "name": f"nca ({nca_mod._REGIME})",
            "family": "nca",
            "oracle_loss": active_reg["oracle"],
            "random_baseline": ln_d,
            "gzip_ratio": nca_gzip,
            "unigram": None,
            "pad_frac": None,
            "pad_ok": True,
            "loose_bound": False,
            "warnings": [],
        }

        # ── Print unified report ─────────────────────────────────────
        all_primary = dyck_rows + simple_rows + [nca_row]
        _print_primary_table(all_primary)
        _print_diagnostics(dyck_rows, simple_rows, nca_blob)

        # Restore stdout (log file auto-closed by context manager).
        sys.stdout = _old_stdout

    print(f"Log written to {_log_path}")


if __name__ == "__main__":
    main()
