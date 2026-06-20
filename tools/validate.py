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

# Patterns whose generic value-membership floor is only a LOOSE lower bound:
#   noisy_palindrome -- random corruptions are credited as predictable when
#                       their value coincidentally reappears, so the true
#                       floor is higher.
#   mixer            -- a whole-context concatenation of sub-patterns; the
#                       induction credit ignores cross-segment uncertainty.
# NOTE: the counting_* patterns are handled by a dedicated EXACT oracle and
# are not in this set.
_LOOSE_FLOOR_PATTERNS = frozenset({"noisy_palindrome", "mixer"})

# Match the production configuration set in generator.py.
generators.dyck._SHARED_IDS = True
nca_mod._SHARED_RULE = True

# The "simple" structural / counting / baseline patterns whose generative law
# is "either a fresh uniform draw or a deterministic copy of an earlier token".
# (dyck / shuffle_dyck / nca carry per-position choice entropy and have their
# own dedicated oracles above; they are intentionally excluded here.)
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
# --------------------------------------------------------------------------- #
# Huge collision-free symbol pool: range objects cost O(1) memory and support
# rng.choice / rng.sample / len / indexing, so generators run unchanged. ID 0
# (PAD_ID) is excluded so pad never collides with content. mixer materialises a
# per-segment vocab (list comprehension), so it gets a smaller -- still large --
# concrete list to keep memory and time bounded.
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


def report_simple_patterns(vocab, L, n_samples, seed=0):
    """Print the minimum-achievable-loss table for every simple pattern."""
    V = len(vocab)
    lnV = math.log(V)
    print(f"\nsimple patterns (vocab={V}, len={L}, n={n_samples}/pattern):")
    print(
        f"  uniform baseline ln(V) = {lnV:.4f} nats   "
        f"(an unstructured iid sequence is irreducible at this value)\n"
    )
    header = (
        f"  {'pattern':<22} {'pad%':>5} {'floor':>7} {'unigram':>8} {'gzip':>5} {'floor/lnV':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
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
        # Floor: `random` is iid uniform (exact ln(V)); counting_* use the
        # dedicated exact oracle that charges switch-point entropy; everything
        # else uses the collision-free free-draw floor (exact for the
        # deterministic patterns, a loose lower bound for the flagged ones).
        if name == "random":
            floor = lnV
            note = ""
        elif name == "counting_anbn":
            floor = oracle_entropy_counting(all_samples, L, k=2, vocab_size=V)
            note = ""
        elif name == "counting_anbncn":
            floor = oracle_entropy_counting(all_samples, L, k=3, vocab_size=V)
            note = ""
        else:
            floor = free_draw_floor(name, fn, L, n_samples, V, seed=seed)
            note = "*" if name in _LOOSE_FLOOR_PATTERNS else ""
        pad_flag = "" if pad_ok else "  <- PAD INSIDE BODY!"
        print(
            f"  {name + note:<22} {100 * _mean(pad_fracs):>4.0f}% "
            f"{floor:>7.4f} "
            f"{unigram:>8.4f} {_mean(gzips):>5.2f} "
            f"{floor / lnV:>10.3f}{pad_flag}"
        )
    print("\n  Legend:")
    print(
        "    pad%      : fraction of tokens that are the reserved masked pad "
        "(ID 0; excluded from the loss)"
    )
    print("    floor     : min achievable mean loss over UNMASKED tokens (the true training floor)")
    print("    unigram   : naive global-unigram cross-entropy (memorization baseline)")
    print("    gzip      : mean gzip compression ratio (README complexity metric)")
    print(
        "    floor/lnV : floor as a fraction of the uniform baseline "
        "(0 = fully predictable, 1 = unstructured)"
    )
    print(
        "    *         : floor is a LOOSE lower bound (true floor is higher); "
        "counting_* use a dedicated EXACT oracle"
    )


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


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

        # shuffle_dyck
        vocab = get_vocab(7)  # Enough for k=3 types + pad; the generators ignore the excess vocab.
        sd = [gen_shuffle_dyck(vocab, L, rng) for _ in range(N)]
        sd_bodies = [_strip_pad(s) for s in sd]
        sd_full_len = sum(len(s) == L for s in sd)
        sd_balanced = sum(is_valid_shuffle_dyck(b) for b in sd_bodies)
        sd_nested = sum(is_valid_nested_dyck(b) for b in sd_bodies)
        print(f"shuffle_dyck (nested Dyck-k, k=3, vocab={len(vocab)}, len={L}):")
        print(f"  exact length == L              : {sd_full_len}/{N}")
        print(f"  valid shuffle-Dyck (balanced)  : {sd_balanced}/{N}")
        print(f"  valid NESTED Dyck-k (stack)    : {sd_nested}/{N}")
        print(f"  unigram cross-entropy (nats)   : {empirical_unigram_entropy(sd):.4f}")
        print(f"  ORACLE achievable loss (nats)  : {oracle_entropy_shuffle(sd, L):.4f}")
        print(f"  ln(6) uniform baseline         : {math.log(6):.4f}")

        # dyck-1
        vocab = get_vocab(3)  # just open, close, pad; the generators ignore the excess vocab
        d1 = [gen_dyck(vocab, L, rng) for _ in range(N)]
        d1_bodies = [_strip_pad(s) for s in d1]
        d1_full_len = sum(len(s) == L for s in d1)
        d1_valid = sum(is_valid_dyck1(b) for b in d1_bodies)
        print(f"\ndyck (Dyck-1, vocab={len(vocab)}, len={L}):")
        print(f"  exact length == L              : {d1_full_len}/{N}")
        print(f"  valid Dyck-1 (balanced)        : {d1_valid}/{N}")
        print(f"  unigram cross-entropy (nats)   : {empirical_unigram_entropy(d1):.4f}")
        print(f"  ORACLE achievable loss (nats)  : {oracle_entropy_dyck1(d1, L):.4f}")
        print(f"  ln(2) uniform baseline         : {math.log(2):.4f}")

        # simple structural / counting / baseline patterns
        # These are composed the production way (random background + repeated
        # instance) using the production vocab of 256.
        report_simple_patterns(get_vocab(256), L, n_samples=200, seed=0)

        # nca
        vocab = get_vocab(
            11
        )  # enough for the state tokens + open/close + pad; the generator ignores the excess vocab
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

        print(
            f"\nnca (grid={nca_mod._GRID_SIZE}, d_state={d_state}, "
            f"regime='{nca_mod._REGIME}', temp={nca_mod._TEMPERATURE}, "
            f"bias={nca_mod._IDENTITY_BIAS}, len={L}):"
        )
        print(
            f"  malformed frames (open/close/cell) : "
            f"{bad_frames['bad_open']}/{bad_frames['bad_close']}/"
            f"{bad_frames['bad_cell']}"
        )
        print(f"  pad-tail tokens (non-pad = bad)    : {tail_total} total, {tail_nonpad} non-pad")

        # Regime ladder: oracle achievable cell loss vs uniform baseline ln(d_state).
        print("  regime ladder (oracle cell loss vs uniform baseline):")
        saved = (nca_mod._TEMPERATURE, nca_mod._IDENTITY_BIAS)
        for name, (temp, bias) in nca_mod._REGIMES.items():
            nca_mod._TEMPERATURE, nca_mod._IDENTITY_BIAS = temp, bias
            H_cell, ln_d = nca_oracle_entropy()
            tag = "NO signal" if H_cell / ln_d > 0.9 else "learnable"
            marker = " <- active" if name == nca_mod._REGIME else ""
            print(
                f"    {name:<13} temp={temp:<4} bias={bias:<4} "
                f"H={H_cell:.4f}  {H_cell / ln_d * 100:5.1f}% of ln(d)  "
                f"[{tag}]{marker}"
            )
        nca_mod._TEMPERATURE, nca_mod._IDENTITY_BIAS = saved

        # Restore stdout (log file auto-closed by context manager).
        sys.stdout = _old_stdout

    print(f"Log written to {_log_path}")


if __name__ == "__main__":
    main()
