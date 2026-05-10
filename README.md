# Patterns Are All You Need

This is a simple codebase for generating synthetic sequence data with structural patterns. The patterns are designed to probe various capabilities a sequence model may need (locality, symmetry, counting, recursion, agreement, etc.) and can be used for "pre-pretraining" before exposing the model to natural language.

The codebase is split across a few focused modules — see [Where things live](#where-things-live) below.

## The pattern catalogue

The 16 currently registered patterns and what each is meant to test:

| Pattern                   | Schematic example          | What it probes                                                                  |
|---------------------------|----------------------------|---------------------------------------------------------------------------------|
| `periodic`                | `ABCABCABC`                | Fixed-period repetition (regular language).                                     |
| `palindrome`              | `ABCCBA`                   | Mirror symmetry around the center (CFG-recognizable).                           |
| `copy`                    | `ABCD ABCD ABCD`           | Block duplication / verbatim copying.                                           |
| `reverse`                 | `ABCD \| DCBA`             | Source + reverse separated by an explicit delimiter.                            |
| `counting_anbn`           | `AAABBB`                   | Equal counts of two symbols (CFG counting `a^n b^n`).                           |
| `counting_anbncn`         | `AAABBBCCC`                | Equal counts of three symbols (mildly context-sensitive `a^n b^n c^n`).         |
| `nested`                  | `ABCDDCBA`                 | Recursive palindromic structure from `S → a S a`.                               |
| `interleaving`            | `ABABAB` or `AABBAABB`     | Alternation / block-interleaving of two symbols.                                |
| `permutation_cycle`       | `ABCD BCDA CDAB DABC`      | Cyclic permutations of a base block.                                            |
| `hierarchical`            | `ABAB CCCC ABAB`           | Local + global structure mixed at multiple scales.                              |
| `noisy_palindrome`        | `ABCXCBA` (~10% corrupted) | Palindrome under random token corruption (robustness to noise).                 |
| `dyck`                    | `(()())`                   | Dyck-1: balanced brackets of a single type.                                     |
| `shuffle_dyck`            | `( [ ) { } ]`              | Typed Dyck-k: k bracket types whose open/close tokens may interleave freely.    |
| `random`                  | `qZ7ξ%`                    | Uniformly random tokens — unstructured baseline / control.                      |
| `identity`                | `AAAAAA`                   | Single-token repetition (zero-entropy floor).                                   |
| `composite_mirror_repeat` | `ABCCBA ABCCBA`            | Multi-rule composition: a small palindrome repeated periodically.               |

Letters in the schematics stand for *distinct vocabulary tokens*; concrete IDs are sampled per call so different samples use different surface tokens.

## How samples are composed

`compose_sample` is what turns a generator into a full training example of length `--max-context-length`. Two regimes:

- **Standard patterns** (everything except `dyck` and `shuffle_dyck`) — `compose_sample` calls the generator **once** to produce a single pattern instance of length drawn from `[length_min, length_max]`, then splices that *exact same instance* into a random-noise background at multiple non-overlapping positions. The number of repetitions is chosen so that the pattern occupies **at least `--signal-floor` of the context** (default `0.5`, i.e. 50%). Gap sizes between copies are randomized so the model does not learn fixed positions, but within one sample the pattern tokens are always identical. Different samples will see different random pattern instances.

  The signal floor is configurable via `--signal-floor`:

  - default: `0.5` (50% signal, 50% random noise)
  - hard bounds: `[0.10, 0.90]` — values outside this range raise `SystemExit`
  - soft bounds: a warning is printed if `F < 0.5` (signal too weak to be reliably learnable) or `F > 0.8` (samples dominated by the pattern with little noise to disambiguate it)

## Output layout

Each pattern writes its shards into a dedicated subdirectory named after the pattern, under the directory inferred from `--output`. For example, with `--output out/patterns.jsonl`:

```
out/
  periodic/
    patterns.0000.jsonl
  palindrome/
    patterns.0000.jsonl
  dyck/
    patterns.0000.jsonl
  ...
```

Each record is a JSON line. By default it includes full metadata:

```json
{"input_ids": [...], "metadata": {"pattern_type": "periodic", "vocab_size": 50257, "max_context_length": 64, "range": [4, 16], "n_insertions": 3, "insertions": [{"start": 5, "length": 8}, ...]}}
```

With `--no-metadata`, only `input_ids` is written — useful when the files are consumed directly by a training pipeline:

```json
{"input_ids": [...]}
```

- **Dyck patterns (`dyck`, `shuffle_dyck`)** — handled as a special case. The entire sample is a *single* valid Dyck expression of length exactly `max_context_length`; there is no random background and no repetition. This is intentional: a Dyck expression is only meaningful as a whole (its brackets must balance globally), so splicing fragments into noise would destroy the structural property the pattern is supposed to test. Conceptually the "signal coverage" for Dyck samples is 100% and `--signal-floor` does not apply.

## How do I add a new pattern?

Adding a new pattern is a three-step process: write a generator function in the right module, decorate it with `@register`, and (optionally) verify with `--debug`.

### 1. The generator contract

Every pattern is a plain function with the signature:

```python
def gen_<name>(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    ...
```

Rules:

- **Input** — `vocab` is the filtered list of token IDs to draw from, `target_len` is the desired sequence length (drawn from `[length_min, length_max]` by the caller), and `rng` is a seeded `random.Random` instance. **Always use `rng`**, never the `random` module directly, so runs stay reproducible under `--seed`.
- **Output** — a `list[int]` of length **exactly** `target_len`. The easiest way to guarantee this is to build the structural prefix and then call the helper `pad_to(out, target_len, vocab, rng)` at the end. `pad_to` truncates if too long and tail-pads with random vocab IDs if too short.
- **No side effects** — do not print, do not write files, do not mutate `vocab`.
- **Use `sample_distinct(vocab, k, rng)`** when you need `k` distinct token IDs (e.g. for the `A`, `B`, `C` symbols of `A^n B^n C^n`). It falls back gracefully if the vocab is smaller than `k`.

### 2. Write the generator in the appropriate module

Place the function in whichever file under `generators/` best matches its theme:

| File                       | Patterns it contains                       |
|----------------------------|--------------------------------------------|
| `generators/structural.py` | symmetry, repetition, positional structure |
| `generators/counting.py`   | symbol-counting patterns                   |
| `generators/dyck.py`       | bracket / Dyck languages                   |
| `generators/baseline.py`   | unstructured controls                      |

If none of the existing files fits, create a new module (e.g. `generators/arithmetic.py`) and add one import line to `generators/__init__.py`.

Decorate the function with `@register(name, description)`:

```python
from registry import register
from utils import pad_to, sample_distinct

@register(
    "my_pattern",
    "One-sentence description of what structural property this pattern tests "
    "(e.g. 'long-range agreement', 'modular arithmetic').",
)
def gen_my_pattern(vocab, target_len, rng):
    # build the structural part
    a, b = sample_distinct(vocab, 2, rng)
    out = [a, b] * (target_len // 2)
    # always end with pad_to to guarantee exact length
    return pad_to(out, target_len, vocab, rng)
```

The decorator inserts an entry into the global `PATTERNS` dict, so the new pattern is automatically picked up by `compose_sample`, the debug printer, and the main write loop. **No other file needs to change** (unless you created a new module, in which case add one import to `generators/__init__.py`).

Naming convention: lowercase, snake_case, descriptive of the structural property (`palindrome`, `counting_anbn`, `shuffle_dyck`). The name appears verbatim in each record's `metadata.pattern_type`.

### 3. Verify with `--debug`

```bash
python generator.py \
  --tokenizer gpt2 \
  --max-context-length 32 \
  --length-min 4 --length-max 4 \
  --samples-per-pattern 1 \
  --patterns all \
  --debug
```

To verify only specific patterns, pass their names to `--patterns`:

```bash
python generator.py \
  --tokenizer gpt2 \
  --max-context-length 32 \
  --length-min 4 --length-max 4 \
  --samples-per-pattern 1 \
  --patterns periodic palindrome dyck \
  --debug
```

`--debug` composes one full sample per registered pattern and prints, for each one, the pattern instance and the full sample. Since every insertion within a sample is identical, only one copy is printed. Confirm:

- `total length` equals `--max-context-length`.
- `n_insertions` is non-zero (otherwise your generator may be raising or producing an empty list).
- The `pattern` row shows the structural property you intended.
- Total signal (`n_insertions * pattern_length`) covers at least `--signal-floor` of the context (default 50%; except for `dyck` / `shuffle_dyck`, which fill the entire sample as a single expression).

A copy of the printed output is also written to `debug.logs` in the current working directory.

For human-readable inspection, add `--mode tokens` to see the decoded strings instead of integer IDs.

## Selecting patterns

Use `--patterns` to control which patterns are generated:

- `--patterns all` (default) — generate every registered pattern.
- `--patterns periodic palindrome` — generate only the named patterns.

The argument accepts one or more pattern names. An unknown name causes an immediate error listing the available choices.

## Conventions and gotchas

- **Parity / divisibility** — many patterns have natural length constraints (palindromes need an even length, `A^n B^n C^n` needs a multiple of 3). Do **not** raise on bad lengths; build the largest valid prefix you can and let `pad_to` fill the rest. See `gen_anbn` and `gen_nested` for examples.

- **Very short `target_len`** — `length_min` is validated to be `>= 2`, but a generator may still be called with `target_len == 2`. Make sure your function returns *something* sensible (even if it degrades to trivial structure). `_pad_to` will rescue length mismatches but will not fix logical bugs. **This will almost never be the case because we will train with lengths much longer than 2.**

- **Vocab size assumptions** — if your pattern needs `k` distinct symbols, get them via `sample_distinct(vocab, k, rng)` so it degrades gracefully when the vocab is small. If it strictly cannot work below some threshold, shrink `k` instead of raising. See `gen_shuffle_dyck` for the pattern. **This will almost never be the case because we will use large vocabularies.**

- **Noise / corruption** — only inject noise once the structural sequence is long enough for the structure to remain recognizable (see `gen_noisy_palindrome`, which guards on `len(out) >= 10`).

- **Determinism** — the only randomness source must be the passed-in `rng`. This guarantees that `--seed` fully reproduces a run.

## Where things live

| Concern                                        | File                                                   |
|------------------------------------------------|--------------------------------------------------------|
| Registry dict + `@register` decorator          | [`registry.py`](registry.py)                           |
| Vocab filtering + `pad_to` + `sample_distinct` | [`utils.py`](utils.py)                                 |
| Composition into a full sample                 | [`compose.py`](compose.py)                             |
| Structural patterns                            | [`generators/structural.py`](generators/structural.py) |
| Counting patterns                              | [`generators/counting.py`](generators/counting.py)     |
| Dyck / bracket patterns                        | [`generators/dyck.py`](generators/dyck.py)             |
| Baseline / control patterns                    | [`generators/baseline.py`](generators/baseline.py)     |
| Pattern registration (imports all modules)     | [`generators/__init__.py`](generators/__init__.py)     |
| CLI + `main()`                                 | [`generator.py`](generator.py)                         |
| Complexity measurement tools                   | [`tools/`](tools)                                      |
| Background, motivation, and theory             | [`logs/`](logs)                                        |

### `/tools`

Contains utilities for analyzing generated datasets:

- **`complexity.py`** — Measures gzip-based complexity metrics for pattern dataset JSONL files. Computes both global gzip complexity (compressed / uncompressed bytes over the full token stream) and mean per-sample complexity (average compression ratio per individual sample). Results are written as `.complexity.yaml` files for each analyzed directory. Useful for characterizing the information density and learnability of pattern distributions.

### `/logs`

Contains background documentation and theoretical foundations:

- **`README.md`** — Explains the motivation for pre-pretraining on structured patterns, surveys relevant literature, and discusses methods for measuring pattern complexity using Kolmogorov complexity approximation (via gzip compression ratio). Provides context for why these synthetic patterns are useful for enhancing downstream language model performance.
