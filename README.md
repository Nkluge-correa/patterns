# Patterns Are All You Need

This is a simple codebase for generating synthetic sequence data with structural patterns. The patterns are designed to probe various capabilities a sequence model may need (locality, symmetry, counting, recursion, agreement, etc.) and can be used for "pre-pretraining" before exposing the model to natural language.

The codebase is split across a few focused modules — see [Where things live](#where-things-live) below.

## The pattern catalogue

The 18 currently registered patterns and what each is meant to test:


| Pattern                   | Schematic example                   | What it probes                                                                  |
|---------------------------|-------------------------------------|---------------------------------------------------------------------------------|
| `periodic`                | `ABCABCABC`                         | Fixed-period repetition (regular language).                                     |
| `palindrome`              | `ABCCBA`                            | Mirror symmetry around the center (CFG-recognizable).                           |
| `copy`                    | `ABCD ABCD ABCD`                    | Block duplication / verbatim copying.                                           |
| `reverse`                 | `ABCD \| DCBA`                      | Source + reverse separated by an explicit delimiter.                            |
| `counting_anbn`           | `AAABBB`                            | Equal counts of two symbols (CFG counting `a^n b^n`).                           |
| `counting_anbncn`         | `AAABBBCCC`                         | Equal counts of three symbols (mildly context-sensitive `a^n b^n c^n`).         |
| `nested`                  | `ABCDDCBA`                          | Recursive palindromic structure from `S → a S a`.                               |
| `interleaving`            | `ABABAB` or `AABBAABB`              | Alternation / block-interleaving of two symbols.                                |
| `permutation_cycle`       | `ABCD BCDA CDAB DABC`               | Cyclic permutations of a base block.                                            |
| `hierarchical`            | `ABAB CCCC ABAB`                    | Local + global structure mixed at multiple scales.                              |
| `noisy_palindrome`        | `ABCXCBA` (~10% corrupted)          | Palindrome under random token corruption (robustness to noise).                 |
| `dyck`                    | `(()())`                            | Dyck-1: balanced brackets of a single type.                                     |
| `shuffle_dyck`            | `([{}])`                            | Nested typed Dyck-k: closer must match the most recently opened type.           |
| `random`                  | `qZ7ξ%`                             | Uniformly random tokens — unstructured baseline / control.                      |
| `identity`                | `AAAAAA`                            | Single-token repetition (zero-entropy floor).                                   |
| `composite_mirror_repeat` | `ABCCBA ABCCBA`                     | Multi-rule composition: a small palindrome repeated periodically.               |
| `mixer`                   | `[periodic][palindrome][etc]`       | Context filled with consecutive segments from different pattern types.          |
| `nca`                     | `<grid> . </grid> <grid> . </grid>` | Stochastic neural cellular automaton rollout.                                   |

* Letters in the schematics stand for *distinct vocabulary tokens*; concrete IDs are sampled per call so different samples use different surface tokens.

## How samples are composed

`compose_sample` is what turns a generator into a full training example of length
`--max-context-length`. The generator is called **once** with
`target_len == max_context_length` and produces structure spanning the whole
context window. There is no random background and no verbatim block repetition.

Every pattern receives a **content vocabulary** with the reserved pad token
(see below) removed, so that token ID `0` can only ever appear as the trailing
slack appended by `pad_to`. Three patterns (`dyck`, `shuffle_dyck`, `nca`) manage
reserved IDs internally and receive the full vocabulary including ID 0.

### The reserved pad token (ID 0)

Token ID `0` (defined as `PAD_ID` in `utils.py`) is a **reserved pad token** used
exclusively for the trailing slack when a generator's structural output doesn't
divide evenly into `max_context_length`. It never appears as genuine pattern
content (generators receive a vocab with ID 0 already stripped). Its loss must
be **masked during training** — it carries zero learnable signal. This mirrors
the convention already used by the `dyck`, `shuffle_dyck`, and `nca` generators
from the outset.

### Theoretical loss floors

The `tools/validate.py` script now reports, for every simple pattern, the
**minimum achievable next-token loss** (\"floor\") — the best cross-entropy a model
that perfectly learned the rule could reach. This lets you compare any trained
model's loss against the true entropy floor of its data distribution. Run with
`python tools/validate.py`. See the `/tools` section below for a summary of each
oracle method.

### Symbol invariance

Pattern symbols are drawn fresh per sample (the *rule* is constant but the
*values* change), which forces the model to learn symbol-invariant structural
rules (reflection, counting, cyclic shift, …) rather than memorizing fixed
token sequences. This is intentional: the task is to infer the generative law
from in-context evidence, not to remember a specific sequence.

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
{"input_ids": [...], "metadata": {"pattern_type": "periodic", "vocab_size": 50257, "max_context_length": 64, "n_insertions": 1, "insertions": [{"start": 0, "length": 64}]}}
```

With `--no-metadata`, only `input_ids` is written — useful when the files are consumed directly by a training pipeline:

```json
{"input_ids": [...]}
```

- **Dyck patterns (`dyck`, `shuffle_dyck`)** — the entire sample is a *single* valid Dyck expression of length exactly `max_context_length`. This is intentional: a Dyck expression is only meaningful as a whole (its brackets must balance globally), so splicing fragments would destroy the structural property. Dyck manages the reserved pad token (ID 0) internally.

- **`nca`** — each sample is a single rollout of a stochastic neural cellular automaton (a tiny CNN with toroidal padding and per-cell categorical sampling). The rollout is serialised as consecutive frames `[<grid>, row-major cells, </grid>]`. Three vocab IDs are reserved: `0` is a dedicated **pad token**, `1` and `2` are the `<grid>` / `</grid>` delimiters, and cell states map to the remaining IDs (`3 .. 3 + d_state`). The 8×8 default grid yields a frame size of 66 tokens, so `--max-context-length` must be at least `132` (two full frames); all powers of 2 ≥ 256 fit comfortably. Any leftover slack is padded with the **pad token (ID 0)** — never an orphan delimiter — and its loss is masked during training.

### Controlling NCA difficulty (regimes)

The raw NCA CNN rule emits very small logits, so at temperature `1.0` with zero identity bias the per-cell softmax is almost uniform and the automaton degenerates into a near-RNG with **no learnable signal** (the next cell state is essentially independent of the grid). `generators/nca.py` exposes a single `_REGIME` knob that selects a `(temperature, identity_bias)` preset placing the dynamics in a labelled difficulty band. Each label is the approximate **oracle next-cell loss** (the best cross-entropy a model that perfectly learned the rule could reach) as a fraction of the uniform baseline `ln(d_state)` — lower means more predictable and easier to learn:

| `_REGIME`        | (temperature, identity_bias) | ~oracle loss / `ln(d_state)` | character                       |
|------------------|------------------------------|------------------------------|---------------------------------|
| `unlearnable`    | `(1.0, 0.0)`                 | ~99%                         | control / baseline (near-RNG)   |
| `learnable_50`   | `(0.2, 0.0)` *(default)*     | ~50%                         | hard but learnable              |
| `learnable_25`   | `(0.5, 2.0)`                 | ~25%                         | clear local structure           |
| `easy`           | `(0.1, 0.0)`                 | ~4%                          | near-deterministic              |

Numbers are measured at `grid=8, d_state=8, _SHARED_RULE_SEED=42` via [`tools/validate.py`](tools/validate.py); re-measure with that tool after changing `_GRID_SIZE`, `_D_STATE`, or the seed. To change difficulty, edit the single line in `generators/nca.py`:

```python
_REGIME = "learnable_50"   # one of: unlearnable, learnable_50, learnable_25, easy
```

### Controlling pattern complexity

Some generators have module-level flags that reduce the difficulty of the
distribution by removing a source of per-sample randomness.  They are meant
to be toggled *before* generation starts (e.g. in the caller script or
config), not changed mid-run.

| Pattern        | Flag           | Default | Effect when `True`                                                                      |
|----------------|----------------|---------|-----------------------------------------------------------------------------------------|
| `nca`          | `_SHARED_RULE` | `False` | All samples evolve under the **same** randomly-initialised NCA network (fixed seed 42). |
| `dyck`         | `_SHARED_IDS`  | `False` | Open/close tokens are always `1, 2` (ID `0` is the pad) instead of a fresh random pair. |
| `shuffle_dyck` | `_SHARED_IDS`  | `False` | Bracket tokens are always `1..k` (openers) and `k+1..2k` (closers); ID `0` is the pad.  |

- **Why this helps.**  With the default per-sample resampling, a model must simultaneously learn the structural rules *and* re-discover which tokens carry the structure on every sample (a meta-learning problem).  Fixing the dynamics (`_SHARED_RULE`) or the token alphabet (`_SHARED_IDS`) collapses this to learning a single, consistent system — much easier for small or early-stage models — while still producing diverse sequences through
stochastic updates and varying initial conditions.

When using the CLI, toggle the flags near the top of `generator.py`:

```python
# generator.py (near the imports)
generators.nca._SHARED_RULE = True   # one NCA network for all samples
generators.dyck._SHARED_IDS = True   # dyck brackets=1,2; shuffle=1..2k; pad=0
```

When calling the generators from your own script, set them directly on the
imported module:

```python
import generators.nca as nca
import generators.dyck as dyck

nca._SHARED_RULE = True
dyck._SHARED_IDS = True
```

## How do I add a new pattern?

Adding a new pattern is a three-step process: write a generator function in the right module, decorate it with `@register`, and (optionally) verify with `--debug`.

### 1. The generator contract

Every pattern is a plain function with the signature:

```python
def gen_<name>(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    ...
```

Rules:

- **Input** — `vocab` is the filtered list of token IDs to draw from, `target_len` is the desired sequence length (always equal to `--max-context-length`; the generator is called exactly once per sample), and `rng` is a seeded `random.Random` instance. **Always use `rng`**, never the `random` module directly, so runs stay reproducible under `--seed`.
- **Output** — a `list[int]` of length **exactly** `target_len`. The easiest way to guarantee this is to build the structural prefix and then call the helper `pad_to(out, target_len, vocab, rng)` at the end. `pad_to` truncates if too long and tail-pads with the reserved `PAD_ID` (token ID `0`, whose loss must be masked during training).
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
| `generators/nca.py`        | neural cellular automaton rollouts         |

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
  --vocab-size 32 \
  --max-context-length 32 \
  --samples-per-pattern 1 \
  --patterns all \
  --debug
```

To verify only specific patterns, pass their names to `--patterns`:

```bash
python generator.py \
  --vocab-size 32 \
  --max-context-length 32 \
  --samples-per-pattern 1 \
  --patterns periodic palindrome dyck \
  --debug
```

`--debug` composes one full sample per registered pattern and prints the pattern
instance and the full sample. Every sample is a single whole-context instance
(`n_insertions = 1`). Confirm:

- `total length` equals `--max-context-length`.
- `n_insertions` is `1` (a single structured instance; any trailing pad is part
  of the same insertion).
- The `pattern` row shows the structural property you intended.
- No `0` tokens appear inside the pattern body (pad only ever fills the tail).

A copy of the printed output is also written to `debug.log` in the current working directory.

## Selecting patterns

Use `--patterns` to control which patterns are generated:

- `--patterns all` (default) — generate every registered pattern.
- `--patterns periodic palindrome` — generate only the named patterns.

The argument accepts one or more pattern names. An unknown name causes an immediate error listing the available choices.

## Conventions and gotchas

- **Reserved pad token (ID 0)** — token ID `0` is reserved as a loss-masked pad. Generators receive a content vocabulary with ID `0` already removed (handled by `compose_sample`), so they can freely use `rng.choice(vocab)` or `sample_distinct(vocab, …)` without ever emitting the pad. The only source of pad in a sample is the trailing slack from `pad_to`, and the only patterns that see ID 0 in their `vocab` argument are `dyck`, `shuffle_dyck`, and `nca` (which manage it internally).

- **Parity / divisibility** — many patterns have natural length constraints (palindromes need an even length, `A^n B^n C^n` needs a multiple of 3). Do **not** raise on bad lengths; build the largest valid prefix you can and let `pad_to` fill the rest with the reserved pad token. See `gen_anbn` and `gen_nested` for examples.

- **Very short `target_len`** — a generator may be called with a small `target_len` (e.g. during quick tests). Make sure your function returns *something* sensible (even if it degrades to trivial structure). `pad_to` will rescue length mismatches but will not fix logical bugs. **This will almost never be the case because we will train with lengths much longer than 2.**

- **Vocab size assumptions** — if your pattern needs `k` distinct symbols, get them via `sample_distinct(vocab, k, rng)` so it degrades gracefully when the vocab is small. If it strictly cannot work below some threshold, shrink `k` instead of raising. See `gen_shuffle_dyck` for the pattern. **This will almost never be the case because we will use large vocabularies.**

- **Noise / corruption** — only inject noise once the structural sequence is long enough for the structure to remain recognizable (see `gen_noisy_palindrome`, which guards on `len(out) >= 10`).

- **Determinism** — the only randomness source must be the passed-in `rng`. This guarantees that `--seed` fully reproduces a run.

## Where things live

| Concern                                        | File                                                     |
|------------------------------------------------|----------------------------------------------------------|
| Registry dict + `@register` decorator          | [`registry.py`](registry.py)                             |
| Vocab filtering + `pad_to` + `sample_distinct` | [`utils.py`](utils.py)                                   |
| Composition into a full sample                 | [`compose.py`](compose.py)                               |
| Structural patterns                            | [`generators/structural.py`](generators/structural.py)   |
| Counting patterns                              | [`generators/counting.py`](generators/counting.py)       |
| Dyck / bracket patterns                        | [`generators/dyck.py`](generators/dyck.py)               |
| Baseline / control patterns                    | [`generators/baseline.py`](generators/baseline.py)       |
| Neural cellular automaton patterns             | [`generators/nca.py`](generators/nca.py)                 |
| Pattern registration (imports all modules)     | [`generators/__init__.py`](generators/__init__.py)       |
| CLI + `main()`                                 | [`generator.py`](generator.py)                           |
| Complexity + validation tools                  | [`tools/`](tools)                                        |
| Parquet → JSONL converter                      | [`tools/parquet_to_jsonl.py`](tools/parquet_to_jsonl.py) |
| Background, motivation, and theory             | [`logs/`](logs)                                          |

### `/tools`

Contains utilities for analyzing generated datasets:

- **`complexity.py`** — Measures gzip-based complexity metrics for pattern dataset JSONL files. Computes both global gzip complexity (compressed / uncompressed bytes over the full token stream) and mean per-sample complexity (average compression ratio per individual sample). Operates in two modes:
  - **Token mode** (default) — reads `input_ids` and casts to the appropriate unsigned integer dtype (uint8/16/32) based on vocabulary size.
  - **Text mode** (`--text-column`) — operates directly on raw text from a named JSONL column, UTF-8 encoding it to a byte stream with a fixed 256-symbol alphabet. Automatically selected when `input_ids` is absent.

  Results are written as `.complexity.yaml` files for each analyzed directory. Useful for characterizing the information density and learnability of pattern distributions.

- **`parquet_to_jsonl.py`** — Converts Parquet dataset files to JSONL format for use with `complexity.py` and other tools.

- **`validate.py`** — Sanity-checks all generators and reports theoretical loss floors. For Dyck (`dyck`, `shuffle_dyck`) it confirms every sample is a balanced/nested-valid word of exact length and reports the **oracle achievable loss** (the best cross-entropy a model that learned the rule could reach) against the uniform baseline. For NCA it verifies frame structure and pad-tail correctness and prints the full **difficulty-regime ladder** (oracle next-cell loss vs `ln(d_state)`). For simple structural/counting/baseline patterns it uses a **collision-free vocabulary replay** to compute the exact minimum achievable loss — the true training floor — per pattern (see the table in the "How samples are composed" section above). Run with `python tools/validate.py`.

### Output metadata

`generator.py` writes two kinds of metadata alongside the shards:

- **Per-record metadata** — each JSONL record carries a `metadata` key (unless `--no-metadata` is set) with `pattern_type`, `vocab_size`, `max_context_length`, insertion positions, and other generation parameters.

- **Per-pattern `.metadata`** — after finishing a pattern, a YAML file named `.metadata` is written to the pattern's output directory (e.g. `out/periodic/.metadata`) summarizing:
  ```yaml
  samples: 1000
  tokens: 1024000
  tokens_per_chunk: 100000000
  chunks: 1
  block_size: 1024
  columns: [input_ids, metadata]
  ```
  This file is a quick-reference manifest for downstream tooling and dataset catalogues.

### `/logs`

Contains background documentation and theoretical foundations:

- **`README.md`** — Explains the motivation for pre-pretraining on structured patterns, surveys relevant literature, and discusses methods for measuring pattern complexity using Kolmogorov complexity approximation (via gzip compression ratio). Provides context for why these synthetic patterns are useful for enhancing downstream language model performance.
