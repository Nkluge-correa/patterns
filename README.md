# Patterns Are All You Need

This is a simple codebase for generating synthetic sequence data with structural patterns. The patterns are designed to probe various capabilities a sequence model may need (locality, symmetry, counting, recursion, agreement, etc.) and can be used for "pre-pretraining" before exposing the model to natural language.

The codebase is split across a few focused modules — see [Where things live](#where-things-live) below.

## Where things live

| Concern                                                 | File                                                       |
|---------------------------------------------------------|------------------------------------------------------------|
| Generators and their implementation contract            | [`generators/`](generators)                                |
| Pattern registration (imports all modules)              | [`generators/__init__.py`](generators/__init__.py)         |
| Baseline / control patterns                             | [`generators/baseline.py`](generators/baseline.py)         |
| Counting patterns                                       | [`generators/counting.py`](generators/counting.py)         |
| Dyck / bracket patterns                                 | [`generators/dyck.py`](generators/dyck.py)                 |
| Mixer pattern                                           | [`generators/mixer.py`](generators/mixer.py)               |
| Neural cellular automaton patterns                      | [`generators/nca.py`](generators/nca.py)                   |
| Structural patterns                                     | [`generators/structural.py`](generators/structural.py)     |
| Background, motivation, and theory                      | [`logs/`](logs)                                            |
| Recorded complexity measurements                        | [`logs/measurements/`](logs/measurements)                  |
| Recorded runs and training logs                         | [`logs/runs/`](logs/runs)                                  |
| Complexity + Epiplexity + validation tools              | [`tools/`](tools)                                          |
| Complexity measurement                                  | [`tools/complexity.py`](tools/complexity.py)               |
| Count unique tokens generated per pattern               | [`tools/count_vocab.py`](tools/count_vocab.py)             |
| Calculate the Pareto frontier of epiplexity vs. compute | [`tools/epiplexity_pareto.py`](tools/epiplexity_pareto.py) |
| Epiplexity calculation and reporting                    | [`tools/epiplexity.py`](tools/epiplexity.py)               |
| Parquet -> JSONL converter                              | [`tools/parquet_to_jsonl.py`](tools/parquet_to_jsonl.py)   |
| Validation of generators (Oracles)                      | [`tools/validate.py`](tools/validate.py)                   |
| Composition into a full sample                          | [`compose.py`](compose.py)                                 |
| CLI + `main()` (create samples here)                    | [`generator.py`](generator.py)                             |
| Registry dict + `@register` decorator                   | [`registry.py`](registry.py)                               |
| Vocab filtering + `pad_to` + `sample_distinct`          | [`utils.py`](utils.py)                                     |

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

## Quirks and Conventions

### How samples are composed?

`compose_sample` is what turns a generator into a full training example of length `--max-context-length`. The generator is called **once** with `target_len == max_context_length` and produces structure spanning the whole context window.

Every pattern receives a **content vocabulary** with the reserved pad token (see below) removed, so that token ID `0` can only ever appear as the trailing slack appended by `pad_to`. Three patterns (`dyck`, `shuffle_dyck`, `nca`) manage reserved IDs internally and receive the full vocabulary including ID 0.

Token ID `0` (defined as `PAD_ID` in `utils.py`) is a **reserved pad token** used exclusively for the trailing slack when a generator's structural output doesn't divide evenly into `max_context_length`. It never appears as genuine pattern content (generators receive a vocab with ID 0 already stripped). Its loss must be **masked during training**.

### Unique token counts

Not every pattern uses the full `--vocab-size` range. Because PAD_ID (`0`) is stripped from the content vocabulary for most patterns, and because some generators always fill the context window exactly (never needing tail padding), the actual number of distinct token IDs that appear in the output can be 255 rather than 256 — or far fewer for the patterns that manage their own reserved IDs (`dyck`, `shuffle_dyck`, `nca`).

Use [`tools/count_vocab.py`](tools/count_vocab.py) to audit unique token counts after generation:

```bash
python tools/count_vocab.py --data-dir ./data/
```

The docstring of [`generator.py`](generator.py) includes a reference table of expected counts per pattern for `--vocab-size 256` and `--max-context-length 4096`.

> **Note**: the unique token count is important to estimate the proper vocab size for a model trained on these patterns. Making sure the model's vocab size matches the actual number of distinct token IDs in the data makes the interpretation of perplexity more intuitive, i.e., if the model does not break the ln(1/actual_vocab_size) mark, then it islearning nothing beyond uniform guessing.

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

Some generators have module-level flags that reduce the difficulty of the distribution by removing a source of per-sample randomness.  They are meant
to be toggled *before* generation starts.

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

- **Input**: `vocab` is the filtered list of token IDs to draw from, `target_len` is the desired sequence length (always equal to `--max-context-length`; the generator is called exactly once per sample), and `rng` is a seeded `random.Random` instance. **Always use `rng`**, never the `random` module directly, so runs stay reproducible under `--seed`.
- **Output**: a `list[int]` of length **exactly** `target_len`. The easiest way to guarantee this is to build the structural prefix and then call the helper `pad_to(out, target_len, vocab, rng)` at the end. `pad_to` truncates if too long and tail-pads with the reserved `PAD_ID` (token ID `0`, whose loss must be masked during training).
- **No side effects**: do not print, do not write files, do not mutate `vocab`.
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
  --patterns my_pattern \
  --debug
```

`--debug` composes one full sample per registered pattern and prints the pattern instance and the full sample. Every sample is a single whole-context instance (`n_insertions = 1`). Confirm:

- `total length` equals `--max-context-length`.
- `n_insertions` is `1` (a single structured instance; any trailing pad is part of the same insertion).
- The `pattern` row shows the structural property you intended.
- No `0` tokens appear inside the pattern body (pad only ever fills the tail).

A copy of the printed output is also written to `debug.log` in the current working directory.
