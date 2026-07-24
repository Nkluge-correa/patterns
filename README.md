# Complexity-Guided Pre-Pretraining — Patterns

This is a simple codebase for generating synthetic sequence data with structural patterns. The patterns are designed to probe various capabilities a sequence model may need (locality, symmetry, counting, recursion, agreement, etc.) and can be used for "pre-pretraining".

> - *Pre-pretraining* is training that comes before the traditional language modeling pretraining stage. The goal is to give a model a head start on learning structural properties that might be useful for language modeling, so that the model can focus on learning the actual language distribution during pretraining rather than having to discover these properties from scratch.

## The pattern catalogue

These are the currently registered 14 patterns we have. All of these can be tunned in term of the vocabulary size, context length, and other properties that are pattern-specific.

| Pattern                   | Schematic example                   | What it probes                                                                  |
|---------------------------|-------------------------------------|---------------------------------------------------------------------------------|
| `periodic`                | `ABCABCABC`                         | Fixed-period repetition (regular language).                                     |
| `copy`                    | `ABCD ABCD`                         | Block duplication / verbatim copying.                                           |
| `counting_anbn`           | `AAABBB`                            | Equal counts of two symbols (CFG counting `a^n b^n`).                           |
| `counting_anbncn`         | `AAABBBCCC`                         | Equal counts of three symbols (mildly context-sensitive `a^n b^n c^n`).         |
| `interleaving`            | `ABABAB` or `AABBAABB`              | Alternation / block-interleaving of two symbols.                                |
| `permutation_cycle`       | `ABCD BCDA CDAB DABC`               | Cyclic permutations of a base block.                                            |
| `hierarchical`            | `ABAB CCCC ABAB`                    | Local + global structure mixed at multiple scales.                              |
| `dyck`                    | `(()())`                            | Dyck-1: balanced brackets of a single type.                                     |
| `shuffle_dyck`            | `([{}])`                            | Nested typed Dyck-k: closer must match the most recently opened type.           |
| `random`                  | `qZ7ξ%`                             | Uniformly random tokens — unstructured baseline / control.                      |
| `identity`                | `AAAAAA`                            | Single-token repetition (zero-entropy floor).                                   |
| `composite_mirror_repeat` | `ABCCBA ABCCBA`                     | Multi-rule composition: a small symmetric block repeated periodically.          |
| `mixer`                   | `[periodic][copy][etc]`             | Context filled with consecutive segments from different pattern types.          |
| `nca`                     | `<grid> . </grid> <grid> . </grid>` | Stochastic neural cellular automaton rollout.                                   |

* Letters in the schematics stand for *distinct vocabulary tokens*; concrete IDs are sampled per call so different samples use different surface tokens.

## How to Use?

```bash
# See python generator.py --help for all options
python generator.py \
        --patterns all \
        --vocab-size 256 \
        --max-context-length 1024 \
        --samples-per-pattern 50000 \
        --output patterns.jsonl \
        --output-dir ./data \
        --no-metadata \
        --seed 42
```

Each pattern writes its shards into a dedicated subdirectory named after the pattern, under the directory inferred from `--output`. For example, with `--output out/patterns.jsonl`:

```
out/
  periodic/
    patterns.0000.jsonl
  copy/
    patterns.0000.jsonl
  dyck/
    patterns.0000.jsonl
  ...
```

Each record is a JSON line. By default it includes full metadata:

```json
{"input_ids": [...], "metadata": {"pattern_type": "periodic", "vocab_size": 50257, "max_context_length": 64, "n_insertions": 1, "insertions": [{"start": 0, "length": 64}]}}
```

With `--no-metadata`, only `input_ids` is written (useful when the files are consumed directly by a training pipeline):

```json
{"input_ids": [...]}
```

## Quirks and Conventions

### Unique token counts

Not every pattern uses the full `--vocab-size` range. Because PAD_ID (`0`) is stripped from the content vocabulary for most patterns, and because some generators always fill the context window exactly (never needing tail padding), the actual number of distinct token IDs that appear in the output can be, for example, 255 rather than 256. For the patterns that manage their own reserved IDs (`dyck`, `shuffle_dyck`, and `nca`), there are also quirks related to padding and other special tokens. See `python generator.py --help` fort more details, or check the raw implementation of each generator in `generators/` for the exact rules.

You can also use [`tools/count_vocab.py`](tools/count_vocab.py) to audit unique token counts after generation:

```bash
python tools/count_vocab.py --data-dir ./data/
```

> **Note**: the unique token count is important to estimate the proper vocab size for a model trained on these patterns. Making sure the model's vocab size matches the actual number of distinct token IDs in the data makes the interpretation of perplexity more intuitive, i.e., if the model does not break the ln(1/actual_vocab_size) mark, then it is learning nothing beyond uniform guessing.

### Controlling NCA difficulty (regimes)

The raw NCA CNN rule emits very small logits, so at temperature `1.0` with zero identity bias the per-cell softmax is almost uniform and the automaton degenerates into a near-RNG with **no learnable signal** (the next cell state is essentially independent of the grid). `generators/nca.py` exposes a single `_REGIME` knob that selects a `(temperature, identity_bias)` preset placing the dynamics in a labelled difficulty band. Each label is the approximate **oracle next-cell loss** (the best cross-entropy a model that perfectly learned the rule could reach) as a fraction of the uniform baseline `ln(d_state)` — lower means more predictable and easier to learn:

| `_REGIME`        | (temperature, identity_bias) | ~oracle loss / `ln(d_state)` | character                              |
|------------------|------------------------------|------------------------------|----------------------------------------|
| `paper`          | `(1e-3, 0.0)` *(default)*    | near-deterministic           | reference paper setup (Lee et al. 2026)|
| `learnable_25`   | `(0.5, 2.0)`                 | ~25%                         | clear local structure                  |
| `learnable_50`   | `(0.2, 0.0)`                 | ~50%                         | hard but learnable                     |
| `unlearnable`    | `(1.0, 0.0)`                 | ~99%                         | control / baseline (near-RNG)          |

Numbers are measured at `grid=8, d_state=8, _SHARED_RULE_SEED=42` via [`tools/validate.py`](tools/validate.py); re-measure with that tool after changing `_GRID_SIZE`, `_D_STATE`, or the seed. To change difficulty, edit the single line in `generators/nca.py`:

```python
_REGIME = "paper"   # one of: paper, learnable_25, learnable_50, unlearnable
```

### Controlling shuffle_dyck difficulty (k)

The `shuffle_dyck` pattern generates nested Dyck-k sequences where `k` bracket types must be correctly matched (a closer must match the most recently opened bracket *type*). The difficulty scales directly with `k`: more bracket types mean a larger set of valid closers to distinguish, making the hierarchical dependency harder to learn. The number of bracket types is controlled by the hardcoded `K` constant in `generators/dyck.py`:

```python
K = 8   # number of bracket types for shuffle_dyck
```

The token budget increases with `k` — the pattern needs `2k` bracket IDs plus one pad token (so `2k + 1` IDs total). When `_SHARED_IDS` is `True` (the default), these are fixed IDs `1..2k` regardless of `--vocab-size`; when `False`, fresh random distinct IDs are drawn per sample from the supplied vocab (and `k` degrades gracefully if the vocab is too small).

## Training on the patterns

For training a model, we use the [`llm-foundry`](https://github.com/Polygl0t/llm-foundry).

## Stay Up to Date

Read our experimental logs in [`logs/README.md`](logs/README.md). All runs are visible on this WandB workspace: [https://wandb.ai/bonn/Patterns](https://wandb.ai/bonn/Patterns).
