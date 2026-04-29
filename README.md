# Patterns

## How do I add a new pattern?

Patterns live in a single registry inside [generator.py](generator.py). Adding a new one is a three-step process: write a generator function, decorate it with `@_register`, and (optionally) verify with `--debug`.

## 1. The generator contract

Every pattern is a plain function with the signature:

```python
def gen_<name>(vocab: list[int], target_len: int, rng: random.Random) -> list[int]:
    ...
```

Rules:

- **Input** — `vocab` is the filtered list of token IDs to draw from, `target_len` is the desired sequence length (drawn from `[length_min, length_max]` by the caller), and `rng` is a seeded `random.Random` instance. **Always use `rng`**, never the `random` module directly, so runs stay reproducible under `--seed`.
- **Output** — a `list[int]` of length **exactly** `target_len`. The easiest way to guarantee this is to build the structural prefix and then call the helper `_pad_to(out, target_len, vocab, rng)` at the end. `_pad_to` truncates if too long and tail-pads with random vocab IDs if too short.
- **No side effects** — do not print, do not write files, do not mutate `vocab`.
- **Use `sample_distinct(vocab, k, rng)`** when you need `k` distinct token IDs (e.g. for the `A`, `B`, `C` symbols of `A^n B^n C^n`). It falls back gracefully if the vocab is smaller than `k`.

## 2. Register the pattern

Decorate the function with `@_register(name, description)`:

```python
@_register(
    "my_pattern",
    "One-sentence description of what structural property this pattern tests "
    "(e.g. 'long-range agreement', 'modular arithmetic').",
)
def gen_my_pattern(vocab, target_len, rng):
    # build the structural part
    a, b = sample_distinct(vocab, 2, rng)
    out = [a, b] * (target_len // 2)
    # always end with _pad_to to guarantee exact length
    return _pad_to(out, target_len, vocab, rng)
```

The decorator inserts an entry into the global `PATTERNS` dict, so the new pattern is automatically picked up by `compose_sample`, the debug printer, and the main write loop. **No other file or function needs to change.**

Naming convention: lowercase, snake_case, descriptive of the structural property (`palindrome`, `counting_anbn`, `shuffle_dyck`). The name appears verbatim in each record's `metadata.pattern_type`.

## 3. Verify with `--debug`

```bash
python generator.py \
  --tokenizer gpt2 \
  --max-context-length 64 \
  --length-min 4 --length-max 16 \
  --samples-per-pattern 1 \
  --debug
```

`--debug` composes one full sample per registered pattern (background + multiple insertions) and prints the first inserted instance with surrounding context. Confirm:

- `total length` equals `--max-context-length`.
- `n_insertions` is non-zero (otherwise your generator may be raising or producing an empty list).
- The `first pattern` slice shows the structural property you intended.

For human-readable inspection, add `--mode tokens` to see the decoded strings instead of integer IDs.

## Conventions and gotchas

- **Parity / divisibility** — many patterns have natural length constraints (palindromes need an even length, `A^n B^n C^n` needs a multiple of 3). Do **not** raise on bad lengths; build the largest valid prefix you can and let `_pad_to` fill the rest. See `gen_anbn` and `gen_nested` for examples.

- **Very short `target_len`** — `length_min` is validated to be `>= 2`, but a generator may still be called with `target_len == 2`. Make sure your function returns *something* sensible (even if it degrades to trivial structure). `_pad_to` will rescue length mismatches but will not fix logical bugs. **This will almost never be the case because we will train with lengths much longer than 2.**

- **Vocab size assumptions** — if your pattern needs `k` distinct symbols, get them via `sample_distinct(vocab, k, rng)` so it degrades gracefully when the vocab is small. If it strictly cannot work below some threshold, shrink `k` instead of raising. See `gen_shuffle_dyck` for the pattern. **This will almost never be the case because we will use large vocabularies.**

- **Noise / corruption** — only inject noise once the structural sequence is long enough for the structure to remain recognizable (see `gen_noisy_palindrome`, which guards on `len(out) >= 10`).

- **Determinism** — the only randomness source must be the passed-in `rng`. This guarantees that `--seed` fully reproduces a run.

## Where things live

| Concern                        | Location in `generator.py`          |
|--------------------------------|---------------------------------------------|
| Registry dict                  | `PATTERNS: Dict[str, Tuple[str, Callable]]` |
| Decorator                      | `_register(name, description)`              |
| Length helper                  | `_pad_to(out, target_len, vocab, rng)`      |
| Distinct-symbol helper         | `sample_distinct(vocab, k, rng)`            |
| Composition into a full sample | `compose_sample(...)`                       |
| Debug printer                  | the `if args.debug:` branch in `main()`     |

## Disclaimer

Yes, this is a monolithic monstrosity of a file. We can refactor later if it gets unwieldy, but for now it's nice to have everything in one place (at least for me).

That's it.
