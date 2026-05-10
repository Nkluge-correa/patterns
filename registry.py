"""Pattern registry: maps pattern names to (description, generator_fn) pairs.

Usage in a generator module
----------------------------
    from registry import register

    @register("my_pattern", "Short description of the pattern.")
    def gen_my_pattern(vocab, target_len, rng):
        ...

The decorated function is inserted into PATTERNS automatically on import.
Any module that imports PATTERNS after the generator modules have been
imported will see all registered patterns.
"""

from typing import Callable, Dict, Tuple

# Maps pattern_name -> (description, generator_fn).
# Generator signature: (vocab: list[int], target_len: int, rng: Random) -> list[int]
PATTERNS: Dict[str, Tuple[str, Callable]] = {}


def register(name: str, description: str):
    """Decorator that registers a generator function under `name`."""
    def deco(fn: Callable) -> Callable:
        PATTERNS[name] = (description, fn)
        return fn
    return deco
