# Patterns Are All You Need

## Main Idea

Using patterns as a "surrogate language" in "pre-pretraining" can significantly enhance the performance of language models on downstream tasks. In essence, this approach involves teaching the model to understand simple and complex patterns before exposing them to natural language. This exposure allows the model to learn useful representations related to sequence modeling, which can be beneficial for various downstream tasks (e.g., language understanding, reasoning, etc.).

Some papers already explored this idea:

* [Program Synthesis using Natural Language](https://arxiv.org/abs/1509.00413)
* [Analysing Mathematical Reasoning Abilities of Neural Models](https://arxiv.org/abs/1904.01557)
* [Pre-Training a Language Model Without Human Language](https://arxiv.org/abs/2012.11995)
* [Injecting structural hints: Using language models to study inductive biases in language learning](https://arxiv.org/abs/2304.13060)
* [LifeGPT: Topology-Agnostic Generative Pretrained Transformer Model for Cellular Automata](https://arxiv.org/abs/2409.12182)

The most promising results come from the last two papers:

* [Training Language Models via Neural Cellular Automata](https://arxiv.org/html/2603.10055v1) -> Uses Neural Cellular Automata (NCA) to generate complex patterns for pre-pretraining. Just 160M tokens of NCA-generated data can lead to significant improvements in downstream performance.
* [Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases](https://arxiv.org/html/2502.19249v2) -> Uses formal languages (e.g., Dyck languages) to pre-pretrain language models, showing that this can impart linguistic biases and improve performance on downstream tasks.

Bottom line:

> "Richly structured non-linguistic data could also be effective for teaching models useful capabilities, and may be more efficient to learn from than natural language data."

## Why this is cool for low-resource / data constrained settings

Data is not always abundant. For many low-resource languages, domains, or tasks, we may not have access to large amounts of natural language data. In such cases, pre-pretraining on synthetic patterns can provide a valuable alternative. By learning to recognize and generate patterns, the model can develop a strong foundation in sequence modeling, which can then cascade into improved performance on downstream tasks, even with limited natural language data. Since this type of data can be generated in "unlimited" quantities, it can be a powerful tool for enhancing model performance in data-constrained settings.

## What we would like to investigate

* What types of patterns are most effective for pre-pretraining?
* How does the complexity of the patterns affect downstream performance?
* How does the amount of pattern data used for pre-pretraining impact downstream performance?
* How does pre-pretraining on patterns compare to pre-pretraining on natural language data in terms of downstream performance?

Other things that would be interesting to investigate:

* How model arquitecture (e.g. transformer vs. hybrid models) interacts with pre-pretraining on patterns.
* How different types of downstream tasks (e.g. language understanding vs. mathematical reasoning) are affected by pre-pretraining on patterns.
* How different language families (e.g. english vs. chinese) are affected by pre-pretraining on patterns.

## How we can measure complexity of patterns

The most straightforward way to measure the complexity of a pattern is to look at its [Kolmogorov complexity](https://liamzebedee.com/maths/papers/kolomogrov-tables-random-numbers.pdf) (i.e., a good approximation of the length of the shortest program that can generate the pattern, since $K$ is uncomputable).

## Example

* Generate a sequence of tokens using some function (e.g. a neural cellular automata, a formal language, etc.).
* Serialize and compress the sequence using **gzip**.
* We can measure the **complexity** of a sequence by the ratio of compressed size to original size:

$$\text{complexity} = \frac{\text{compressed bytes}}{\text{original bytes}}$$

* Or, we can also talk about the compression ratio, which is the inverse of complexity:

$$\text{compression ratio} = \frac{\text{original bytes}}{\text{compressed bytes}}$$

The intuition is that gzip compression approximates **Kolmogorov complexity**:

* Highly compressible sequences contain regular, predictable structure.
* Poorly compressible sequences are more chaotic and information-rich.

Note: ***"matching the complexity of synthetic pre-training data to the target domain maximizes transfer."**** This means that if we want to improve performance on natural language tasks, we should pre-pretrain on patterns that have a similar level of complexity to natural language data ([source](https://arxiv.org/abs/2603.10055)).

## Experimental setup

A proper ablation study would involve:

* Training a language model from scratch on a fixed text dataset for a fixed number of steps / tokens.
  * 10 billion tokens of natural language data could be a good place to start.
  * [Fineweb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu#smaller-sample-versions) is a good candidate for the text dataset since it is relatively small and clean.
  * [Open-Web-Math](https://huggingface.co/datasets/open-web-math/open-web-math) is a good candidate for the text dataset if we want to test on more math-heavy data.
  * [CodeParrot](https://huggingface.co/datasets/codeparrot/codeparrot-clean) is a good candidate for the text dataset if we want to test on more code-heavy data.
  * [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) is a good candidate for the text dataset if we want to test on more than just English data.
    * Suggestions of languages that are very different from one another:
      * English (Indo-European, SVO word order, Latin script)
      * Chinese (Sino-Tibetan, SVO word order, logographic script)
      * Arabic (Afro-Asiatic, VSO word order, abjad script)
      * Hindi (Indo-European, SOV word order, Devanagari script)
      * Japanese (Japonic, SOV word order, mixed script)
* Pre-pretraining the model on a fixed amount of pattern data before training on the text dataset.
* Compare how our evaluations differ as we tweak the level of complexity of the pre-pretraining patterns, and amount of pre-pretraining data used.
* We could also ablate different proportions of the pre-pretraining data (e.g. 0%, 25%, 50%) and natural language data (e.g. 100%, 75%, 50%) to see how the two interact.
* We should keep comparisons fair by controlling for the total amount of training data (pattern + natural language) and total number of training steps / tokens across all conditions.

Note: Is very important to control all other aspects of the training process (e.g. model architecture, hyperparameters, etc.) to ensure that any differences in performance can be attributed to the pre-pretraining patterns.

* For model architecture, we use a simple transformer-based language model (e.g. Llama2) to keep things manageable.
* For sizes, we can test a coulple of scales (e.g., 500M, 1,5B, 3B) to see how the effects of pre-pretraining on patterns scale with model size.
  * We expect that the benefits of pre-pretraining on patterns will be more pronounced for smaller models, and decrease as model size increases (see [source](https://arxiv.org/html/2603.10055v1#S5)).
* A softmax-attention transformer vs. a hybrid model (e.g., Olmo3-Hybrid) would be interesting to compare, but maybe we can save that for a follow-up project.
* For the pre-pretraining patterns, we don't really need a tokenizer since we can just generate sequences of token IDs directly.
* When we move to natural language pretraining, we should:
    * Use a standard tokenizer from HuggingFace.
    * We re-initialize (and re-size) the model's embedding layer to match the tokenizer's vocabulary size, and randomly initialize the new parameters.
    * If we follow the results and insights from [source](https://arxiv.org/html/2603.10055v1#S5), we should only maintain the attention weights accross the pre-pretraining and pre-training phases, and re-initialize all other parameters (e.g. feedforward layers, layer norms, etc.) to ensure that any benefits we see are due to the attention patterns learned during pre-pretraining. ***"[...] attention layers learn general-purpose mechanisms for tracking dependencies and inferring latent rules, while MLP layers specialize in storing domain-specific patterns and statistics. This division may explain why attention transfers universally from NCA to language, whereas MLP weights can introduce interference when the source and target domains differ substantially."***

* Training hyperparameters:

| Hyperparameter              | Pre-pre-training | Pre-training                                  |
|-----------------------------|------------------|-----------------------------------------------|
| Effective batch size        | 16               | 512                                           |
| Sequence length             | 1024 tokens      | 1024 tokens                                   |
| Learning rate               | $1\times10^{-4}$ | $5\times10^{-4}$                              |
| LR schedule                 | Cosine w/ warmup | Cosine w/ warmup                              |
| Warmup steps (% total)      | 10%              | 10%                                           |
| Weight decay                | None             | $1\times10^{-4}$                              |
| Gradient clipping           | None             | 1.0                                           |

Note: We should run every training condition with multiple random seeds (e.g. 3-5) to ensure that our results are robust and not due to random chance.

## How can we measure the "usefulness" of patterns for pre-pretraining?

Performance. We can measure a couple of things:

* Final perplexity on a held out validation set of natural language data. We can also use different types of natural language data (e.g. normal text vs. code vs. math) to see how pre-pretraining on patterns affects the performance on different types of downstream data.
* Convergence speed during pre-training. We can track the training loss and perplexity on the natural language data during pre-training to see if pre-pretraining on patterns leads to faster convergence.
* Simple benchmarks that have good "signal" at early stages of training (e.g. [HellaSwag](https://arxiv.org/abs/1905.07830), [PIQA](https://arxiv.org/abs/1911.11641), [LAMBADA](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), etc.). This allows us to track the improvement in downstream performance.

We could but would be more complicated:

* Finetune on a downstream task (e.g. [GSM8K](https://arxiv.org/abs/2205.12646)) and evaluate performance on that task. This would be more troublesome to do, but not too difficult if we stick to a simple SFT and tasks that have a good train/test split.

## What patterns should we use for pre-pretraining?

### Baseline patterns

These are simple patterns that can be generated with a small amount of code and are intended to serve as unstructured baselines or controls. Random will by definition have the highest complexity, while identity will have the lowest complexity (zero-entropy floor):

| Pattern                   | Schematic example          | What it probes                                                                  |
|---------------------------|----------------------------|---------------------------------------------------------------------------------|
| `random`                  | `qZ7ξ%`                    | Uniformly random tokens (unstructured baseline / control).                      |
| `identity`                | `AAAAAA`                   | Single-token repetition (zero-entropy floor).                                   |

### Simple patterns

These are simple patterns that can be generated with a small amount of code and are itended to (maybe?) teach the model some basic capabilities related to sequence modeling:

| Pattern                   | Schematic example               | What it probes                                                                  |
|---------------------------|---------------------------------|---------------------------------------------------------------------------------|
| `periodic`                | `ABCABCABC`                     | Fixed-period repetition (regular language).                                     |
| `palindrome`              | `ABCCBA`                        | Mirror symmetry around the center (CFG-recognizable).                           |
| `copy`                    | `ABCD ABCD ABCD`                | Block duplication / verbatim copying.                                           |
| `reverse`                 | `ABCD \| DCBA`                  | Source + reverse separated by an explicit delimiter.                            |
| `counting_anbn`           | `AAABBB`                        | Equal counts of two symbols (CFG counting `a^n b^n`).                           |
| `counting_anbncn`         | `AAABBBCCC`                     | Equal counts of three symbols (mildly context-sensitive `a^n b^n c^n`).         |
| `nested`                  | `ABCDDCBA`                      | Recursive palindromic structure from `S → a S a`.                               |
| `interleaving`            | `ABABAB` or `AABBAABB`          | Alternation / block-interleaving of two symbols.                                |
| `permutation_cycle`       | `ABCD BCDA CDAB DABC`           | Cyclic permutations of a base block.                                            |
| `hierarchical`            | `ABAB CCCC ABAB`                | Local + global structure mixed at multiple scales.                              |
| `noisy_palindrome`        | `ABCXCBA` (~10% corrupted)      | Palindrome under random token corruption (robustness to noise).                 |
| `composite_mirror_repeat` | `ABCCBA ABCCBA`                 | Multi-rule composition: a small palindrome repeated periodically.               |
| `mixer`                   | `[periodic][palindrome][etc]`   | Context filled with consecutive segments from different pattern types.          |

### Complex patterns

#### Dyck languages

Dyck languages are formal languages that consist of balanced strings of parentheses (or brackets) of various types. They are a classic example of context-free languages and are often used to test the ability of models to learn hierarchical structures and long-range dependencies. The simplest Dyck language (Dyck-1) consists of balanced parentheses of a single type, while more complex versions (Dyck-k) involve multiple types of parentheses that can interleave freely.

| Pattern                   | Schematic example                   | What it probes                                                                  |
|---------------------------|-------------------------------------|---------------------------------------------------------------------------------|
| `dyck`                    | `(()())`                            | Dyck-1: balanced brackets of a single type.                                     |
| `shuffle_dyck`            | `( [ ) { } ]`                       | Typed Dyck-k: k bracket types whose open/close tokens may interleave freely.    |

#### Neural Cellular Automata (NCA)

NCA is a generalization of classical cellular automata (Wolfram, [1984](https://www.sciencedirect.com/science/article/pii/0167278984902458)), where the update rule is parametrized as a neural network, allowing the dynamics to be diversely sampled rather than hand-designed.

| Pattern                   | Schematic example                   | What it probes                                                                  |
|---------------------------|-------------------------------------|---------------------------------------------------------------------------------|
| `nca`                     | `<grid> . </grid> <grid> . </grid>` | Stochastic neural cellular automaton rollout.                                   |

> "NCAs, despite having simple local rules, can generate arbitrarily complex structures when rolled out over long time horizons, making them a promising source of synthetic training data." ([source](https://arxiv.org/html/2603.10055v1))

**Methodology (this implementation)**

We use 2D neural cellular automata (NCA) to generate sequences whose complexity is controlled by the universal gzip-based filter shared with every other pattern (`--min-complexity`).

* The system operates on an **8×8 grid** with periodic (toroidal) boundaries and **8 possible cell states**. Each cell is encoded as an **8-dimensional one-hot vector**.
  - The grid is intentionally tiny so a handful of rollout steps fills typical context windows (256–4096 tokens).
  - 8 cell states comfortably fit inside the project's 256-ID vocabulary budget while leaving room for two reserved delimiter tokens.
* Cell updates are determined by a neural network ($f_\theta$), which looks at each cell's **3×3 neighborhood** (with circular padding) and predicts the next state using a softmax distribution with temperature $\tau = 1.0$:

$$c_i(t+1) \sim \mathrm{softmax}\left(f_{\theta}(c_{\mathcal{N}(i)}(t))/\tau\right)$$

* The transition model consists of:

  * a **3×3 convolution layer** with 4 channels (`VALID` padding applied to a circularly-padded input),
  * a **1×1 convolution** lifting to 16 channels,
  * **ReLU**,
  * a final **1×1 convolution** producing logits over the 8 possible next states.

To create diverse behaviors:

* For every rollout, both the neural network parameters ($\theta$) and the initial grid states are sampled fresh from a torch RNG seeded deterministically from the caller's `random.Random` instance, so every sample reflects a different dynamical rule while the overall run remains reproducible under `--seed`.
* Weights are drawn from a LeCun-style normal ($\mathrm{std} = 1/\sqrt{\mathrm{fan\_in}}$); biases are zero.
* Initial cell states are sampled per cell from a softmax over an 8-dimensional standard-normal vector (i.e. a spatially-uniform categorical prior whose class probabilities vary across rules).
* The first **4 rollout steps are discarded as burn-in** so the recorded trajectory captures the rule's attractor rather than the random initial condition.
* This produces dynamics ranging from stable and predictable patterns to highly chaotic ones.
* Complexity filtering is delegated to the shared `--min-complexity` flag (gzip compression ratio of the final flattened sample); the same threshold (≥ 0.5 ≈ compression ratio ≤ 2.0) used in the NCA paper can be reproduced by passing `--min-complexity 0.5 --patterns nca`.

In terms of tokenization:

* We use **direct per-cell tokens** rather than the 2×2 patch tokenization from the reference paper (Lee et al. [2026](https://arxiv.org/html/2603.10055v1)). The project's shared vocabulary is the contiguous range $[0, 256)$, and the paper's patch scheme would inflate the effective vocab (e.g. $10^4 = 10000$ patch tokens for $k=10$ states) far beyond that budget. Each cell state maps bijectively to a single token ID via a simple offset: `cell_state s -> vocab[s + 2]`. So state `0 → 2`, state `1 → 3`, ..., state `7 → 9`. With an 8×8 grid this yields $64$ cell tokens per frame.
* We serialize each timestep in **row-major order** (left-to-right, top-to-bottom) wrapped in reserved `<grid>` / `</grid>` delimiter tokens  (the first two vocab IDs, i.e., `0` and `1`) so the per-frame payload is $1 + 64 + 1 = 66$ tokens. The delimiters and the cell states live in disjoint ID ranges, so a frame is unambiguously parseable.
* Each NCA sample uses only IDs `{0, 1}` (delimiters) and `{2..9}` (cell states), which trivially fits inside the project's `[0, 256)` vocabulary budget.
* Frames are concatenated to fill `--max-context-length`. The minimum context is **two full frames (132 tokens)**; all powers of two >= 256 fit at least 3 frames (256 -> 3, 512 -> 7, 1024 -> 15, 2048 -> 31, 4096 -> 62). We pack as many *complete* frames as fit and never emit a half-frame (which would leave an orphan delimiter); any residual tail (fewer than 66 tokens) is filled with random cell-state IDs so the sample lands at exactly `--max-context-length`.

## Why Should Any of This Work?

* **Patterns will force genuine rule learning.** Unlike natural language, this kind of data contain no semantic shortcuts or co-occurrence priors. Every sequence is generated by a hidden deterministic rule, so the model cannot rely on memorization or semantic associations (see https://arxiv.org/abs/2303.09540). To predict the next token, it must infer the underlying rule from context. This makes these patterns a valuable training signal for in-context learning.
* **This is the same core mechanism used in language modeling.** Prior work suggests that language models implicitly infer latent concepts or rules within a sequence. Predicting text requires conditioning on those inferred structures. Similar mechanisms appear in math, code, and formal algorithmic tasks. The hypothesis is that pre-pretraining strengthens this general-purpose inference ability (see https://arxiv.org/abs/2406.04216).
* **Transfer occurs through attention-based in-context learning circuits.** Transferable knowledge is primarily stored in attention layers, not MLPs. They connect this to “induction heads,” attention circuits known to support in-context learning by copying and extending patterns from earlier tokens. These learned attention mechanisms can then transfer to downstream domains (see https://arxiv.org/abs/2209.11895).
* **Deterministic systems can still produce learnable structure (epiplexity).** Even though these patterns are deterministic, they can generate complex signals that finite-capacity models cannot simply brute-force simulate. According to the concept of “epiplexity,” models must learn higher-level abstractions and representations to predict these systems efficiently. Training on diverse and complex patterns may therefore help models learn abstract representations that are also useful for natural language (see https://arxiv.org/html/2601.03220).

## Experiment log

### 2026-06-16: Data-generation bugfixes: dyck pad token, nested matching, and NCA pad + regime

**Context.** Initial pre-pretraining experiments with ~50M-parameter models showed the models plateauing at a loss floor that seemed higher than the patterns' true entropy. We suspected either insufficient model capacity or bugs in the data-generation pipeline. Systematic audit of the generators revealed two independent sets of design deficiencies in `dyck.py` and `nca.py` that needed correction before pre-pretraining could produce meaningful signal.

#### dyck.py

**Problem 1: No pad token; truncation instead of masking.**

Both `gen_dyck` and `gen_shuffle_dyck` returned `sequence[:target_len]`. If the greedy loop overshot `target_len` (which it routinely did), the sequence was silently truncated. While the truncated content was still valid Dyck structure, the approach didn't use the project-wide `PAD_ID` (ID 0) convention, so any parity-remainder slack wasn't loss-masked. This also meant ID 0 could appear as a legitimate bracket token instead of being reserved as the pad.

**Problem 2: `_SHARED_IDS` used ID 0 as a content token.**

When `_SHARED_IDS=True`, the old code mapped openers to `vocab[0], vocab[1]` for dyck and `vocab[:k]` for shuffle_dyck. Since ID 0 is the project-wide pad token, this would have placed bracket tokens in the pad slot, making it impossible to distinguish content from padding during training.

**Problem 3: Naive budget handling in dyck.**

The old generator used `depth < target_len // 2` as a heuristic for whether it could open, then appended all closers in a cleanup loop and truncated. This could produce sequences where the last token was a truncated balanced structure: learnable but messy, and not guaranteed to produce a valid balanced prefix up to `target_len`.

**Problem 4: `shuffle_dyck` was a true shuffle language, not nested.**

The old `shuffle_dyck` maintained a flat `counts` array of open-bracket counts per type and used `rng.choice(open_types)` to pick which closer to emit. This meant any closer type was equally valid for any open bracket — a *shuffle* Dyck language. In a shuffle language, the closer *type* is not deterministic from context (it's a uniform choice among open types), so the model faces irreducible entropy on closer prediction. This inflates the loss floor and obscures whether the model is actually learning hierarchical structure. The corrected version uses a proper stack discipline: a closer must match the most recently opened (top-of-stack) bracket type, making the closer type a deterministic function of the context — a genuine hierarchical dependency the model must track.

#### nca.py

**Problem 1: Tail padded with delimiter tokens, counted in loss.**

The old code filled leftover slack with `vocab[0]`. Since `_OPEN_IDX=0` and `vocab[0]` was the `<grid>` delimiter, the tail consisted of repeated `<grid>` tokens that were counted in the training loss. A model could trivially predict `<grid>` after a certain position, creating a spurious learnable artifact unrelated to the NCA dynamics.

**Problem 2: Delimiters at IDs 0 and 1 collided with the project-wide pad convention.**

`_OPEN_IDX=0`, `_CLOSE_IDX=1`, `_N_RESERVED=2` meant the `<grid>` delimiter occupied ID 0: the same slot that should be reserved project-wide as `PAD_ID`. This would have placed `<grid>` tokens in the pad slot, making them loss-masked (invisible to training) and breaking the frame structure.

**Problem 3: Default regime was unlearnable.**

`_TEMPERATURE=1.0` and `_IDENTITY_BIAS=0.0` produced near-uniform noise (~99% of ln(d_state)). At this setting, the NCA's next-cell transition is essentially independent of the grid state: there is negligible learnable signal. Any model training on this
would plateau at the uniform-entropy baseline regardless of capacity, making it useless as a pre-pretraining signal. The corrected version introduces a "learnable_50" regime with `_TEMPERATURE=0.2` and `_IDENTITY_BIAS=0.0`, which produces a more challenging but learnable signal (~50% of ln(d_state)) that encourages the model to track the grid state without being overwhelmed by noise.

We also introduce a regime ladder with multiple presets (unlearnable, learnable_50, learnable_25, easy) to allow systematic exploration of how NCA complexity affects pre-pretraining transfer.

| Regime         | (temperature, identity_bias) | ~oracle loss / ln(d_state) | Interpretation                                        |
|----------------|------------------------------|----------------------------|-------------------------------------------------------|
| `unlearnable`  | (1.0, 0.0)                   | ~99%                       | Near-uniform noise; no learnable signal (old default) |
| `learnable_50` | (0.2, 0.0)                   | ~50%                       | Hard but learnable (new default)                      |
| `learnable_25` | (0.5, 2.0)                   | ~25%                       | Clear local structure                                 |
| `easy`         | (0.1, 0.0)                   | ~4%                        | Near-deterministic                                    |

**Open questions.**

- Does the nested constraint make shuffle_dyck *too* easy? The deterministic closer type means the model only needs to learn stack discipline, not bracket-type disambiguation.
- The NCA regime was calibrated to produce a learnable signal when the grid size and d_state are fixed at 8, but this should be tuned if those parameters change. Should we add a `--nca-regime` flag to allow systematic exploration of NCA complexity?

### 2026-06-17: Data-generation bugfixes: masked pad, whole-context fill, and counting randomization

**Context.** After fixing the dyck and NCA generators (see 2026-06-16), we turned to the "simple" pattern pipeline (structural, counting, baseline) and found three additional independent design deficiencies that did not exist in the now-corrected `dyck` / `nca`
generators, but were equally damaging to pre-pretraining signal.

**Problem 1: ~50% of every sample was irreducible uniform noise.**

`compose.py` spliced a single pattern instance into a uniform-random background filling ~50% of the context (`signal_floor = 0.5`). That background was iid uniform over all V vocab IDs and was counted in the training loss, contributing a fixed ~2.7 nats (at V=256) that no model could reduce. This pinned the observable training loss far above the pattern's true entropy floor, making it impossible to tell whether the model was actually learning the rule.

**Problem 2: The task collapsed to "locate-and-copy."**

`compose.py` stamped the *exact same pattern instance* verbatim at 45–60 non-overlapping positions. The only thing a model needed to do was locate one copy and echo it for all others: the structural distinction between a palindrome, a periodic block, and a counting  sequence was invisible to the loss objective. The model wasn't learning any generative law; it was learning a block-copy shortcut.

**Problem 3: Counting patterns degenerated into positional lookups.**

`counting_anbn` and `counting_anbncn` used `n = target_len // k`, which placed the symbol-switch boundary at a fixed, predictable position (e.g., always the midpoint for `k=2`). A model could solve this with a positional "first half = symbol A, second half = symbol B" heuristic: no actual counting required.

**Problem 4: `pad_to` used random filler instead of a masked pad token.**

The `pad_to` helper in `utils.py` filled divisibility slack with `rng.choice(vocab)` (random content IDs). The dyck and nca fixes (2026-06-16) had already established ID 0 as a dedicated, loss-masked pad; the simple patterns did not yet adopt this convention, so any trailing slack was counted in the loss as unpredictable noise.

**Problem 5: Minor robustness issues.**

- `noisy_palindrome` could theoretically corrupt pad positions (harmless today because `gen_palindrome` always fits exactly, but fragile).
- `mixer` sub-generators' pad tails leaked into the mixer body (fixed by stripping pad before concatenation).

All these issues were fixed, but we are still unsure if this will work, i.e., the models will learn the intended rules rather than finding shortcuts or being overwhelmed by noise. We created a `tool/validate.py` script to replay the generators and measure the actual oracle next-token entropy of each pattern type, which will help us set realistic expectations for the loss floors and interpret the training curves. See its current output in [`tools/validate.logs`](../tools/validate.logs).

**Open questions.**

- The counting floors (0.18 / 0.13 nats) are very low: will models actually learn the counting mechanism, or will they find a shortcut?
- `noisy_palindrome` and `mixer` floors are loose lower bounds; we should measure the actual oracle next-token entropy via exact replay of the generating process (as done for `dyck` / `shuffle_dyck` / `counting_*`).

### 2026-06-18: Epiplexity — measuring structural information beyond loss and gzip

**Context.** After fixing the data-generation pipeline, we face a measurement gap. Our current metrics — training loss, validation loss, and gzip complexity tell us *how compressible* a pattern is and *how predictable* it is once the rule is known (via `tools/validate.py`), but neither captures *how much learnable structure* a pattern contains. Two patterns can have the same validation loss yet differ radically in how many reusable circuits the model had to build to reach that loss. We need a metric that quantifies the structural information a model actually absorbs from training data, i.e., information that may transfer to downstream tasks. The **epiplexity** framework from Finzi et al. ([2026](https://arxiv.org/html/2601.03220)) could provide us that.

#### What is epiplexity?

**Epiplexity** ($S_T$) is the amount of *structural, learnable information* that a computationally bounded observer can extract from data. It decomposes the total information content of a dataset into two complementary components:

| Component                | Symbol   | Meaning                                                                 | What it captures                                                                  |
|--------------------------|----------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| **Epiplexity**           | $S_T(X)$ | Structural patterns the observer *can* learn and compress into a model  | Grammar rules, symmetries, long-range dependencies, emergent dynamics             |
| **Time-bounded entropy** | $H_T(X)$ | Inherently unpredictable randomness under the observer's compute budget | CSPRNG output, uniform noise, irreducible stochasticity in the generating process |

The total time-bounded information is $S_T + H_T$. The formal definition uses a two-part Minimum Description Length (MDL) objective under a fixed runtime budget $T$:

$$S_T(X) = |\mathrm{P}^\star|,\quad \mathrm{P}^\star = \arg\min_{\mathrm{P} \in \mathcal{P}_T} \left\{ |\mathrm{P}| + \mathbb{E}\left[-\log P(X)\right] \right\}$$

where $\mathcal{P}_T$ is the set of probabilistic models evaluable in time $T$.

Given that gzip complexity approximates Kolmogorov complexity (total information content), and validation loss approximates the irreducible entropy (unpredictability), epiplexity fills the gap by quantifying the learnable structure that lies between these two extremes, i.e., would help to differentiate two datasets with the *compressability* of gzip but different *predictability*. As far as I know, the relation of epiplexity to pre-pretraining has not been explored before, but it seems like a promising lens for understanding why certain patterns are more effective for transfer than others.

#### Why gzip complexity is insufficient

Our current complexity metric (`tools/complexity.py`) measures:

$$\text{complexity} = \frac{\text{compressed bytes}}{\text{original bytes}}$$

This approximates **Kolmogorov complexity** — the *total* information content, not the *structural* component. It cannot distinguish three fundamentally different cases that all score the same:

| Data                                    | Gzip complexity | $S_T$ (epiplexity) | $H_T$ (time-bounded entropy) | Why they differ                                                       |
|-----------------------------------------|-----------------|--------------------|------------------------------|-----------------------------------------------------------------------|
| `random` (uniform noise)                | $\approx 1.0$   | $\approx 0$        | $\approx n$ (maximal)        | Incompressible because it's actually random                           |
| Richly structured (e.g., `dyck`, `nca`) | $\approx 1.0$   | **high**           | moderate                     | Incompressible because it encodes complex learnable structure         |
| `palindrome`                            | $\approx 1.0$   | low                | moderate                     | "Incompressible" because LZ77's sliding window misses mirror symmetry |

Gzip conflates these three cases. Epiplexity will disambiguate them: `random` has $S_T \approx 0$ (nothing to learn), `dyck` has $S_T \gg 0$ (rich structure absorbed), and `palindrome` falls somewhere in between. For pre-pretraining, we want patterns with **high $S_T$**, i.e., data that forces the model to build non-trivial internal circuits.

The three-part spectrum of all data:

```
                    High Epiplexity (S_T)
                    (complex, structured, learnable)
                           /\
                          /  \
                         /    \
                        /      \
                       /        \
                      /          \
    Low S_T, Low H_T /            \ High S_T, High H_T
    (trivial,       /              \ (natural language,
     predictable)  /                \  chess, NCA)
                  /__________________\
    Low S_T, High H_T  
    (CSPRNG, uniform noise,  
     shuffled pixels, Rule 30 ECA)
```

#### Prequential estimation methodology

True epiplexity is incomputable (it requires searching over all programs). But we can use the **prequential coding** approximation--the simplest method from Finzi et al. (2026)--which requires only a loss curve:

$$S_T(X) \approx \sum_{i=0}^{M-1} \left( \log\frac{1}{P_i(Z_i)} - \log\frac{1}{P_M(Z_i)} \right)$$

$$H_T(X) \approx \mathbb{E}\left[\log\frac{1}{P_M(X)}\right] \quad\text{(final validation loss × dataset size)}$$

In plain terms:

- **$S_T$** is the **area between the training loss curve and the final training loss**, accumulated over all training tokens. Each step where the model's loss exceeds its final floor contributes excess nats — this excess *is* the structural information the model absorbed.
- **$H_T$** is the **final validation loss** — the irreducible per-token unpredictability that remains even after the model has extracted all learnable structure.

The time bound $T$ is the total FLOPs spent: $T = 6ND + 2N\mathcal{D}$ (forward + backward passes for $N$ parameters, $D$ training tokens, $\mathcal{D}$ test tokens).

**Key properties of the prequential estimate:**

| Property                                          | Implication                                                                                                                              |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| Convex loss curves (steep early drop → long tail) | The area is concentrated in early training; $S_T$ captures rapid rule discovery                                                          |
| $S_T$ is an **upper bound** on true epiplexity    | We can only overestimate, never underestimate — conservative for ranking                                                                 |
| $S_T$ is **observer-relative**                    | Depends on model size $N$ and compute budget $T$; a pattern that looks random to a 1M-param model may show structure to a 1B-param model |
| $S_T$ saturates                                   | Once the model has extracted all learnable structure, further training adds negligible $S_T$ — the loss curve flattens                   |

#### Implementation: `tools/epiplexity.py`

We implemented the prequential estimator as a standalone CLI tool that computes $S_T$ and $H_T$ from training artifacts.

#### Example: FineWeb-Edu 670M baseline

As a reference point, we ran the tool on our existing 670M-parameter model trained on 5.2B tokens of FineWeb-Edu natural language data. The full report is at [`logs/runs/fineweb-edu-670m/epiplexity.md`](runs/fineweb-edu-670m/epiplexity.md).

| Metric                                   | Value                 | Interpretation                                                                                                |
|------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------------|
| $S_T$ (epiplexity)                       | **0.6533 bits/token** | The model absorbed ~0.65 bits of structural information per training token                                    |
| $H_T$ (time-bounded entropy)             | **3.8887 bits/token** | ~3.89 bits/token of irreducible unpredictability remains                                                      |
| Structural fraction $S_T / (S_T + H_T)$  | **14.38%**            | Natural language is ~14% learnable structure, ~86% irreducible entropy at this scale                          |
| $S_T$ per 1B tokens                      | **653M bits**         | Normalized for cross-dataset comparison                                                                       |
| Gap to language entropy floor (1.8 nats) | **0.8954 nats**       | The model is still well above the estimated entropy floor — more training would help                          |
| Gzip complexity                          | 0.3766                | Natural language is moderately compressible by gzip, but $S_T$ reveals the *structural* fraction specifically |

This gives us a natural-language baseline: when we pre-pretrain on patterns and then transfer to FineWeb-Edu, we can compare $S_T$ of the pre-pretraining phase against this 0.65 bits/token figure, and track whether structural information acquired during pre-pretraining is conserved or amplified in the downstream phase.

#### How epiplexity maps onto our research questions

| Our question                                                | Epiplexity's answer                                                                                                                                                                |
|-------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Which patterns are most "useful" for pre-pretraining?       | Patterns with **high $S_T$** — not just low loss. High $S_T$ means the model had to build non-trivial internal circuits.                                                           |
| How does pattern complexity affect downstream performance?  | $S_T$ quantifies structural complexity *as absorbed by the model* — more relevant than gzip or oracle loss alone.                                                                  |
| Is gzip complexity a good proxy for "learnable complexity"? | **No.** Gzip conflates noise and structure. $S_T$ separates them. A quadrant plot ($S_T$ vs. gzip) would reveal which patterns are high-structure vs. merely incompressible noise. |
| How much pattern data is enough?                            | $S_T$ saturates as the model extracts all available structure. When $S_T$ stops growing with more data, the pattern is "exhausted."                                                |
| Does pre-pretraining build transferable circuits?           | Measure $S_T$ during pre-pretraining, then measure it again during NL training. If pattern-acquired $S_T$ is conserved or amplified, transfer is occurring.                        |
