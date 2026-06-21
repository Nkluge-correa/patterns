# Epiplexity Report: `fineweb-edu-670m`

## ⚠️ Warnings
- 17 steps have loss BELOW final training loss (2.6291); the model may still be improving.

## Inputs

| Parameter | Value |
|---|---|
| Model parameters | 670,000,000 |
| Total training tokens | 5,200,000,000 |
| Total training FLOPs | 2.09e+19 FLOPs |
| Loss-curve points | 10,000 |
| Initial training loss | 11.0803 nats/token |
| Final training loss | 2.6291 nats/token |
| Final validation loss | 2.6954 nats/token |

## Epiplexity Decomposition

| Quantity | Symbol | Per-token (bits) | Total |
|---|---|---|---|
| **Epiplexity** (structural info) | $S_T$ | **0.6533** | 3.40e+09 bits |
| **Time-bounded entropy** (random info) | $H_T$ | **3.8887** | depends on test-set size |
| **Total information** | $S_T + H_T$ | **4.5420** | — |

* **Epiplexity $S_T$** — measures the structural, learnable information the model absorbed from the training data.
* **Time-bounded entropy $H_T$** — the remaining per-token unpredictability in the test data.
* **$S_T + H_T$** — together they sum to the total time-bounded information content.
* **High $S_T / (S_T+H_T)$ ratio** — means the data contains rich learnable structure beyond surface statistics.

## Derived Ratios

| Metric | Value |
|---|---|
| Structural fraction $S_T / (S_T+H_T)$ | **14.38%** |
| Language / data entropy floor | 1.8000 nats (2.5969 bits) |
| $H_T$ as fraction of entropy floor | 149.75% |
| Gap to entropy floor | 0.8954 nats (1.2918 bits) |
| $S_T$ per 1B training tokens | 653,263,789 bits |

* **Structural fraction** — the proportion of total information that is learnable structure (vs. irreducible randomness).
* **Language / data entropy floor** — the estimated Shannon entropy of the data source — the minimum achievable loss.
* **$H_T$ as fraction of entropy floor** — shows how close the model is to this minimum: 100% means the model has reached the entropy floor; values above 100% mean there is still room to improve.
* **$S_T$ per 1B tokens** — normalizes epiplexity for cross-dataset comparison.

## Reference Metrics

| Metric | Value |
|---|---|
| Gzip complexity | 0.3766 |

* **Gzip complexity** (compressed / original bytes) — approximates total Kolmogorov complexity. Unlike epiplexity, it cannot separate random noise from learnable structure: both incompressible noise and richly structured data score near 1.0.
* **Oracle next-token loss** — the irreducible entropy of the generating process — only knowable for synthetic patterns.

## Interpretation

**Moderate structural fraction** — the data contains meaningful learnable structure.  The model likely built non-trivial internal circuits that *may* transfer to downstream tasks.
