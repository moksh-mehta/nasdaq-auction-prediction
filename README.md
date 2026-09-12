# Nasdaq Closing Auction Prediction

Predicting the Nasdaq official closing cross from auction imbalance messages
disseminated between 15:50 and 16:00.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Unzip the dataset (21 daily `.csv.gz` files) into `data/`, then run the notebooks
in order.

## Notebooks

| Notebook | Purpose | Output |
|---|---|---|
| `01_dataset.ipynb` | Explore the raw files, define the dependent variable, clean the panel | `data/clean.parquet` |
| `02_analysis.ipynb` | Construct 16 features, identify what predicts the target | `data/features.parquet` |
| `03_modeling.ipynb` | Fit and evaluate on a date-based split | — |

## Target

`y_bps = 1e4 × (cross / mid − 1)` — the return from the prevailing mid at message
time to the official closing cross. This is the P&L of entering a position after
15:50 and exiting with an auction order at 16:00.

Predicting the *level* of `cross` is not a real problem: the current mid alone
explains ~99.9% of its variance. All predictable structure lives in the residual,
and the residual is the P&L.

## Result

Ridge on 16 features scores a zero-R² of **0.1213** out-of-sample on 5 held-out
days, against **0.1015** for the exchange's own `ref_price` used raw. Sign accuracy
58.9%.

That +0.020 is the whole result: most of what is knowable about the cross is already
in `ref_price`, and the auction imbalance features add about a fifth on top of it.

Signal is heavily concentrated near the close — zero-R² 0.235 in the final minute
against roughly zero before 15:55.

A gradient-boosting model scores higher on the test set (0.1433) but loses to ridge
on all three cross-validation folds, so it is reported and rejected rather than
selected. See the end of `03_modeling.ipynb`.

## Method notes

- Splits are **by date**, never random — consecutive messages for the same
  symbol-day are near-duplicates and a random split leaks badly.
- Scored as R² against a **zero** prediction, not the sample mean.
- `cv()` and `fit_test()` are separate functions so the test set cannot be used for
  model selection by accident. It is scored once, for one model.
- Feature clipping bounds are fitted on training data only.
- Fold-by-fold results are reported alongside test results, since 21 days is a small
  effective sample.
