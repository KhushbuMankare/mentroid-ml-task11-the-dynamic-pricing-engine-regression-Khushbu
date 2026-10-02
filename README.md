# Dynamic Pricing Engine — Regression

Predict a continuous sale price from three market variables, with an honest investigation of what the synthetic noise (`noise=15`) does and does not allow.

> **All numbers below come from running `python dynamic_pricing.py`** (full console log in `run_output.txt`). Nothing is hand-edited.

---

## 1. Problem Statement

Given a competitor's price, our inventory level and a demand score, predict the `Sale_Price`. The task is a regression problem evaluated with **Mean Absolute Error (MAE)** — the average size of the pricing mistake in price units.

## 2. Dataset

```python
X, y = make_regression(n_samples=1000, n_features=3, noise=15, random_state=42)
```

| Property | Value |
|---|---|
| Samples | 1000 (800 train / 200 test) |
| Features | 3 |
| `noise` | 15 |
| `random_state` | 42 (dataset and split are fully reproducible) |

### What `noise=15` actually means
`make_regression` builds the target as a **linear combination of the features plus Gaussian noise**:

```
y = X · true_coefficients + ε,   ε ~ Normal(mean = 0, standard deviation = 15)
```

- It is the **standard deviation of random noise added to every row**, not a list of corrupted rows.
- No observation is "wrong"; some rows simply received a larger random draw.
- Consequence: the noise is **irreducible**. Even a model that knows the true coefficients exactly has an expected MAE of `15 × √(2/π) ≈ 11.97` and RMSE ≈ 15. This is the **noise floor** that all results should be judged against.

## 3. Features

The assignment's mapping is used as-is. The data is synthetic, so the names are labels, not real market behaviour.

| Column | Source | Relationship learned |
|---|---|---|
| `Competitor_Price` | `X[:, 0]` | Positive; largest effect (~98 price units per +1 unit) |
| `Inventory_Level` | `X[:, 1]` | Positive here (~83 per +1 unit). In a real market, higher stock often *lowers* price — this sign is an artefact of random synthetic generation, **not** a business insight |
| `Demand_Score` | `X[:, 2]` | Positive; smallest effect (~26 per +1 unit) |
| `Sale_Price` | `y` | Target. Synthetic range is about −426 to +489, i.e. it includes negative "prices", another reminder it is not real pricing data |

## 4. Methodology

```
Data Generation (fixed by assignment)
      ↓
Train-Test Split (80/20, test set locked away)
      ↓
Baseline Models (Linear, Ridge)           ─┐
      ↓                                    │  all scored with repeated
Model Comparison (RF, Gradient Boosting)   │  5-fold CV (x3) on TRAIN only
      ↓                                    │
Robust Regression (Huber)                  │
      ↓                                    │
Residual Analysis (train residuals)        │
      ↓                                    │
Noise/Robustness Experiments               │
 (filtering, winsorisation – applied      ─┘
  inside each CV fold, train data only)
      ↓
Final Model chosen by CV + simplicity rule
      ↓
Fit on full train → evaluate ONCE on untouched test set
      ↓
Final MAE + Actual vs Predicted plot
```

**Anti-leakage rules enforced in code**
- Test set is not used for any decision. Test MAEs of other approaches are printed only *after* the final choice, labelled "informational".
- Residual filtering and winsorisation are recomputed **inside each CV fold** from that fold's training portion; the validation fold is never altered.
- Same CV folds for every approach, so comparisons are **paired**.
- **Decision rule (set before running):** a more complex approach replaces the simple reference only if it lowers CV MAE by **≥ 1%** *and* by **> 2 paired standard errors**.

## 5. Model Comparison

Cross-validation on the training set (5-fold × 3 repeats, 15 scores per row). Sorted by CV MAE.

| Approach | CV MAE (mean ± std) | Gain vs Linear | Meaningful? | Test MAE* |
|---|---|---|---|---|
| Linear Regression + filter \|r\|>3σ | 11.809 ± 0.474 | +0.009% | No | 12.312 |
| **Linear Regression (selected)** | **11.810 ± 0.474** | reference | — | **12.307** |
| Ridge (alpha via RidgeCV) | 11.810 ± 0.474 | −0.005% | No | 12.306 |
| Linear + winsor 1/99 | 11.812 ± 0.475 | −0.015% | No | 12.304 |
| Linear + winsor 2/98 | 11.812 ± 0.478 | −0.019% | No | 12.306 |
| Huber Regression | 11.815 ± 0.477 | −0.048% | No | 12.333 |
| Linear + winsor 5/95 | 11.816 ± 0.479 | −0.056% | No | 12.294 |
| Linear + filter \|r\|>2.5σ | 11.832 ± 0.490 | −0.190% | No | 12.294 |
| Linear + filter \|r\|>2σ | 11.838 ± 0.484 | −0.241% | No | 12.269 |
| Gradient Boosting | 17.067 ± 1.173 | −44.5% | No (much worse) | 16.314 |
| Random Forest | 21.956 ± 1.495 | −85.9% | No (much worse) | 22.176 |

\*Test MAE column is **informational only**, printed after selection. It was not used to choose anything. With 200 test rows, the 95% bootstrap CI half-width on MAE is about ±1.2, so differences of ~0.05 between the top nine rows are pure noise.

Why linear wins: `make_regression` generates `y` as an exactly linear function of the features. A linear model has the correct inductive bias; trees approximate a straight line with staircases, and with only 800 noisy points they partly fit the noise (RF/GB are visibly worse). Ridge ≈ OLS because there are only 3 features, 800 samples and no collinearity, so there is nothing to regularise.

## 6. Noise Investigation

**Why `noise=15` was retained.** It is part of the supplied task, and the data was never regenerated, edited or subsampled.

**What residual analysis showed (training set, Linear Regression).**

| Statistic | Value | Gaussian reference |
|---|---|---|
| Residual mean | 0.000 | 0 |
| Residual std | 14.76 | ≈15 (the injected σ) |
| Skew | −0.002 | 0 |
| Excess kurtosis | −0.14 | 0 |
| Beyond 2σ (robust) | 4.88% | 4.55% |
| Beyond 2.5σ | 1.62% | 1.24% |
| Beyond 3σ | 0.12% | 0.27% |

The residuals are symmetric, bell-shaped, with tails matching a normal distribution, and their spread recovers σ≈15. The "large" residuals (max |r| ≈ 45, about 3σ) are the expected tails of Gaussian noise, **not** corrupted rows. The model has also recovered the true coefficients closely (below).

**Treatments tested (training data only, inside CV):** residual filtering at 2σ / 2.5σ / 3σ, residual winsorisation at 1/99, 2/98, 5/95 (no rows removed), and Huber regression.

**Did they help?** No. Differences versus plain Linear Regression ranged from −0.24% to +0.009%, far below the 1% threshold. The 3σ filter's "gain" of 0.001 MAE is statistically meaningless and was **not** adopted. More aggressive filtering got slightly *worse*, which is expected: removing tail points of a Gaussian shrinks the estimated noise and discards information.

**Why this makes sense.** Robust methods help when there are *outliers from a different process* (heavy tails, data-entry errors). Pure Gaussian noise has none; for Gaussian errors, OLS is the best linear unbiased estimator and Huber is marginally less efficient.

**Decision.** Plain Linear Regression, no noise treatment.

## 7. Results

| Metric (final model: Linear Regression, test set, evaluated once) | Value |
|---|---|
| **MAE** | **12.307** (95% bootstrap CI 11.10 – 13.51) |
| RMSE | 15.217 |
| R² | 0.9859 |
| Train MAE | 11.754 |
| Theoretical noise-floor MAE | ≈ 11.97 |

**Interpretation.** Test MAE is within bootstrap noise of the theoretical floor, and train/test MAE are close, so the model is essentially as good as any model can be on this data; there is no meaningful headroom to chase. The apparent gap (12.31 vs 11.97) is well inside sampling noise for 200 rows.

Learned coefficients vs the true generating coefficients (known only because `make_regression(coef=True)` exposes them; never used for tuning):

| Feature | Estimated | True |
|---|---|---|
| Competitor_Price | 98.457 | 98.357 |
| Inventory_Level | 83.457 | 81.874 |
| Demand_Score | 25.780 | 25.967 |
| Intercept | −0.669 | 0 |

Each coefficient is the expected change in `Sale_Price` for a +1 unit change in that feature with the others held fixed. Because the data is synthetic, these describe the generator, not any real market. Tree feature importances are not reported since no tree model was selected.

### Plots (`outputs/`)

| File | Content |
|---|---|
| `actual_vs_predicted.png` | Required plot: test actual vs predicted with y = x line |
| `residual_plot.png` | Test residuals vs predicted (no trend or funnel) |
| `residual_distribution.png` | Training residual histogram with normal reference |
| `model_comparison.png` | CV MAE of every approach vs noise floor |

## 8. Business Interpretation

In a real system a model like this would estimate a **reference price** from competitor price, stock position and demand signals; a rules layer would then adjust it. The synthetic data shows the workflow (clean split, honest validation, interpretable coefficients), but **it does not demonstrate real pricing behaviour**: the relationships are random, linear and noise is Gaussian by construction. A real deployment would also need to handle causality (a model predicting observed prices is not a model of how demand *responds* to price), and the MAE here has no business meaning without real units and margins.

## 9. Future Improvements (production extension, not part of this dataset)

The assignment supplies three features, so none of the following were added or simulated.

- **Richer features:** `Historical_Sales`, `Seasonality`, `Day_of_Week`, `Hour_of_Day`, `Promotion`, `Customer_Segment`, `Market_Trends`, `Price_Elasticity`.
- **Real historical pricing data** with proper time-based train/validation/test splits (no random splits on time-ordered data).
- **Time-series features and demand forecasting** (lags, rolling windows, holiday calendars).
- **Price elasticity estimation** (log-log models, causal methods or controlled price experiments) so the engine optimises revenue/profit, not just predicts price.
- **Competitor monitoring** pipelines with data-quality checks.
- **Business constraints:** minimum/maximum price rules, margin floors, max change per interval, MAP agreements.
- **Fairness and customer impact:** avoid discriminatory or exploitative personalised pricing; audit by segment; comply with local regulation.
- **Monitoring and drift detection:** track input distributions, error by segment, and business KPIs; alert on drift.
- **Retraining policy:** scheduled and drift-triggered, with shadow-mode and A/B validation before rollout.
- **Online inference:** a versioned model behind a low-latency API (e.g. FastAPI) with logging, fallbacks and rollback.
- **Non-linear models** (gradient boosting etc.) become relevant once real, non-linear data exists. They lost here only because this data is exactly linear.

## 10. Project Structure

```
mentroid-ml-task11-the-dynamic-pricing-engine-regression-Khushbu/
├── dynamic_pricing.py
├── requirements.txt
├── README.md
├── run_output.txt
├── .gitignore
└── outputs/
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    ├── residual_distribution.png
    └── model_comparison.png
```

## 11. How to Run

```bash
git clone https://github.com/<your-username>/mentroid-ml-task11-the-dynamic-pricing-engine-regression-Khushbu.git
cd mentroid-ml-task11-the-dynamic-pricing-engine-regression-Khushbu
pip install -r requirements.txt
python dynamic_pricing.py
```

Runtime is under a minute. Plots are written to `outputs/`. Results are deterministic (`random_state=42` everywhere).

## 12. Author

**Name:** Khushbu Mankare  
**Email:** khushbumankare1@gmail.com

## 13. Limitations

- Synthetic, linear, Gaussian-noise data: conclusions about "linear beats trees" and "noise treatment doesn't help" are specific to this generator.
- A single 80/20 split gives a test MAE with wide uncertainty (about ±1.2); CV on the training set is the more reliable comparison.
- Repeated-CV folds overlap, so the paired SEs are somewhat optimistic; the additional 1% minimum-gain rule guards against this.
- Residuals are in-sample for the diagnostic; with 4 parameters and 800 rows the optimism is negligible.
