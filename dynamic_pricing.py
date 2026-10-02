"""
Dynamic Pricing Engine - Regression
===================================

Run with:  python dynamic_pricing.py

Workflow (selection NEVER touches the test set):

    generate data -> split -> cross-validate every approach on TRAIN only
    -> select final approach -> fit on full TRAIN -> evaluate ONCE on TEST

Dataset is fixed by the assignment (noise=15, random_state=42) and is never
modified. All noise-treatment experiments (filtering / capping) are applied to
TRAINING data only, and inside each CV fold, so no information leaks from the
validation fold or the test set.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")  # headless backend: saves PNGs without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
NOISE = 15  # fixed by the assignment - do not change
OUTPUT_DIR = "outputs"
FEATURES = ["Competitor_Price", "Inventory_Level", "Demand_Score"]
TARGET = "Sale_Price"

# Decision rule for adopting anything more complex than the reference model:
# it must beat the reference in CV MAE by >= 1% AND by > 2 paired standard errors.
MIN_RELATIVE_GAIN = 0.01
MIN_SE_MULTIPLE = 2.0


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
def generate_dataset():
    """Generate the supplied dataset exactly as specified by the assignment."""
    X, y = make_regression(n_samples=1000, n_features=3, noise=NOISE, random_state=RANDOM_STATE)
    return pd.DataFrame(X, columns=FEATURES), pd.Series(y, name=TARGET)


def get_true_coefficients(X_check, y_check):
    """
    make_regression can also return the TRUE coefficients used to build y.
    Same seed -> identical X and y (asserted below). We use this only to
    explain the data (noise floor, coefficient recovery), never to tune models.
    """
    X, y, coef = make_regression(
        n_samples=1000, n_features=3, noise=NOISE, random_state=RANDOM_STATE, coef=True
    )
    assert np.allclose(X, X_check.to_numpy()) and np.allclose(y, y_check.to_numpy())
    return pd.Series(coef, index=FEATURES)


def split_data(X, y):
    """80/20 random split. The test set is set aside and used once at the end."""
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------
def build_models():
    """Experiments 1-3. Hyperparameters are deliberately modest / near default
    to avoid hidden tuning. Ridge picks its own alpha INSIDE each training fold."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge (alpha via RidgeCV)": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Huber Regression": HuberRegressor(epsilon=1.35, max_iter=1000),
    }


def robust_sigma(residuals):
    """Outlier-resistant std estimate (MAD * 1.4826 equals sigma for Gaussian data)."""
    return 1.4826 * np.median(np.abs(residuals - np.median(residuals)))


def fit_with_noise_treatment(base_model, X_train, y_train, method=None, param=None):
    """
    Fit a model, optionally with a noise treatment applied to TRAINING data only.

    method=None      : plain fit
    method="filter"  : drop training rows with |residual| > param * robust_sigma, refit
    method="winsor"  : cap training residuals at the param / (100-param) percentiles
                       (target becomes prediction + capped residual), refit.
                       No rows are deleted.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    first_pass = clone(base_model).fit(X_train, y_train)
    if method is None:
        return first_pass

    pred = first_pass.predict(X_train)
    resid = y_train - pred

    if method == "filter":
        keep = np.abs(resid - np.median(resid)) <= param * robust_sigma(resid)
        return clone(base_model).fit(X_train[keep], y_train[keep])

    if method == "winsor":
        lo, hi = np.percentile(resid, [param, 100 - param])
        y_capped = pred + np.clip(resid, lo, hi)
        return clone(base_model).fit(X_train, y_capped)

    raise ValueError(f"unknown method: {method}")


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------
def evaluate_model(model, X, y):
    """MAE / RMSE / R2 for an already-fitted model."""
    pred = model.predict(np.asarray(X))
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": math.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def cross_validate_approach(base_model, X_train, y_train, cv, method=None, param=None):
    """
    Repeated K-fold CV on the TRAINING set. Noise treatment is re-done inside
    every fold using only that fold's training part; the validation fold is
    never filtered or capped. Returns one MAE per fold.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    maes = []
    for tr_idx, va_idx in cv.split(X_train):
        model = fit_with_noise_treatment(base_model, X_train[tr_idx], y_train[tr_idx], method, param)
        maes.append(mean_absolute_error(y_train[va_idx], model.predict(X_train[va_idx])))
    return np.array(maes)


def paired_comparison(reference_scores, candidate_scores):
    """Same folds -> compare fold by fold. Positive gain = candidate is better."""
    diff = reference_scores - candidate_scores
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    mean_gain = diff.mean()
    meaningful = (mean_gain / reference_scores.mean() >= MIN_RELATIVE_GAIN) and (
        mean_gain > MIN_SE_MULTIPLE * se
    )
    return mean_gain, se, meaningful


# --------------------------------------------------------------------------
# 4. Residual analysis (Experiment 4)
# --------------------------------------------------------------------------
def analyze_residuals(model, X_train, y_train):
    """Residuals on the TRAINING set only. Flags large ones but does NOT call them bad data."""
    resid = np.asarray(y_train) - model.predict(np.asarray(X_train))
    sigma_rob = robust_sigma(resid)
    r = pd.Series(resid)
    stats = {
        "mean": r.mean(),
        "std": r.std(),
        "robust_sigma(MAD)": sigma_rob,
        "skew": r.skew(),
        "excess_kurtosis": r.kurt(),
    }
    # Share of residuals beyond k sigma vs. what a pure Gaussian would give.
    tails = []
    for k in (2.0, 2.5, 3.0):
        observed = np.mean(np.abs(resid - np.median(resid)) > k * sigma_rob)
        expected = math.erfc(k / math.sqrt(2))
        tails.append((k, observed, expected))
    flagged = np.argsort(-np.abs(resid))[:10]
    return resid, stats, tails, flagged


# --------------------------------------------------------------------------
# 5. Plots
# --------------------------------------------------------------------------
def plot_actual_vs_predicted(y_true, y_pred, mae, title_model, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3, label="Test observations")
    lims = [min(y_true.min(), y_pred.min()) - 10, max(y_true.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect prediction (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual Sale_Price")
    ax.set_ylabel("Predicted Sale_Price")
    ax.set_title(f"Actual vs Predicted Price (Test Set)\n{title_model} | MAE = {mae:.2f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(y_pred, resid, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, resid, alpha=0.6, edgecolor="k", linewidth=0.3, label="Test residuals")
    ax.axhline(0, color="r", linestyle="--", linewidth=2, label="Zero error")
    ax.set_xlabel("Predicted Sale_Price")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot (Test Set)\nNo visible trend or funnel shape = no obvious model misspecification")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_distribution(resid, sigma_rob, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(resid, bins=40, density=True, alpha=0.7, edgecolor="k", label="Training residuals")
    grid = np.linspace(resid.min(), resid.max(), 300)
    pdf = np.exp(-0.5 * (grid / sigma_rob) ** 2) / (sigma_rob * math.sqrt(2 * math.pi))
    ax.plot(grid, pdf, "r-", linewidth=2, label=f"Normal(0, {sigma_rob:.1f}) reference")
    ax.axvline(0, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution (Training Set)\nBell-shaped residuals are what Gaussian synthetic noise looks like")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(summary, noise_floor, path):
    s = summary.sort_values("CV MAE (mean)", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 2))
    ax.barh(s.index, s["CV MAE (mean)"], xerr=s["CV MAE (std)"], capsize=3, alpha=0.8, edgecolor="k")
    ax.axvline(noise_floor, color="r", linestyle="--", label=f"Theoretical noise floor ~ {noise_floor:.2f}")
    lo = max(0, s["CV MAE (mean)"].min() - 3)
    ax.set_xlim(lo, s["CV MAE (mean)"].max() + 2)
    ax.set_xlabel("CV MAE (lower is better; error bars = std across folds; x-axis truncated)")
    ax.set_title("Model / Noise-Treatment Comparison\n(Repeated 5-fold CV, training set only)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. Experiments
# --------------------------------------------------------------------------
def run_noise_experiment(models, reference_name, X_train, y_train, cv):
    """Experiments 5 and 6, applied to the best baseline model. Returns CV scores."""
    base = models[reference_name]
    scores = {}
    for k in (2.0, 2.5, 3.0):  # Experiment 5: residual-based filtering
        scores[f"{reference_name} + filter |r|>{k}sigma"] = (
            base, "filter", k, cross_validate_approach(base, X_train, y_train, cv, "filter", k))
    for p in (1, 2, 5):  # Experiment 6: winsorise residuals, keep every row
        scores[f"{reference_name} + winsor {p}/{100 - p}"] = (
            base, "winsor", p, cross_validate_approach(base, X_train, y_train, cv, "winsor", p))
    return scores


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

    # ---------------- Data ----------------
    X, y = generate_dataset()
    true_coef = get_true_coefficients(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y)

    section("DATA")
    print(f"Samples: {len(X)} | features: {X.shape[1]} | noise={NOISE} | random_state={RANDOM_STATE}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)} (test set untouched until final step)")
    print("\nWhat noise=15 means: y = X @ true_coef + Normal(0, sd=15) added to every row.")
    noise_floor_mae = NOISE * math.sqrt(2 / math.pi)
    print(f"With Gaussian noise sd=15, the best ANY model can do is:")
    print(f"  expected MAE  ~ 15*sqrt(2/pi) = {noise_floor_mae:.2f}")
    print(f"  expected RMSE ~ 15.00")
    print("\nTrue generating coefficients (known only because make_regression can return them):")
    print(true_coef.to_string())
    print(f"\nTarget range: [{y.min():.1f}, {y.max():.1f}] -> synthetic, includes negative 'prices'.")

    # ---------------- Cross-validation of all approaches (Experiments 1,2,3,7) ----------------
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)  # same folds every call
    models = build_models()
    cv_results = {}  # name -> (base_model, method, param, fold_scores)

    section("EXPERIMENTS 1-3 + 7: baseline, tree and robust models (5-fold x 3 repeats, TRAIN only)")
    for name, model in models.items():
        cv_results[name] = (model, None, None, cross_validate_approach(model, X_train, y_train, cv))
        print(f"{name:<28} CV MAE = {cv_results[name][3].mean():.3f} +/- {cv_results[name][3].std():.3f}")

    # Experiment 4 needs the best *baseline* (Experiment 1 models) by CV
    baseline_candidates = ["Linear Regression", "Ridge (alpha via RidgeCV)"]
    reference_name = min(baseline_candidates, key=lambda n: cv_results[n][3].mean())
    print(f"\nBest baseline by CV: {reference_name}")

    # ---------------- Experiment 4: residual analysis ----------------
    section("EXPERIMENT 4: residual analysis (TRAINING set only)")
    ref_fit = clone(models[reference_name]).fit(np.asarray(X_train), np.asarray(y_train))
    resid, stats, tails, flagged = analyze_residuals(ref_fit, X_train, y_train)
    for k, v in stats.items():
        print(f"  {k:<20}: {v:8.3f}")
    print("\n  Share of residuals beyond k robust-sigma (observed vs Gaussian expectation):")
    for k, obs, exp in tails:
        print(f"    >{k} sigma: observed {obs*100:5.2f}% | Gaussian would give {exp*100:5.2f}%")
    print("\n  Ten largest |residuals| in training data (NOT automatically 'bad data'):")
    top = X_train.iloc[flagged].copy()
    top["Actual"] = np.asarray(y_train)[flagged]
    top["Residual"] = resid[flagged]
    print(top.to_string())
    print("\n  Interpretation: if residuals look Gaussian, large ones are just the tails of the\n"
          "  injected N(0, 15) noise. They are expected, not corrupted.")
    plot_residual_distribution(resid, robust_sigma(resid), f"{OUTPUT_DIR}/residual_distribution.png")

    # ---------------- Experiments 5 and 6 ----------------
    section("EXPERIMENTS 5-6: residual filtering and winsorisation (inside CV, TRAIN only)")
    noise_scores = run_noise_experiment(models, reference_name, X_train, y_train, cv)
    cv_results.update(noise_scores)
    for name, (_, _, _, sc) in noise_scores.items():
        print(f"{name:<52} CV MAE = {sc.mean():.3f} +/- {sc.std():.3f}")

    # ---------------- Summary table with paired comparison ----------------
    section("EXPERIMENT 7: CV summary and paired comparison vs reference")
    ref_scores = cv_results[reference_name][3]
    rows = []
    for name, (_, _, _, sc) in cv_results.items():
        gain, se, ok = paired_comparison(ref_scores, sc)
        rows.append({
            "Approach": name,
            "CV MAE (mean)": sc.mean(),
            "CV MAE (std)": sc.std(),
            "Gain vs reference": gain,
            "Gain (%)": 100 * gain / ref_scores.mean(),
            "Paired SE": se,
            "Meaningful?": "yes" if (ok and name != reference_name) else ("reference" if name == reference_name else "no"),
        })
    summary = pd.DataFrame(rows).set_index("Approach").sort_values("CV MAE (mean)")
    print(summary.to_string())
    print(f"\nReference = {reference_name}. 'Meaningful' needs >= {MIN_RELATIVE_GAIN*100:.0f}% lower MAE "
          f"AND gain > {MIN_SE_MULTIPLE:.0f} paired SE.")
    print(f"Theoretical noise floor MAE ~ {noise_floor_mae:.2f} (no model can beat this on average).")
    plot_model_comparison(summary, noise_floor_mae, f"{OUTPUT_DIR}/model_comparison.png")

    # ---------------- Final selection (CV only) ----------------
    section("FINAL SELECTION (made BEFORE looking at the test set)")
    best_name = summary.index[0]
    if best_name != reference_name and summary.loc[best_name, "Meaningful?"] != "yes":
        print(f"Lowest CV MAE: '{best_name}', but its gain over '{reference_name}' is not meaningful.")
        print(f"-> Keeping the simpler reference: {reference_name}")
        final_name = reference_name
    else:
        final_name = best_name
        print(f"-> Selected: {final_name}")
    base_model, method, param, _ = cv_results[final_name]

    # ---------------- Final training + single test evaluation ----------------
    final_model = fit_with_noise_treatment(base_model, X_train, y_train, method, param)
    test_pred = final_model.predict(X_test.to_numpy())
    test_metrics = evaluate_model(final_model, X_test, y_test)
    train_metrics = evaluate_model(final_model, X_train, y_train)

    # Bootstrap CI over test errors (does not modify the test set; shows its sampling noise)
    rng = np.random.default_rng(RANDOM_STATE)
    abs_err = np.abs(y_test.to_numpy() - test_pred)
    boot = [rng.choice(abs_err, len(abs_err), replace=True).mean() for _ in range(2000)]
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

    section("FINAL TEST-SET EVALUATION (done once)")
    print(f"Final approach : {final_name}")
    print(f"Test MAE       : {test_metrics['MAE']:.3f}   (95% bootstrap CI: {ci_lo:.2f} - {ci_hi:.2f})")
    print(f"Test RMSE      : {test_metrics['RMSE']:.3f}")
    print(f"Test R2        : {test_metrics['R2']:.4f}")
    print(f"Train MAE      : {train_metrics['MAE']:.3f}  (gap to test is small -> no overfitting)")
    print(f"Noise floor MAE: ~{noise_floor_mae:.2f}")

    plot_actual_vs_predicted(y_test.to_numpy(), test_pred, test_metrics["MAE"], final_name,
                             f"{OUTPUT_DIR}/actual_vs_predicted.png")
    plot_residuals(test_pred, y_test.to_numpy() - test_pred, f"{OUTPUT_DIR}/residual_plot.png")

    # ---------------- Interpretability ----------------
    section("INTERPRETABILITY")
    est = final_model[-1] if hasattr(final_model, "steps") else final_model
    if hasattr(est, "coef_"):
        scale = final_model[0].scale_ if hasattr(final_model, "steps") else np.ones(3)
        coef_raw = est.coef_ / scale  # back to original feature units
        table = pd.DataFrame({"Estimated coef": coef_raw, "True coef": true_coef.values}, index=FEATURES)
        print(table.to_string())
        print(f"Intercept (estimated): {final_model.predict(np.zeros((1, 3)))[0]:.3f}")
        print("\nEach coefficient = expected change in Sale_Price for +1 unit of that feature, other features fixed.")
    if hasattr(est, "feature_importances_"):
        print(pd.Series(est.feature_importances_, index=FEATURES).to_string())

    # ---------------- Transparency: test MAE of everything, AFTER selection is locked ----------------
    section("TRANSPARENCY TABLE: test MAE of every approach (reported AFTER selection; NOT used to select)")
    out = []
    for name, (bm, m, p, sc) in cv_results.items():
        fitted = fit_with_noise_treatment(bm, X_train, y_train, m, p)
        out.append({"Approach": name, "CV MAE": sc.mean(),
                    "Test MAE (informational)": mean_absolute_error(y_test, fitted.predict(X_test.to_numpy()))})
    print(pd.DataFrame(out).set_index("Approach").sort_values("CV MAE").to_string())
    print(f"\nReminder: test MAE has sampling noise of roughly +/-{(ci_hi - ci_lo)/2:.2f} (95% CI half-width), "
          "so differences of a few tenths between approaches are not distinguishable on 200 test rows.")

    print(f"\nPlots saved in ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
