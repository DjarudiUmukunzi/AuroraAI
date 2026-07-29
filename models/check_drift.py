"""
check_drift.py

MLOps piece of Phase 4: compares the statistical distribution of newly
fetched Kp data against the distribution the deployed model was trained
on. A meaningful shift (space weather entering an unusually active or
quiet stretch, a sensor/feed change, etc.) is exactly the kind of thing
that should trigger a retraining review rather than silently degrading
forecast quality.

Method: a simple two-sample Kolmogorov-Smirnov test on the Kp value
distributions (baseline vs. current). This is a standard, lightweight
drift-detection approach - no need for a heavier framework at this data
scale.

Run:
    python models/check_drift.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from scipy import stats

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kp_index.csv")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "training_baseline.json")

DRIFT_P_VALUE_THRESHOLD = 0.05  # standard significance level for KS test


def load_kp_series() -> pd.Series:
    df = pd.read_csv(RAW_PATH)
    kp_col = next(c for c in df.columns if c.lower() in ("kp", "kp_index"))
    return pd.to_numeric(df[kp_col], errors="coerce").dropna()


def save_baseline():
    """Call this once, right after training, to snapshot 'what normal
    looked like' at training time. train_model.py can call this
    automatically - see the integration note at the bottom of this file."""
    series = load_kp_series()
    baseline = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "values": series.tolist(),
        "mean": float(series.mean()),
        "std": float(series.std()),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f)
    print(f"Baseline saved: mean={baseline['mean']:.2f}, std={baseline['std']:.2f}, n={len(series)}")


def check_drift() -> dict:
    if not os.path.exists(BASELINE_PATH):
        print("No baseline found yet. Run save_baseline() once after a training run first.")
        sys.exit(0)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    current = load_kp_series()
    baseline_values = baseline["values"]

    ks_stat, p_value = stats.ks_2samp(baseline_values, current)
    drifted = p_value < DRIFT_P_VALUE_THRESHOLD

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "baseline_mean": baseline["mean"],
        "baseline_std": baseline["std"],
        "baseline_n": len(baseline_values),
        "current_mean": float(current.mean()),
        "current_std": float(current.std()),
        "current_n": len(current),
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": bool(drifted),
    }

    print(f"Baseline: mean={result['baseline_mean']:.2f}, std={result['baseline_std']:.2f} (n={result['baseline_n']})")
    print(f"Current:  mean={result['current_mean']:.2f}, std={result['current_std']:.2f} (n={result['current_n']})")
    print(f"KS statistic: {ks_stat:.3f}, p-value: {p_value:.4f}")

    if drifted:
        print(f"\n⚠️  DRIFT DETECTED (p < {DRIFT_P_VALUE_THRESHOLD}) — the current Kp distribution")
        print("   differs significantly from training-time data. Consider retraining.")
    else:
        print(f"\n✅ No significant drift (p >= {DRIFT_P_VALUE_THRESHOLD}) — model's training")
        print("   distribution still looks representative of current conditions.")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-baseline", action="store_true", help="Snapshot the current data as the new baseline"
    )
    args = parser.parse_args()

    if args.save_baseline:
        save_baseline()
    else:
        result = check_drift()
        # Non-zero exit lets this be used as a CI gate too, same pattern
        # as the model quality gate in ci.yml
        sys.exit(1 if result["drift_detected"] else 0)