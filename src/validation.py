"""Model validation: discrimination metrics and risk-bucket calibration."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def ks_statistic(y_true, pd_hat) -> float:
    """Kolmogorov-Smirnov statistic between predicted PD distributions.

    Measures the maximum vertical distance between the CDF of predicted
    PDs for performing loans and for defaulted loans. Larger is better.
    """
    y_true = np.asarray(y_true)
    return stats.ks_2samp(pd_hat[y_true == 0], pd_hat[y_true == 1]).statistic


def validation_metrics(y_true, pd_hat) -> dict:
    """Compute AUC, Gini, KS and Brier score for a set of predicted PDs.

    - AUC: probability that a random defaulter is ranked above a random
      performer.
    - Gini: 2 * AUC - 1, a rescaled version of AUC.
    - KS: see `ks_statistic`.
    - Brier: mean squared error between predicted PD and the realised
      0/1 outcome; lower is better.
    """
    auc = roc_auc_score(y_true, pd_hat)
    return {
        "AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks_statistic(y_true, pd_hat),
        "Brier": brier_score_loss(y_true, pd_hat),
    }


def assign_buckets(y_true, pd_hat, n_buckets: int = 5) -> pd.DataFrame:
    """Rank loans by predicted PD and cut them into equal-sized buckets.

    Bucket 1 contains the lowest predicted PDs (safest loans), bucket
    `n_buckets` the highest. Ranking with `method="first"` ensures ties
    are broken deterministically so bucket sizes stay equal.
    """
    loans = pd.DataFrame({"pd": np.asarray(pd_hat), "default": np.asarray(y_true)})
    loans["bucket"] = pd.qcut(loans["pd"].rank(method="first"), n_buckets, labels=False) + 1
    return loans


def risk_bucket_table(y_true, pd_hat, n_buckets: int = 5) -> pd.DataFrame:
    """Compare average predicted PD with the observed default rate per bucket.

    A well-calibrated model has `avg_predicted_pd` close to
    `observed_default_rate` in every bucket.
    """
    loans = assign_buckets(y_true, pd_hat, n_buckets)
    return loans.groupby("bucket").agg(
        loans=("default", "size"),
        avg_predicted_pd=("pd", "mean"),
        observed_default_rate=("default", "mean"),
    ).round(3)
