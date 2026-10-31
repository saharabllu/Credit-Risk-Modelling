"""Stress testing scenarios and Expected Loss (EL = PD x LGD x EAD).

The scenarios below are illustrative assumptions, not estimates from
macroeconomic data. The German Credit dataset contains no macro variables,
so the link between economic conditions and PD cannot be estimated from
the data. Each scenario is defined by an explicit, documented assumption:

- Baseline:         observed PDs, LGD 45%, no EAD multiplier.
- Adverse:          PD odds shifted by +0.4 on the log-odds scale,
                    LGD 50%, EAD x1.05.
- Severe:           PD odds shifted by +0.8, LGD 60%, EAD x1.10.
- Repayment burden: PD re-estimated after multiplying `duration` by 1.30
                    and `credit_amount` by 1.40; LGD 45%, no EAD multiplier.
"""

import numpy as np

BASE_LGD = 0.45

SCENARIOS = [
    {
        "name": "Baseline",
        "pd_logit_shift": 0.0,
        "lgd": BASE_LGD,
        "ead_multiplier": 1.00,
        "feature_shocks": {},
    },
    {
        "name": "Adverse",
        "pd_logit_shift": 0.4,
        "lgd": 0.50,
        "ead_multiplier": 1.05,
        "feature_shocks": {},
    },
    {
        "name": "Severe",
        "pd_logit_shift": 0.8,
        "lgd": 0.60,
        "ead_multiplier": 1.10,
        "feature_shocks": {},
    },
    {
        "name": "Repayment burden",
        "pd_logit_shift": 0.0,
        "lgd": BASE_LGD,
        "ead_multiplier": 1.00,
        "feature_shocks": {"duration": 1.30, "credit_amount": 1.40},
    },
]


def shift_pd(pd_hat, logit_shift: float):
    """Shift PD by a constant on the log-odds scale.

    Working on log-odds keeps PD strictly between 0 and 1 and multiplies
    the odds of default by a constant factor for every loan (about 1.5
    for +0.4 and 2.2 for +0.8). Multiplying PD directly would push high
    PDs above 100%.
    """
    pd_hat = np.clip(pd_hat, 1e-6, 1 - 1e-6)
    log_odds = np.log(pd_hat / (1 - pd_hat))
    return 1 / (1 + np.exp(-(log_odds + logit_shift)))


def apply_scenario(model, X, scenario: dict):
    """Loan-level PD, EAD and Expected Loss under one scenario.

    A copy of `X` is used so the caller's data is never modified.
    Feature shocks (if any) are applied first, the model then re-scores
    the shocked data, and PD is finally shifted on the log-odds scale.
    EAD is approximated by `credit_amount` (with the scenario multiplier);
    LGD is taken from the scenario.
    """
    shocked = X.copy()
    for column, multiplier in scenario["feature_shocks"].items():
        shocked[column] = shocked[column] * multiplier

    pd_hat = shift_pd(model.predict_proba(shocked)[:, 1], scenario["pd_logit_shift"])
    ead = shocked["credit_amount"].to_numpy() * scenario["ead_multiplier"]
    expected_loss = pd_hat * scenario["lgd"] * ead
    return pd_hat, ead, expected_loss


def run_scenario(model, X, scenario: dict) -> dict:
    """Portfolio-level summary for a single scenario."""
    pd_hat, ead, expected_loss = apply_scenario(model, X, scenario)

    return {
        "Scenario": scenario["name"],
        "Avg PD": pd_hat.mean(),
        "LGD": scenario["lgd"],
        "Total EAD (DM)": ead.sum(),
        "Expected Loss (DM)": expected_loss.sum(),
        "EL / EAD": expected_loss.sum() / ead.sum(),
    }
