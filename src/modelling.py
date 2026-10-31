"""Model training: logistic regression and decision tree with GridSearchCV."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import (
    RANDOM_STATE,
    make_preprocessor,
    get_column_types,
)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def train_logistic_regression(X_train, y_train):
    """Fit a logistic regression pipeline with 5-fold CV over the regularisation strength.

    Numeric features are standardised inside the pipeline, so scaling is
    refitted on each CV fold and no test information leaks into training.
    Returns the fitted GridSearchCV object; access `.best_estimator_` and
    `.best_params_` for the chosen model and hyperparameters.
    """
    numeric_cols, categorical_cols = get_column_types(X_train)
    pipeline = Pipeline([
        ("prep", make_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={"model__C": [0.01, 0.1, 1, 10]},
        scoring="roc_auc",
        cv=CV,
    )
    search.fit(X_train, y_train)
    return search


def train_decision_tree(X_train, y_train):
    """Fit a decision tree pipeline with 5-fold CV over depth and leaf size.

    No scaling is applied because trees are invariant to monotonic
    transformations of the features. Trees are regularised through
    `max_depth` and `min_samples_leaf` to limit overfitting on a small
    training set.
    """
    numeric_cols, categorical_cols = get_column_types(X_train)
    pipeline = Pipeline([
        ("prep", make_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])
    search = GridSearchCV(
        pipeline,
        param_grid={
            "model__max_depth": [3, 4, 5, 6],
            "model__min_samples_leaf": [10, 20, 30, 50],
        },
        scoring="roc_auc",
        cv=CV,
    )
    search.fit(X_train, y_train)
    return search


def cv_summary(search):
    """Tabulate CV results, best parameter combination first."""
    import pandas as pd
    table = pd.DataFrame(search.cv_results_)
    param_cols = [c for c in table.columns if c.startswith("param_")]
    cols = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    return table[cols].sort_values("rank_test_score").round(3)
