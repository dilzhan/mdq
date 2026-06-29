from itertools import product

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from .config import RANDOM_STATE, BIZ, MC_LIGHT, MC_DARK, EDGES, PLT_PARAMS, mc_title
from matplotlib.colors import LinearSegmentedColormap as _LSC

plt.rcParams.update(PLT_PARAMS)
_CM_CMAP = _LSC.from_list("mc_cm", [MC_LIGHT, BIZ])


def get_models(random_state=RANDOM_STATE):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "knn": KNeighborsClassifier(),
        "svm": CalibratedClassifierCV(
            estimator=LinearSVC(max_iter=1000, random_state=random_state),
            cv=5,
        ),
        "random_forest": RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    return models


def get_model_grids(random_state=RANDOM_STATE):
    grids = {
        "logistic_regression": {
            "C": 10.0 ** np.arange(-2, 3),
            "l1_ratio": [0, 0.25, 0.5, 0.75, 1],
            "max_iter": [1000],
            "random_state": [random_state],
            "solver": ["saga"],
        },
        "knn": {
            "n_neighbors": [15, 25, 35, 50, 75, 100],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "svm": {
            "estimator__C": [0.01, 0.1, 1, 5, 10, 20, 50, 100],
            "estimator__max_iter": [1000],
            "estimator__random_state": [random_state],
            "cv": [5],
        },
        "random_forest": {
            "n_estimators": [10, 20, 30],
            "max_depth": [2, 3, 4, 5],
            "max_features": ["sqrt", "log2", 0.5],
            "random_state": [random_state],
        },
        "xgboost": {
            "n_estimators": [35, 50],
            "max_depth": [3, 4],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7],
            "colsample_bytree": [0.4, 0.5],
            "min_child_weight": [20],
            "gamma": [1.8],
            "reg_alpha": [2.5],
            "reg_lambda": [12.0],
            "random_state": [random_state],
        },
    }

    return grids


def find_threshold_for_min_tpr(
    y_true,
    y_score,
    min_tpr=0.995,
    step=0.01,
):
    """
    Find the highest threshold that keeps TPR above min_tpr.

    This follows the notebook logic: raise the threshold until the business-card
    recall drops below the required level, then step back slightly.
    """

    for threshold in np.arange(0, 1 + step, step):
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, normalize="true")

        if cm[1, 1] < min_tpr:
            return max(0.0, threshold - step)

    return 1


def grid_search_with_constraints(
    base_model,
    grid,
    X_train,
    y_train,
    X_val,
    y_val,
    min_tpr=0.995,
    min_fpr=0.015,
    max_fpr=0.05,
):
    """
    Search model parameters using the business constraints from the notebook.

    Returns only configurations whose validation FPR is inside the expected
    range while maintaining the required TPR.
    """
    keys = list(grid.keys())
    configs = [dict(zip(keys, values)) for values in product(*grid.values())]
    results = []

    for params in configs:
        model = clone(base_model)
        model.set_params(**params)
        model.fit(X_train, y_train)

        y_score = model.predict_proba(X_val)[:, 1]
        threshold = find_threshold_for_min_tpr(y_val, y_score, min_tpr=min_tpr)
        y_pred = (y_score >= threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred, normalize="true")

        tpr = cm[1, 1]
        fpr = cm[0, 1]

        if min_fpr <= fpr <= max_fpr:
            results.append(
                {
                    **params,
                    "model": model,
                    "threshold": threshold,
                    "tpr": tpr,
                    "fpr": fpr,
                }
            )

    return results


def evaluate_at_threshold(model, X, y, threshold):
    """Evaluate a fitted model at a fixed probability threshold."""
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    return confusion_matrix(y, y_pred, normalize="true")


def plot_gridsearch_results(results, X_test, y_test, n_cols=4):
    n_results = len(results)
    n_rows = (n_results + n_cols - 1) // n_cols

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 3 * n_rows))
    axes = axes.flatten()

    for i in range(n_results):
        model = results[i]["model"]
        threshold = results[i]["threshold"]

        cm = evaluate_at_threshold(model, X_test, y_test, threshold)

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap=_CM_CMAP,
            linewidths=0.8,
            linecolor=EDGES,
            annot_kws={"size": 11, "fontweight": "600", "color": MC_DARK},
            ax=axes[i],
        )
        mc_title(axes[i], f"Model #{i} | thr={threshold:.3f}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(n_results, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def collect_model_probabilities(models, X):
    """Collect class-1 probabilities from a list of fitted models."""

    probs = np.zeros((X.shape[0], len(models)))

    for i, model in enumerate(models):
        probs[:, i] = model.predict_proba(X)[:, 1]

    return probs


def uncertainty_score(probs, threshold, bandwidth=0.02):
    """
    Combine model disagreement and threshold ambiguity into one uncertainty score.
    """
    probs = np.asarray(probs)
    mean_probability = probs.mean(axis=1)

    model_uncertainty = probs.var(axis=1) / 0.24
    model_uncertainty = np.clip(model_uncertainty, 0, 1)

    threshold_uncertainty = np.exp(
        -((mean_probability - threshold) ** 2) / (2 * bandwidth**2)
    )

    uncertainty = 0.8 * model_uncertainty + 0.2 * threshold_uncertainty

    return mean_probability, uncertainty
