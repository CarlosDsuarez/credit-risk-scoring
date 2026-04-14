import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Fit logistic regression with L2 regularization, tuning C via 5-fold CV.

    Returns dict with: model, best_C, cv_auc_scores, cv_auc_mean, cv_auc_std.
    # ASSUMPTION: class_weight='balanced' to handle 8% default imbalance.
    # ASSUMPTION: C grid {0.001, 0.01, 0.1, 1, 10}; lbfgs solver.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=42,
        ),
        param_grid={'C': [0.001, 0.01, 0.1, 1, 10]},
        cv=cv, scoring='roc_auc', n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_['C']

    model = LogisticRegression(
        penalty='l2', C=best_C, solver='lbfgs', max_iter=1000,
        class_weight='balanced', random_state=42,
    )
    model.fit(X_train, y_train)

    cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring='roc_auc')['test_score']

    return {
        'model': model,
        'best_C': best_C,
        'cv_auc_scores': cv_scores,
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std()),
    }


def get_coefficients(
    model: LogisticRegression,
    feature_names: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    """
    Compute logistic regression coefficients with bootstrap standard errors,
    z-statistics, and approximate p-values.

    # ASSUMPTION: SE estimated via n_bootstrap resamples of training data.
    """
    rng = np.random.default_rng(42)
    n = len(y_train)
    boot_coefs = np.zeros((n_bootstrap, len(feature_names)))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b, y_b = X_train[idx], y_train[idx]
        if len(np.unique(y_b)) < 2:
            boot_coefs[i] = model.coef_[0]
            continue
        m = LogisticRegression(
            penalty='l2', C=model.C, solver='lbfgs', max_iter=500,
            class_weight='balanced', random_state=i,
        )
        m.fit(X_b, y_b)
        boot_coefs[i] = m.coef_[0]

    coefs    = model.coef_[0]
    std_errs = boot_coefs.std(axis=0)
    z_stats  = np.where(std_errs > 0, coefs / std_errs, 0.0)
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stats)))

    return pd.DataFrame({
        'feature':     feature_names,
        'coefficient': coefs,
        'std_error':   std_errs,
        'z_stat':      z_stats,
        'p_value':     p_values,
    }).sort_values('coefficient', key=abs, ascending=False).reset_index(drop=True)
