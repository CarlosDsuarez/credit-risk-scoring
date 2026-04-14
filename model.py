import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Fit logistic regression with L2 regularization, tuning C via 5-fold CV.

    Returns dict with: model, best_C, cv_auc_scores, cv_auc_mean, cv_auc_std.
    # ASSUMPTION: class_weight='balanced' to handle 8% default imbalance.
    # ASSUMPTION: C grid {0.001, 0.01, 0.1, 1, 10}; lbfgs solver.
    # NOTE: cv_auc_scores are the held-out fold scores from GridSearchCV for best_C,
    #       not re-scored on the full training set.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        LogisticRegression(
            penalty='l2', solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=42,
        ),
        param_grid={'C': [0.001, 0.01, 0.1, 1, 10]},
        cv=cv, scoring='roc_auc', n_jobs=-1,
        return_train_score=False,
    )
    grid_search.fit(X_train, y_train)
    best_C = grid_search.best_params_['C']

    # Use best_estimator_ (already refit on full training set by GridSearchCV)
    model = grid_search.best_estimator_

    # Extract the per-fold AUC scores for best_C from GridSearchCV results
    best_idx = grid_search.best_index_
    cv_scores = np.array([
        grid_search.cv_results_[f'split{i}_test_score'][best_idx]
        for i in range(5)
    ])

    return {
        'model': model,
        'best_C': best_C,
        'cv_auc_scores': cv_scores,
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std': float(cv_scores.std(ddof=1)),
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
    # ASSUMPTION: Bootstrap resamples that produce a single-class target are skipped
    #             (nan row) to avoid distorting SE estimates.
    """
    if len(feature_names) != X_train.shape[1]:
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but X_train has "
            f"{X_train.shape[1]} columns."
        )

    rng = np.random.default_rng(42)
    n = len(y_train)
    boot_coefs = np.full((n_bootstrap, len(feature_names)), np.nan)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b, y_b = X_train[idx], y_train[idx]
        if len(np.unique(y_b)) < 2:
            # Skip single-class resamples — nan row excluded from std computation
            continue
        m = LogisticRegression(
            penalty='l2', C=model.C, solver='lbfgs', max_iter=1000,
            class_weight='balanced', random_state=i,
        )
        m.fit(X_b, y_b)
        boot_coefs[i] = m.coef_[0]

    coefs = model.coef_[0]
    # nanstd with ddof=1 ignores skipped rows and uses sample SE formula
    std_errs = np.nanstd(boot_coefs, axis=0, ddof=1)
    z_stats  = np.where(std_errs > 0, coefs / std_errs, 0.0)
    p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stats)))

    return pd.DataFrame({
        'feature':     feature_names,
        'coefficient': coefs,
        'std_error':   std_errs,
        'z_stat':      z_stats,
        'p_value':     p_values,
    }).sort_values('coefficient', key=abs, ascending=False).reset_index(drop=True)
