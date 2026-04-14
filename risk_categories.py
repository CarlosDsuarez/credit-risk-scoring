import numpy as np
import pandas as pd


def compute_thresholds(pd_train: np.ndarray) -> tuple:
    """Derive P25 and P75 thresholds from training-set predicted PDs.
    # ASSUMPTION: Thresholds computed on TRAINING set PDs only and applied to test set.
    """
    return float(np.percentile(pd_train, 25)), float(np.percentile(pd_train, 75))


def assign_risk_category(pd_score: float, threshold_25: float, threshold_75: float) -> str:
    """
    Map a single PD score to LOW / MEDIUM / HIGH.
    # ASSUMPTION: boundary at t25 -> MEDIUM; boundary at t75 -> HIGH.
    """
    if pd_score < threshold_25:
        return 'LOW'
    if pd_score < threshold_75:
        return 'MEDIUM'
    return 'HIGH'


def assign_risk_categories(
    pd_scores: np.ndarray, threshold_25: float, threshold_75: float
) -> list:
    return [assign_risk_category(p, threshold_25, threshold_75) for p in pd_scores]


def validate_categories(y_true: np.ndarray, categories: list) -> dict:
    """Return default rate, count, and n_defaults per risk bucket."""
    df = pd.DataFrame({'default': y_true, 'category': categories})
    tbl = df.groupby('category')['default'].agg(
        default_rate='mean', count='count', n_defaults='sum'
    )
    return tbl.to_dict(orient='index')


def get_risk_categories_dict(threshold_25: float, threshold_75: float) -> dict:
    return {
        'LOW':    {'min_pd': 0.0,          'max_pd': threshold_25, 'color': 'green'},
        'MEDIUM': {'min_pd': threshold_25,  'max_pd': threshold_75, 'color': 'yellow'},
        'HIGH':   {'min_pd': threshold_75,  'max_pd': 1.0,          'color': 'red'},
    }
