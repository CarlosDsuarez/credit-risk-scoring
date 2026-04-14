# tests/test_validation.py
import numpy as np
import pytest
from validation import (
    compute_discrimination_metrics,
    hosmer_lemeshow_test,
    compute_calibration_slope,
)


@pytest.fixture
def mock_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.08, size=300)
    # Predictions positively correlated with truth -> AUC > 0.5
    y_pred = np.clip(y_true * 0.5 + rng.uniform(0.0, 0.25, 300), 0.01, 0.99)
    return y_true, y_pred


def test_auc_above_random(mock_predictions):
    y_true, y_pred = mock_predictions
    m = compute_discrimination_metrics(y_true, y_pred)
    assert m['auc_roc'] > 0.5


def test_gini_equals_2auc_minus_1(mock_predictions):
    y_true, y_pred = mock_predictions
    m = compute_discrimination_metrics(y_true, y_pred)
    assert abs(m['gini'] - (2 * m['auc_roc'] - 1)) < 1e-10


def test_ks_stat_in_unit_interval(mock_predictions):
    y_true, y_pred = mock_predictions
    m = compute_discrimination_metrics(y_true, y_pred)
    assert 0.0 <= m['ks_stat'] <= 1.0


def test_concentration_ratio_in_unit_interval(mock_predictions):
    y_true, y_pred = mock_predictions
    m = compute_discrimination_metrics(y_true, y_pred)
    assert 0.0 <= m['concentration_ratio'] <= 1.0


def test_brier_score_in_unit_interval(mock_predictions):
    y_true, y_pred = mock_predictions
    m = compute_discrimination_metrics(y_true, y_pred)
    assert 0.0 <= m['brier_score'] <= 1.0


def test_hl_p_value_in_unit_interval(mock_predictions):
    y_true, y_pred = mock_predictions
    result = hosmer_lemeshow_test(y_true, y_pred)
    assert 0.0 <= result['hl_p_value'] <= 1.0


def test_hl_returns_decile_table(mock_predictions):
    y_true, y_pred = mock_predictions
    result = hosmer_lemeshow_test(y_true, y_pred)
    assert 'decile_table' in result
    assert len(result['decile_table']) > 0


def test_calibration_slope_is_float(mock_predictions):
    y_true, y_pred = mock_predictions
    slope = compute_calibration_slope(y_true, y_pred)
    assert isinstance(slope, float)
