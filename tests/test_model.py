import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from data_generator import generate_synthetic_data
from feature_engineering import (
    compute_financial_ratios, winsorize_ratios, preprocess_features, RATIO_COLS,
)
from model import train_model, get_coefficients


@pytest.fixture(scope='module')
def prepared_data():
    df = generate_synthetic_data(n_obs=400, seed=42)
    df = compute_financial_ratios(df)
    df = winsorize_ratios(df, RATIO_COLS)
    y = df['default_indicator'].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, stratify=y, random_state=42
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test  = df.iloc[test_idx].reset_index(drop=True)
    X_train, scaler, feature_names = preprocess_features(df_train, RATIO_COLS, fit_scaler=True)
    X_test, _, _ = preprocess_features(df_test, RATIO_COLS, scaler=scaler, fit_scaler=False)
    return (
        X_train.values, X_test.values,
        df_train['default_indicator'].values,
        df_test['default_indicator'].values,
        feature_names,
    )


def test_train_returns_required_keys(prepared_data):
    X_train, _, y_train, _, _ = prepared_data
    result = train_model(X_train, y_train)
    for key in ['model', 'best_C', 'cv_auc_mean', 'cv_auc_std', 'cv_auc_scores']:
        assert key in result, f"Missing key: {key}"


def test_cv_auc_above_random(prepared_data):
    X_train, _, y_train, _, _ = prepared_data
    result = train_model(X_train, y_train)
    assert result['cv_auc_mean'] > 0.55, f"CV AUC {result['cv_auc_mean']:.3f} too low"


def test_best_C_is_valid(prepared_data):
    X_train, _, y_train, _, _ = prepared_data
    result = train_model(X_train, y_train)
    assert result['best_C'] in [0.001, 0.01, 0.1, 1, 10]


def test_get_coefficients_shape(prepared_data):
    X_train, _, y_train, _, feature_names = prepared_data
    result = train_model(X_train, y_train)
    coef_df = get_coefficients(result['model'], feature_names, X_train, y_train, n_bootstrap=10)
    assert len(coef_df) == len(feature_names)


def test_get_coefficients_has_required_columns(prepared_data):
    X_train, _, y_train, _, feature_names = prepared_data
    result = train_model(X_train, y_train)
    coef_df = get_coefficients(result['model'], feature_names, X_train, y_train, n_bootstrap=10)
    for col in ['feature', 'coefficient', 'std_error', 'z_stat', 'p_value']:
        assert col in coef_df.columns
