# tests/test_feature_engineering.py
import pytest
import numpy as np
import pandas as pd
from data_generator import generate_synthetic_data
from feature_engineering import (
    compute_financial_ratios, winsorize_ratios,
    preprocess_features, compute_vif, RATIO_COLS,
)


@pytest.fixture(scope='module')
def raw_df():
    return generate_synthetic_data(n_obs=200, seed=42)


def test_ratio_cols_constant_has_18_entries():
    assert len(RATIO_COLS) == 18


def test_compute_ratios_adds_all_ratio_cols(raw_df):
    df = compute_financial_ratios(raw_df)
    for col in RATIO_COLS:
        assert col in df.columns, f"Missing ratio column: {col}"


def test_no_inf_after_ratios(raw_df):
    df = compute_financial_ratios(raw_df)
    for col in RATIO_COLS:
        assert not np.isinf(df[col]).any(), f"Inf in {col}"


def test_winsorize_clips_to_p1_p99(raw_df):
    df = compute_financial_ratios(raw_df)
    df_w = winsorize_ratios(df, RATIO_COLS)
    for col in RATIO_COLS:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        assert df_w[col].min() >= p1 - 1e-9
        assert df_w[col].max() <= p99 + 1e-9


def test_preprocess_fit_returns_correct_shape(raw_df):
    df = compute_financial_ratios(raw_df)
    df = winsorize_ratios(df, RATIO_COLS)
    X, scaler, feature_names = preprocess_features(df, RATIO_COLS, fit_scaler=True)
    assert X.shape[0] == len(raw_df)
    assert X.shape[1] == len(feature_names)


def test_scaler_fit_only_on_train_no_leakage(raw_df):
    df = compute_financial_ratios(raw_df)
    df = winsorize_ratios(df, RATIO_COLS)
    train = df.iloc[:140].reset_index(drop=True)
    test  = df.iloc[140:].reset_index(drop=True)
    X_train, scaler, feature_names = preprocess_features(train, RATIO_COLS, fit_scaler=True)
    X_test, _, _ = preprocess_features(test, RATIO_COLS, scaler=scaler, fit_scaler=False)
    assert X_test.shape[0] == 60


def test_compute_vif_returns_dataframe(raw_df):
    df = compute_financial_ratios(raw_df)
    df = winsorize_ratios(df, RATIO_COLS)
    X, _, feature_names = preprocess_features(df, RATIO_COLS, fit_scaler=True)
    vif_df = compute_vif(X)
    assert 'feature' in vif_df.columns
    assert 'VIF' in vif_df.columns
    assert len(vif_df) == len(feature_names)
