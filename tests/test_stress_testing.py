import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from data_generator import generate_synthetic_data
from feature_engineering import (
    compute_financial_ratios, winsorize_ratios, preprocess_features, RATIO_COLS,
)
from model import train_model
from risk_categories import compute_thresholds
from stress_testing import univariate_sensitivity, recession_scenario, industry_cliff_risk


@pytest.fixture(scope='module')
def fitted_bundle():
    df = generate_synthetic_data(n_obs=400, seed=42)
    df_ratios = compute_financial_ratios(df)
    df_ratios = winsorize_ratios(df_ratios, RATIO_COLS)
    y = df_ratios['default_indicator'].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.3, stratify=y, random_state=42
    )
    df_train = df_ratios.iloc[train_idx].reset_index(drop=True)
    df_test  = df_ratios.iloc[test_idx].reset_index(drop=True)
    df_orig_test = df.iloc[test_idx].reset_index(drop=True)
    X_train, scaler, feature_names = preprocess_features(df_train, RATIO_COLS, fit_scaler=True)
    result = train_model(X_train.values, df_train['default_indicator'].values)
    model = result['model']
    pd_train = model.predict_proba(X_train.values)[:, 1]
    t25, t75 = compute_thresholds(pd_train)
    return model, scaler, feature_names, df_train, df_orig_test, t25, t75


def test_univariate_sensitivity_shape(fitted_bundle):
    model, scaler, feature_names, df_train, _, t25, t75 = fitted_bundle
    X_median = np.zeros(len(feature_names))
    sens = univariate_sensitivity(model, feature_names, X_median)
    assert isinstance(sens, pd.DataFrame)
    assert len(sens) == len(feature_names) * 2
    assert 'delta_pd' in sens.columns


def test_univariate_sensitivity_baseline_pd_in_range(fitted_bundle):
    model, scaler, feature_names, df_train, _, t25, t75 = fitted_bundle
    X_median = np.zeros(len(feature_names))
    sens = univariate_sensitivity(model, feature_names, X_median)
    assert (sens['baseline_pd'].between(0, 1)).all()


def test_recession_returns_required_keys(fitted_bundle):
    model, scaler, feature_names, df_train, _, t25, t75 = fitted_bundle
    result = recession_scenario(df_train, model, scaler, feature_names, t25, t75)
    assert 'pct_high_risk_recession' in result
    assert 'mean_pd_recession' in result
    assert 0.0 <= result['pct_high_risk_recession'] <= 1.0


def test_industry_cliff_risk_shape(fitted_bundle):
    model, scaler, feature_names, df_train, df_orig_test, t25, t75 = fitted_bundle
    n_test = len(df_orig_test)
    cats = ['HIGH'] * (n_test // 3) + ['MEDIUM'] * (n_test // 3) + ['LOW'] * (n_test - 2 * (n_test // 3))
    result = industry_cliff_risk(df_orig_test, cats)
    assert isinstance(result, pd.DataFrame)
    assert 'flagged' in result.columns
    assert len(result) == df_orig_test['industry'].nunique()
