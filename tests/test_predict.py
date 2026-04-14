# tests/test_predict.py
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from data_generator import generate_synthetic_data
from feature_engineering import (
    compute_financial_ratios, winsorize_ratios, preprocess_features, RATIO_COLS,
)
from model import train_model, get_coefficients
from risk_categories import compute_thresholds
from predict import predict_default_probability


@pytest.fixture(scope='module')
def artifacts_and_sample():
    df = generate_synthetic_data(n_obs=500, seed=42)
    df_ratios = compute_financial_ratios(df)
    df_ratios = winsorize_ratios(df_ratios, RATIO_COLS)
    y = df_ratios['default_indicator'].values
    train_idx, _ = train_test_split(
        np.arange(len(df)), test_size=0.3, stratify=y, random_state=42
    )
    df_train = df_ratios.iloc[train_idx].reset_index(drop=True)
    X_train, scaler, feature_names = preprocess_features(df_train, RATIO_COLS, fit_scaler=True)
    result = train_model(X_train.values, df_train['default_indicator'].values)
    model = result['model']
    coef_df = get_coefficients(model, feature_names, X_train.values,
                               df_train['default_indicator'].values, n_bootstrap=10)
    pd_train = model.predict_proba(X_train.values)[:, 1]
    t25, t75 = compute_thresholds(pd_train)
    winsorize_bounds = {
        col: (df_train[col].quantile(0.01), df_train[col].quantile(0.99))
        for col in RATIO_COLS if col in df_train.columns
    }
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'threshold_25': t25,
        'threshold_75': t75,
        'coef_df': coef_df,
        'winsorize_bounds': winsorize_bounds,
    }
    sample = df.iloc[0].to_dict()
    return artifacts, sample


def test_predict_returns_required_keys(artifacts_and_sample):
    artifacts, sample = artifacts_and_sample
    result = predict_default_probability(sample, artifacts)
    for key in ['borrower_id', 'predicted_pd', 'risk_category',
                'confidence_interval_95', 'top_3_risk_drivers', 'recommendation']:
        assert key in result, f"Missing key: {key}"


def test_predicted_pd_in_unit_interval(artifacts_and_sample):
    artifacts, sample = artifacts_and_sample
    result = predict_default_probability(sample, artifacts)
    assert 0.0 <= result['predicted_pd'] <= 1.0


def test_risk_category_valid(artifacts_and_sample):
    artifacts, sample = artifacts_and_sample
    result = predict_default_probability(sample, artifacts)
    assert result['risk_category'] in {'LOW', 'MEDIUM', 'HIGH'}


def test_top_3_drivers_count(artifacts_and_sample):
    artifacts, sample = artifacts_and_sample
    result = predict_default_probability(sample, artifacts)
    assert len(result['top_3_risk_drivers']) == 3


def test_confidence_interval_ordered(artifacts_and_sample):
    artifacts, sample = artifacts_and_sample
    result = predict_default_probability(sample, artifacts)
    lo, hi = result['confidence_interval_95']
    assert lo <= hi


def test_recommendation_is_string(artifacts_and_sample):
    artifacts, sample = artifacts_and_sample
    result = predict_default_probability(sample, artifacts)
    assert isinstance(result['recommendation'], str)
    assert len(result['recommendation']) > 0
