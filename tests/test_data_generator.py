import pytest
import pandas as pd
from data_generator import generate_synthetic_data

REQUIRED_COLS = [
    'borrower_id', 'default_indicator', 'annual_revenue', 'ebitda',
    'total_debt', 'cash_on_hand', 'current_assets', 'current_liabilities',
    'inventory', 'accounts_receivable', 'accounts_payable',
    'interest_expense', 'tax_rate', 'industry', 'years_operating',
    'credit_score', 'prior_defaults', 'loan_amount', 'loan_tenor',
]

def test_shape():
    df = generate_synthetic_data(n_obs=800)
    assert len(df) == 800

def test_required_columns():
    df = generate_synthetic_data(n_obs=100)
    for col in REQUIRED_COLS:
        assert col in df.columns, f"Missing column: {col}"

def test_default_rate_in_range():
    df = generate_synthetic_data(n_obs=800)
    dr = df['default_indicator'].mean()
    assert 0.05 <= dr <= 0.12, f"Default rate {dr:.3f} outside [5%, 12%]"

def test_reproducibility():
    df1 = generate_synthetic_data(seed=42)
    df2 = generate_synthetic_data(seed=42)
    pd.testing.assert_frame_equal(df1, df2)

def test_positive_revenue():
    df = generate_synthetic_data()
    assert (df['annual_revenue'] > 0).all()

def test_no_nulls_in_key_columns():
    df = generate_synthetic_data()
    key = ['annual_revenue', 'ebitda', 'total_debt', 'default_indicator']
    assert df[key].isnull().sum().sum() == 0

def test_five_industries():
    df = generate_synthetic_data()
    assert df['industry'].nunique() == 5

def test_credit_score_range():
    df = generate_synthetic_data()
    assert df['credit_score'].between(300, 850).all()
