# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ASSUMPTION: These 18 ratios cover the 5 credit analysis dimensions from the spec.
# years_operating and prior_defaults are included as numeric features directly.
RATIO_COLS = [
    'current_ratio', 'quick_ratio', 'cash_ratio', 'working_capital_to_assets',
    'debt_to_ebitda', 'debt_to_assets', 'interest_coverage', 'dscr',
    'ebitda_margin', 'net_margin', 'roa',
    'asset_turnover', 'days_receivable', 'days_payable', 'cash_conversion_cycle',
    'credit_score_norm', 'years_operating', 'prior_defaults',
]


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 18 financial ratios from raw balance sheet / P&L data."""
    d = df.copy()
    eps = 1e-6

    # Approximate total assets (no explicit balance sheet provided)
    # ASSUMPTION: total_assets ≈ current_assets + 0.5 * annual_revenue (PP&E proxy)
    total_assets = d['current_assets'] + 0.5 * d['annual_revenue']
    net_income = (d['ebitda'] - d['interest_expense']) * (1.0 - d['tax_rate'])

    # Liquidity
    d['current_ratio'] = d['current_assets'] / (d['current_liabilities'] + eps)
    d['quick_ratio']   = (d['current_assets'] - d['inventory']) / (d['current_liabilities'] + eps)
    d['cash_ratio']    = d['cash_on_hand'] / (d['current_liabilities'] + eps)
    d['working_capital_to_assets'] = (
        (d['current_assets'] - d['current_liabilities']) / (total_assets + eps)
    )

    # Solvency
    # ASSUMPTION: ebitda.abs() used so negative-EBITDA firms produce a positive ratio (magnitude-based).
    d['debt_to_ebitda']    = d['total_debt'] / (d['ebitda'].abs() + eps)
    d['debt_to_assets']    = d['total_debt'] / (total_assets + eps)
    d['interest_coverage'] = d['ebitda'] / (d['interest_expense'] + eps)
    # DSCR: NOPAT / (interest + estimated principal = 10% of debt)
    d['dscr'] = (
        d['ebitda'] * (1.0 - d['tax_rate'])
        / (d['interest_expense'] + 0.1 * d['total_debt'] + eps)
    )

    # Profitability
    d['ebitda_margin'] = d['ebitda'] / (d['annual_revenue'] + eps)
    d['net_margin']    = net_income / (d['annual_revenue'] + eps)
    d['roa']           = net_income / (total_assets + eps)

    # Efficiency
    daily_revenue = d['annual_revenue'] / 365.0
    d['asset_turnover']         = d['annual_revenue'] / (total_assets + eps)
    d['days_receivable']        = d['accounts_receivable'] / (daily_revenue + eps)
    d['days_payable']           = d['accounts_payable'] / (daily_revenue + eps)
    # ASSUMPTION: CCC simplified to DSO - DPO; Days Inventory Outstanding omitted (inventory
    #             is captured separately via quick_ratio and cash_ratio).
    d['cash_conversion_cycle']  = d['days_receivable'] - d['days_payable']

    # Credit quality
    d['credit_score_norm'] = d['credit_score'] / 850.0
    # years_operating and prior_defaults already present in df — keep as-is

    return d


def winsorize_ratios(df: pd.DataFrame, ratio_cols: list) -> pd.DataFrame:
    """Clip each ratio at its 1st and 99th percentile."""
    d = df.copy()
    for col in ratio_cols:
        if col in d.columns:
            p1  = d[col].quantile(0.01)
            p99 = d[col].quantile(0.99)
            d[col] = d[col].clip(lower=p1, upper=p99)
    return d


def preprocess_features(
    df: pd.DataFrame,
    ratio_cols: list,
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
) -> tuple:
    """
    Standardize ratio_cols and one-hot encode industry (drop_first=True).

    Returns (X: pd.DataFrame, scaler: StandardScaler, feature_names: list).
    IMPORTANT: When fit_scaler=False, pass the scaler fitted on training data to avoid leakage.
    """
    d = df[ratio_cols + ['industry']].copy()

    industry_dummies = pd.get_dummies(d['industry'], prefix='industry', drop_first=True)
    d = d.drop(columns=['industry'])
    d = pd.concat([d, industry_dummies.astype(float)], axis=1)
    feature_names = list(d.columns)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(d.values.astype(float))
    else:
        if scaler is None:
            raise ValueError(
                "A fitted scaler must be provided when fit_scaler=False. "
                "Pass the scaler returned from the training call."
            )
        X_scaled = scaler.transform(d.values.astype(float))

    X = pd.DataFrame(X_scaled, columns=feature_names, index=df.index)
    return X, scaler, feature_names


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each feature in X."""
    X_vals = X.values.astype(float)
    records = []
    for i, col in enumerate(X.columns):
        vif = variance_inflation_factor(X_vals, i)
        records.append({'feature': col, 'VIF': round(float(vif), 2)})
    return pd.DataFrame(records).sort_values('VIF', ascending=False).reset_index(drop=True)
