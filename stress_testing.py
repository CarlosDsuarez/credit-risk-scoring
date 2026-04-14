import numpy as np
import pandas as pd
from feature_engineering import preprocess_features, RATIO_COLS


def univariate_sensitivity(
    model,
    feature_names: list,
    X_median: np.ndarray,
) -> pd.DataFrame:
    """
    Hold all features at median (0 in standardized space), shock one +/-1 SD at a time.
    Records baseline PD, shocked PD, and delta for each feature/direction.
    # ASSUMPTION: X_median is in standardized space; +/-1 unit = +/-1 SD of original ratio.
    """
    baseline_pd = float(model.predict_proba(X_median.reshape(1, -1))[0, 1])
    records = []
    for i, feature in enumerate(feature_names):
        for direction, label in [(1, '+1 SD'), (-1, '-1 SD')]:
            X_shocked = X_median.copy()
            X_shocked[i] += direction
            shocked_pd = float(model.predict_proba(X_shocked.reshape(1, -1))[0, 1])
            records.append({
                'feature':     feature,
                'shock':       label,
                'baseline_pd': baseline_pd,
                'shocked_pd':  shocked_pd,
                'delta_pd':    shocked_pd - baseline_pd,
            })
    return pd.DataFrame(records).sort_values('delta_pd', key=abs, ascending=False).reset_index(drop=True)


def recession_scenario(
    df_ratios: pd.DataFrame,
    model,
    scaler,
    feature_names: list,
    threshold_25: float,
    threshold_75: float,
) -> dict:
    """
    Apply recession shocks to ratio-level DataFrame, re-scale with training scaler,
    and measure portfolio migration to HIGH risk.

    Shocks:
    - ebitda_margin   - 0.05  (-500 bps)
    - net_margin      - 0.04  (-400 bps)
    - asset_turnover  x 0.80  (revenue -20% proxy)
    - roa             x 0.80

    # ASSUMPTION: Shocks applied to winsorized ratios before re-scaling.
    # ASSUMPTION: Industry dummies re-aligned to feature_names after preprocess_features.
    """
    df_rec = df_ratios.copy()
    for col, shock in [('ebitda_margin', -0.05), ('net_margin', -0.04)]:
        if col in df_rec.columns:
            df_rec[col] = df_rec[col] + shock
    for col in ['asset_turnover', 'roa']:
        if col in df_rec.columns:
            df_rec[col] = df_rec[col] * 0.80

    X_rec, _, _ = preprocess_features(df_rec, RATIO_COLS, scaler=scaler, fit_scaler=False)
    X_rec_aligned = X_rec.reindex(columns=feature_names, fill_value=0.0)

    pd_rec = model.predict_proba(X_rec_aligned.values)[:, 1]
    cats = [
        'HIGH' if p >= threshold_75 else ('LOW' if p < threshold_25 else 'MEDIUM')
        for p in pd_rec
    ]
    return {
        'pct_high_risk_recession': float(sum(c == 'HIGH' for c in cats) / len(cats)),
        'mean_pd_recession':       float(pd_rec.mean()),
    }


def industry_cliff_risk(df: pd.DataFrame, categories: list) -> pd.DataFrame:
    """
    For each industry, compute % of borrowers in HIGH risk bucket.
    Flag industries where HIGH concentration > 2x portfolio average.
    """
    d = df[['industry']].copy()
    d['category'] = categories
    industry_high = d.groupby('industry')['category'].apply(
        lambda x: (x == 'HIGH').mean()
    )
    avg_high = float(industry_high.mean())
    result = pd.DataFrame({
        'industry':      industry_high.index,
        'pct_high_risk': industry_high.values,
        'vs_average':    industry_high.values / (avg_high + 1e-10),
        'flagged':       industry_high.values > 2.0 * avg_high,
    }).sort_values('pct_high_risk', ascending=False).reset_index(drop=True)
    return result
