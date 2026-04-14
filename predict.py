# predict.py
import numpy as np
import pandas as pd
from feature_engineering import compute_financial_ratios, RATIO_COLS


_RECOMMENDATIONS = {
    'LOW':    'Approve -- low credit risk. Standard terms apply.',
    'MEDIUM': 'Conditional approval -- enhanced monitoring recommended.',
    'HIGH':   'Decline or escalate to senior credit committee.',
}


def predict_default_probability(borrower_financials: dict, model_artifacts: dict) -> dict:
    """
    Score a single borrower.

    Input:  dict with same keys as the raw dataset (from data_generator).
    Output: {
        borrower_id, predicted_pd, risk_category,
        confidence_interval_95, top_3_risk_drivers, recommendation
    }

    # ASSUMPTION: borrower_financials includes 'industry' for dummy encoding.
    # ASSUMPTION: winsorize_bounds from model_artifacts are applied before scaling.
    # ASSUMPTION: CI computed via +/-1.96 SE of the linear predictor.
    # ASSUMPTION: Risk drivers = coefficient x scaled feature value (log-odds contribution).
    """
    borrower_id = borrower_financials.get('borrower_id', 'UNKNOWN')

    # 1. Build single-row DataFrame and compute ratios
    df = pd.DataFrame([borrower_financials])
    df_ratios = compute_financial_ratios(df)

    # 2. Winsorize using training bounds
    bounds = model_artifacts.get('winsorize_bounds', {})
    for col, (lo, hi) in bounds.items():
        if col in df_ratios.columns:
            df_ratios[col] = df_ratios[col].clip(lower=lo, upper=hi)

    # 3. Scale using training scaler (no fit -- prevent leakage)
    # Build feature matrix manually to handle industry dummies correctly:
    # a single borrower may only have one industry, so pd.get_dummies produces
    # fewer columns than training. We therefore:
    #   i.  Extract ratio columns directly from df_ratios.
    #   ii. One-hot encode the industry column and reindex to the full set of
    #       industry dummy columns seen at training time (filling missing with 0).
    #   iii. Concatenate and apply the fitted scaler column-by-column using the
    #        scaler's stored mean_ and scale_ after aligning to feature_names.
    scaler        = model_artifacts['scaler']
    feature_names = model_artifacts['feature_names']

    # Determine which feature_names are industry dummies vs ratio cols
    industry_dummy_cols = [f for f in feature_names if f.startswith('industry_')]

    # Build ratio part
    ratio_part = df_ratios[RATIO_COLS].copy().reset_index(drop=True)

    # Build industry dummy part aligned to training columns
    industry_dummies = pd.get_dummies(df_ratios['industry'], prefix='industry', drop_first=True)
    industry_dummies = industry_dummies.reindex(columns=industry_dummy_cols, fill_value=0.0).astype(float)
    industry_dummies = industry_dummies.reset_index(drop=True)

    # Combine into aligned DataFrame matching feature_names order
    full_df = pd.concat([ratio_part, industry_dummies], axis=1)
    full_df = full_df.reindex(columns=feature_names, fill_value=0.0)

    # Apply fitted scaler transform
    X_arr = scaler.transform(full_df.values.astype(float))  # shape (1, n_features)

    # 4. Predict PD
    model     = model_artifacts['model']
    pd_score  = float(model.predict_proba(X_arr)[0, 1])

    # 5. Confidence interval via linear predictor SE
    log_odds = float(model.decision_function(X_arr)[0])
    coef_df  = model_artifacts.get('coef_df')
    if coef_df is not None:
        std_errs = (
            coef_df.set_index('feature')['std_error']
            .reindex(feature_names)
            .fillna(0.0)
            .values
        )
        se_lp = float(np.sqrt(((X_arr[0] * std_errs) ** 2).sum()))
    else:
        se_lp = 0.10  # fallback SE
    lo_lp = log_odds - 1.96 * se_lp
    hi_lp = log_odds + 1.96 * se_lp
    ci_lo = float(1.0 / (1.0 + np.exp(-lo_lp)))
    ci_hi = float(1.0 / (1.0 + np.exp(-hi_lp)))

    # 6. Risk category
    t25 = model_artifacts['threshold_25']
    t75 = model_artifacts['threshold_75']
    if pd_score < t25:
        category = 'LOW'
    elif pd_score < t75:
        category = 'MEDIUM'
    else:
        category = 'HIGH'

    # 7. Top-3 risk drivers (log-odds contribution = coef x scaled feature)
    coefs         = model.coef_[0]
    contributions = coefs * X_arr[0]
    top3_idx      = np.argsort(np.abs(contributions))[::-1][:3]
    top_3         = [(feature_names[i], round(float(contributions[i]), 4)) for i in top3_idx]

    return {
        'borrower_id':            borrower_id,
        'predicted_pd':           round(pd_score, 4),
        'risk_category':          category,
        'confidence_interval_95': (round(min(ci_lo, ci_hi), 4), round(max(ci_lo, ci_hi), 4)),
        'top_3_risk_drivers':     top_3,
        'recommendation':         _RECOMMENDATIONS[category],
    }
