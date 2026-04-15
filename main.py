# main.py
"""
End-to-end Credit Risk Scoring Pipeline.

Run:  python main.py

Outputs:
  - Validation Report printed to stdout
  - Plots saved to outputs/ directory
  - Returns model_artifacts dict for downstream use (e.g., predict.py)
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split

from data_generator import generate_synthetic_data
from feature_engineering import (
    compute_financial_ratios, winsorize_ratios,
    preprocess_features, compute_vif, RATIO_COLS,
)
from model import train_model, get_coefficients
from validation import (
    compute_discrimination_metrics, hosmer_lemeshow_test, compute_calibration_slope,
    plot_roc_curve, plot_calibration_curve, plot_feature_importance, plot_vif_table,
)
from risk_categories import (
    compute_thresholds, assign_risk_categories, validate_categories, get_risk_categories_dict,
)
from stress_testing import recession_scenario, industry_cliff_risk
from report import print_validation_report

OUTPUT_DIR = 'outputs'


def run_pipeline() -> dict:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 58)
    print('CREDIT RISK SCORING MODEL — PIPELINE START')
    print('=' * 58)

    # ── 1. Generate data ──────────────────────────────────────────
    print('\n[1/8] Generating synthetic data...')
    df = generate_synthetic_data(n_obs=800, default_rate=0.08, seed=42)
    print(f'      {len(df)} borrowers | Default rate: {df["default_indicator"].mean():.1%}')

    # ── 2. Feature engineering ────────────────────────────────────
    print('[2/8] Computing financial ratios and preprocessing...')
    df_ratios = compute_financial_ratios(df)

    # ── 3. Stratified train/test split ───────────────────────────
    print('[3/8] Splitting train/test (70/30 stratified)...')
    y = df_ratios['default_indicator'].values
    train_idx, test_idx = train_test_split(
        np.arange(len(df_ratios)), test_size=0.3, stratify=y, random_state=42
    )
    df_train     = df_ratios.iloc[train_idx].reset_index(drop=True)
    df_test      = df_ratios.iloc[test_idx].reset_index(drop=True)
    df_orig_test = df.iloc[test_idx].reset_index(drop=True)

    # Winsorize using TRAIN bounds only — apply same bounds to TEST (no leakage)
    df_train = winsorize_ratios(df_train, RATIO_COLS)
    winsorize_bounds = {
        col: (df_train[col].quantile(0.01), df_train[col].quantile(0.99))
        for col in RATIO_COLS if col in df_train.columns
    }
    for col, (lo, hi) in winsorize_bounds.items():
        if col in df_test.columns:
            df_test[col] = df_test[col].clip(lower=lo, upper=hi)

    y_train = df_train['default_indicator'].values
    y_test  = df_test['default_indicator'].values

    # Fit scaler on TRAIN only — apply to TEST (no leakage)
    X_train, scaler, feature_names = preprocess_features(df_train, RATIO_COLS, fit_scaler=True)
    X_test, _, _ = preprocess_features(df_test, RATIO_COLS, scaler=scaler, fit_scaler=False)
    X_test = X_test.reindex(columns=feature_names, fill_value=0.0)

    vif_df = compute_vif(X_train)

    # ── 4. Train model ────────────────────────────────────────────
    print('[4/8] Training logistic regression (L2, CV tuning)...')
    train_result = train_model(X_train.values, y_train)
    model = train_result['model']
    print(
        f'      Best C: {train_result["best_C"]} | '
        f'CV AUC: {train_result["cv_auc_mean"]:.4f} +/- {train_result["cv_auc_std"]:.4f}'
    )

    # ── 5. Coefficients ───────────────────────────────────────────
    print('[5/8] Computing coefficient standard errors (200 bootstrap resamples)...')
    coef_df = get_coefficients(model, feature_names, X_train.values, y_train, n_bootstrap=200)

    # ── 6. Validation ─────────────────────────────────────────────
    print('[6/8] Validating on hold-out test set...')
    y_pred_proba = model.predict_proba(X_test.values)[:, 1]
    disc_metrics = compute_discrimination_metrics(y_test, y_pred_proba)
    hl_result    = hosmer_lemeshow_test(y_test, y_pred_proba)
    cal_slope    = compute_calibration_slope(y_test, y_pred_proba)
    print(f'      Test AUC-ROC: {disc_metrics["auc_roc"]:.4f}')

    plot_roc_curve(y_test, y_pred_proba, disc_metrics['auc_roc'], OUTPUT_DIR)
    plot_calibration_curve(y_test, y_pred_proba, OUTPUT_DIR)
    plot_feature_importance(coef_df, OUTPUT_DIR)
    plot_vif_table(vif_df, OUTPUT_DIR)

    # ── 7. Risk categorization ────────────────────────────────────
    print('[7/8] Assigning risk categories...')
    pd_train      = model.predict_proba(X_train.values)[:, 1]
    t25, t75      = compute_thresholds(pd_train)
    cats_test     = assign_risk_categories(y_pred_proba, t25, t75)
    cat_stats     = validate_categories(y_test, cats_test)
    risk_cat_dict = get_risk_categories_dict(t25, t75)

    # ── 8. Stress testing ─────────────────────────────────────────
    print('[8/8] Running stress tests...')
    rec_result   = recession_scenario(df_train, model, scaler, feature_names, t25, t75)
    cliff_df     = industry_cliff_risk(df_orig_test, cats_test)

    # ── Report ────────────────────────────────────────────────────
    metrics = {
        'discrimination': disc_metrics,
        'calibration':    {**hl_result, 'calibration_slope': cal_slope},
        'cv':             {
            'cv_auc_mean': train_result['cv_auc_mean'],
            'cv_auc_std':  train_result['cv_auc_std'],
        },
        'coef_df':        coef_df,
        'vif_df':         vif_df,
        'category_stats': cat_stats,
        'stress':         {**rec_result, 'industry_cliff': cliff_df},
        'risk_categories': risk_cat_dict,
    }
    print_validation_report(metrics)

    print(f'Plots saved to: {OUTPUT_DIR}/')
    print('Pipeline complete.\n')

    # Build model_artifacts for predict.py
    # winsorize_bounds computed earlier from training set (no leakage)
    return {
        'model':            model,
        'scaler':           scaler,
        'feature_names':    feature_names,
        'threshold_25':     t25,
        'threshold_75':     t75,
        'coef_df':          coef_df,
        'winsorize_bounds': winsorize_bounds,
    }


if __name__ == '__main__':
    run_pipeline()
