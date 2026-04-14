# report.py
import pandas as pd


def print_validation_report(metrics: dict) -> None:
    """
    Print the 9-section institutional Validation Report to stdout.

    Expected keys in metrics:
        discrimination: dict from compute_discrimination_metrics()
        calibration:    dict -- keys: hl_p_value, calibration_slope, decile_table
        cv:             dict -- keys: cv_auc_mean, cv_auc_std
        coef_df:        pd.DataFrame from get_coefficients()
        vif_df:         pd.DataFrame from compute_vif()
        category_stats: dict from validate_categories()
        stress:         dict -- keys: pct_high_risk_recession, industry_cliff (pd.DataFrame)
        risk_categories: dict from get_risk_categories_dict()
    """
    LINE = '=' * 58
    THIN = '-' * 58

    def _status(auc, hl_p):
        if auc >= 0.75 and hl_p > 0.05:
            return 'PRODUCTION READY', 'Well-calibrated'
        if auc >= 0.70:
            return 'PILOT / MONITORING REQUIRED', 'Adequate'
        return 'RECALIBRATION NEEDED', 'Under-performing'

    disc = metrics['discrimination']
    cal  = metrics['calibration']
    cv   = metrics['cv']
    auc  = disc['auc_roc']
    hl_p = cal['hl_p_value']
    use_status, cal_status = _status(auc, hl_p)

    print(f'\n{"CREDIT RISK SCORING MODEL -- VALIDATION REPORT":^58}')
    print(LINE)

    # 1. Executive Summary
    print('\n1. EXECUTIVE SUMMARY')
    print(THIN)
    print(f'   Model AUC-ROC:          {auc:.4f}')
    print(f'   Calibration Status:     {cal_status}')
    print(f'   Recommended Use:        {use_status}')

    # 2. Discrimination
    print('\n2. DISCRIMINATION METRICS (Test Set)')
    print(THIN)
    print(f'   AUC-ROC:             {disc["auc_roc"]:.4f}   (Target: > 0.75)')
    print(f'   Gini:                {disc["gini"]:.4f}   (Target: > 0.50)')
    print(f'   K-S Statistic:       {disc["ks_stat"]:.4f}   (Target: > 0.30)')
    print(f'   Concentration Ratio: {disc["concentration_ratio"]:.2%}  (Target: 50-70%)')
    print(f'   Brier Score:         {disc["brier_score"]:.4f}   (Lower = better)')

    # 3. Calibration
    print('\n3. CALIBRATION METRICS (Test Set)')
    print(THIN)
    print(f'   Hosmer-Lemeshow p:   {hl_p:.4f}   (Target: > 0.05)')
    print(f'   Calibration Slope:   {cal["calibration_slope"]:.4f}   (Target: 1.0 +/- 0.15)')
    decile_tbl = cal.get('decile_table')
    if decile_tbl is not None:
        print('\n   Actual vs Predicted Default Rate by Decile:')
        print(f'   {"Decile":>7} {"Observed":>10} {"Expected":>10} {"N":>6}')
        for idx, row in decile_tbl.iterrows():
            print(f'   {int(idx):>7} {int(row["observed"]):>10} {row["expected"]:>10.1f} {int(row["n"]):>6}')

    # 4. Feature Importance
    coef_df = metrics.get('coef_df')
    vif_df  = metrics.get('vif_df')
    print('\n4. FEATURE IMPORTANCE & INTERPRETATION')
    print(THIN)
    if coef_df is not None:
        print(f'   {"Feature":<32} {"Coef":>8} {"SE":>8} {"z":>7} {"p":>8}')
        for _, row in coef_df.head(10).iterrows():
            print(
                f'   {row["feature"]:<32} {row["coefficient"]:>8.4f}'
                f' {row["std_error"]:>8.4f} {row["z_stat"]:>7.2f} {row["p_value"]:>8.4f}'
            )
    if vif_df is not None:
        flagged = vif_df[vif_df['VIF'] > 5]
        if len(flagged) > 0:
            print(f'\n   Features with VIF > 5 (multicollinearity flag):')
            for _, row in flagged.iterrows():
                print(f'     {row["feature"]}: VIF = {row["VIF"]:.1f}')
        else:
            print('\n   No features with VIF > 5.')

    # 5. Cross-Validation Stability
    print('\n5. CROSS-VALIDATION STABILITY')
    print(THIN)
    print(f'   5-Fold AUC-ROC:      {cv["cv_auc_mean"]:.4f} +/- {cv["cv_auc_std"]:.4f}')
    stable = 'Stable' if cv['cv_auc_std'] < 0.03 else 'Unstable -- high variance across folds'
    print(f'   Conclusion:          {stable}')

    # 6. Risk Categories
    cat_stats = metrics.get('category_stats', {})
    total = sum(v.get('count', 0) for v in cat_stats.values())
    print('\n6. RISK CATEGORIZATION DISTRIBUTION (Test Set)')
    print(THIN)
    for cat in ['LOW', 'MEDIUM', 'HIGH']:
        if cat in cat_stats:
            v   = cat_stats[cat]
            pct = v['count'] / max(total, 1)
            dr  = v['default_rate']
            print(f'   {cat:<8}: {pct:>6.1%} of test set | Default rate: {dr:.2%}')

    # 7. Stress Tests
    stress = metrics.get('stress', {})
    print('\n7. STRESS TEST RESULTS')
    print(THIN)
    rec_pct = stress.get('pct_high_risk_recession', 0.0)
    print(f'   Recession scenario:  {rec_pct:.1%} of portfolio migrates to HIGH risk')
    cliff = stress.get('industry_cliff')
    if cliff is not None:
        flagged_ind = cliff[cliff['flagged']]
        if len(flagged_ind) > 0:
            names = ', '.join(flagged_ind['industry'].tolist())
            print(f'   Industry cliff risk: {names} (>2x avg HIGH concentration)')
        else:
            print('   Industry cliff risk: No sectors flagged.')

    # 8. Limitations
    print('\n8. MODEL LIMITATIONS & RECOMMENDATIONS')
    print(THIN)
    print('   Known weaknesses:')
    print('     - Synthetic data: real-world performance may differ')
    print('     - Cross-sectional snapshot: no panel/temporal dynamics captured')
    print('     - Linear log-odds assumption: non-linear interactions not modelled')
    print('   Retraining frequency: Quarterly, or after major macroeconomic shifts')
    print('   Next steps: Panel data, macro overlay, GBM benchmark comparison')

    # 9. Technical Appendix
    print('\n9. TECHNICAL APPENDIX')
    print(THIN)
    print('   Model equation: log(PD/(1-PD)) = beta_0 + SUM(beta_i * Ratio_i) + Industry_FE')
    print('   Regularization: L2 (Ridge), lambda tuned via 5-fold StratifiedKFold CV')
    print('   Class imbalance: class_weight="balanced"')
    print('   Coefficient SE:  Bootstrap (200 resamples of training set)')
    print('   Random seed:     42 throughout pipeline')
    print('   Training data:   Synthetic, 800 obs, ~8% default rate')
    print(LINE)
    print()
