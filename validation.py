# validation.py
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# ── Discrimination ─────────────────────────────────────────────────────────────

def compute_discrimination_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """AUC-ROC, Gini, K-S, Concentration Ratio (top 20%), Brier Score."""
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "y_true must contain both classes (0 and 1). "
            f"Got unique values: {np.unique(y_true)}"
        )
    auc  = roc_auc_score(y_true, y_pred_proba)
    gini = 2.0 * auc - 1.0

    defaults     = y_pred_proba[y_true == 1]
    non_defaults = y_pred_proba[y_true == 0]
    ks_stat, _   = stats.ks_2samp(defaults, non_defaults)

    n         = len(y_true)
    top20_idx = np.argsort(y_pred_proba)[::-1][: max(1, int(0.2 * n))]
    cr        = y_true[top20_idx].sum() / max(y_true.sum(), 1)

    brier = brier_score_loss(y_true, y_pred_proba)

    return {
        'auc_roc':              float(auc),
        'gini':                 float(gini),
        'ks_stat':              float(ks_stat),
        'concentration_ratio':  float(cr),
        'brier_score':          float(brier),
    }


# ── Calibration ────────────────────────────────────────────────────────────────

def hosmer_lemeshow_test(
    y_true: np.ndarray, y_pred_proba: np.ndarray, n_groups: int = 10
) -> dict:
    """
    Hosmer-Lemeshow goodness-of-fit test.
    H0: model is well-calibrated. p > 0.05 -> fail to reject -> good calibration.
    # ASSUMPTION: Small expected counts clamped to 1e-6 to avoid division by zero.
    """
    df = pd.DataFrame({'y': y_true, 'p': y_pred_proba})
    df['decile'] = pd.qcut(df['p'], q=n_groups, labels=False, duplicates='drop')

    grouped = df.groupby('decile').agg(
        observed=('y', 'sum'),
        expected=('p', 'sum'),
        n=('y', 'count'),
    )
    exp_safe    = np.maximum(grouped['expected'], 1e-6)
    non_exp     = grouped['n'] - grouped['expected']
    non_exp_safe = np.maximum(non_exp, 1e-6)
    chi2 = (
        (grouped['observed'] - grouped['expected']) ** 2 / exp_safe
        + (grouped['n'] - grouped['observed'] - non_exp) ** 2 / non_exp_safe
    ).sum()
    dof     = max(len(grouped) - 2, 1)
    p_value = float(1.0 - stats.chi2.cdf(chi2, dof))

    return {
        'hl_chi2':        float(chi2),
        'hl_df':          dof,
        'hl_p_value':     p_value,
        'n_groups_actual': len(grouped),
        'decile_table':   grouped,
    }


def compute_calibration_slope(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Regress actual defaults on log-odds of predicted PD.
    Slope = 1.0 -> perfectly calibrated.
    # ASSUMPTION: Uses logistic regression on log-odds (Platt scaling interpretation).
    """
    log_odds = np.log(
        np.clip(y_pred_proba, 1e-8, 1 - 1e-8)
        / np.clip(1.0 - y_pred_proba, 1e-8, 1 - 1e-8)
    )
    cal_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
    cal_model.fit(log_odds.reshape(-1, 1), y_true)
    return float(cal_model.coef_[0][0])


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, auc: float, output_dir: str = 'outputs'
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Credit Risk Model')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, output_dir: str = 'outputs'
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_pred, frac_pos, 's-', color='steelblue', label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=150)
    plt.close(fig)


def plot_feature_importance(coef_df: pd.DataFrame, output_dir: str = 'outputs') -> None:
    os.makedirs(output_dir, exist_ok=True)
    top = coef_df.head(15).copy()
    colors = ['#c0392b' if c > 0 else '#27ae60' for c in top['coefficient']]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top['feature'].iloc[::-1], top['coefficient'].abs().iloc[::-1],
            color=colors[::-1])
    ax.set_xlabel('|Coefficient|  (red = increases PD risk, green = decreases)')
    ax.set_title('Feature Importance — Top 15 by |Coefficient|')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
    plt.close(fig)


def plot_vif_table(vif_df: pd.DataFrame, output_dir: str = 'outputs') -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(vif_df) * 0.28)))
    ax.axis('off')
    tbl = ax.table(
        cellText=vif_df.values,
        colLabels=vif_df.columns,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    ax.set_title('VIF Table (flag: VIF > 5)', pad=16)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'vif_table.png'), dpi=150)
    plt.close(fig)
