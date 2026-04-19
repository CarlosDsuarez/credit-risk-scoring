# Credit Risk Scoring Model

End-to-end pipeline for scoring borrower default probability using logistic regression with L2 regularization. Covers data generation, feature engineering, model training, validation, risk categorization, and stress testing.

## Features

- **Synthetic data generation** — 800 borrowers, configurable default rate
- **Feature engineering** — financial ratios (leverage, liquidity, coverage, etc.), industry dummies, VIF analysis
- **No data leakage** — train/test split before winsorization and scaling; all transforms fitted on train only
- **Model** — logistic regression with cross-validated L2 tuning
- **Validation** — AUC-ROC, Hosmer-Lemeshow test, calibration slope, bootstrap standard errors
- **Risk categories** — LOW / MEDIUM / HIGH based on PD thresholds derived from training set
- **Stress testing** — recession scenario and industry cliff-risk analysis
- **Single-borrower scoring** — `predict.py` returns PD, 95% CI, risk category, top-3 risk drivers, and recommendation

## Project Structure

```
credit_risk_scoring/
├── main.py               # Pipeline orchestrator — run this
├── predict.py            # Single-borrower scoring API
├── data_generator.py     # Synthetic dataset generation
├── feature_engineering.py
├── model.py              # Logistic regression training + bootstrap CIs
├── validation.py         # Metrics and plots
├── risk_categories.py    # Threshold computation and category assignment
├── stress_testing.py     # Recession and industry cliff scenarios
├── report.py             # Validation report to stdout
├── tests/                # Unit tests
├── outputs/              # Generated plots (ROC, calibration, feature importance, VIF)
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

Output: validation report in stdout + 4 plots saved to `outputs/`.

## Single-Borrower Scoring

```python
from main import run_pipeline
from predict import predict_default_probability

artifacts = run_pipeline()

result = predict_default_probability(
    borrower_financials={
        "borrower_id": "B001",
        "total_assets": 500_000,
        "total_liabilities": 300_000,
        "ebitda": 80_000,
        "interest_expense": 15_000,
        "current_assets": 120_000,
        "current_liabilities": 60_000,
        "net_income": 40_000,
        "revenue": 400_000,
        "industry": "manufacturing",
    },
    model_artifacts=artifacts,
)

print(result)
# {
#   'borrower_id': 'B001',
#   'predicted_pd': 0.0412,
#   'risk_category': 'LOW',
#   'confidence_interval_95': (0.021, 0.068),
#   'top_3_risk_drivers': [('debt_to_equity', 0.312), ...],
#   'recommendation': 'Approve -- low credit risk. Standard terms apply.'
# }
```

## Outputs

| File | Description |
|------|-------------|
| `outputs/roc_curve.png` | ROC curve with AUC |
| `outputs/calibration_curve.png` | Calibration (reliability) curve |
| `outputs/feature_importance.png` | Coefficients with bootstrap CIs |
| `outputs/vif_table.png` | Variance inflation factors |

## Requirements

- Python 3.8+
- numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, statsmodels

## Tests

```bash
pytest tests/
```
