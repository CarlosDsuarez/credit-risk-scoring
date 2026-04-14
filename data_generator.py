import numpy as np
import pandas as pd


def generate_synthetic_data(n_obs: int = 800, default_rate: float = 0.08, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic borrower dataset with realistic correlation structure.
    Default rate is driven by Debt/EBITDA, Interest Coverage, prior defaults, credit score.

    # ASSUMPTION: accounting identity Assets = Liabilities + Equity is approximated via
    # total_assets_approx = current_assets + 0.5 * annual_revenue (used downstream in ratios).
    # ASSUMPTION: default rate is calibrated to ~8% via the intercept term (-3.0).
    """
    # NOTE: `default_rate` parameter is kept for API compatibility but does not dynamically
    #       control the actual rate. Actual rate (~8%) is determined by the logistic intercept.
    rng = np.random.default_rng(seed)
    industries = ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Energy']

    # --- Revenue & operating metrics ---
    annual_revenue = rng.lognormal(mean=10.5, sigma=1.2, size=n_obs)  # ~$50M median
    ebitda_margin = rng.normal(0.15, 0.08, size=n_obs)
    ebitda = annual_revenue * ebitda_margin

    # --- Debt & interest ---
    # ASSUMPTION: Debt/EBITDA drawn from lognormal(1.1, 0.5) -> median ~3x
    debt_ebitda_ratio = rng.lognormal(mean=1.1, sigma=0.5, size=n_obs)
    # ASSUMPTION: total_debt uses abs(ebitda) for the ~3% of rows where ebitda < 0.
    # This treats debt_ebitda_ratio as a magnitude ratio regardless of EBITDA sign.
    total_debt = np.maximum(np.abs(ebitda) * debt_ebitda_ratio, 0.0)
    interest_rate = rng.uniform(0.04, 0.08, size=n_obs)
    interest_expense = total_debt * interest_rate
    tax_rate = rng.uniform(0.15, 0.35, size=n_obs)

    # --- Balance sheet ---
    current_liabilities = annual_revenue * rng.uniform(0.10, 0.30, size=n_obs)
    current_ratio_sample = rng.lognormal(mean=0.5, sigma=0.3, size=n_obs)
    current_assets = current_liabilities * current_ratio_sample
    inventory = current_assets * rng.uniform(0.20, 0.40, size=n_obs)
    accounts_receivable = current_assets * rng.uniform(0.20, 0.40, size=n_obs)
    accounts_payable = current_liabilities * rng.uniform(0.30, 0.60, size=n_obs)
    cash_on_hand = np.maximum(current_assets - inventory - accounts_receivable, 0.0)

    # --- Credit quality & loan ---
    credit_score = rng.normal(650, 80, size=n_obs).clip(300, 850)
    prior_defaults = rng.choice([0, 1], size=n_obs, p=[0.85, 0.15])
    loan_amount = annual_revenue * rng.uniform(0.10, 0.50, size=n_obs)
    loan_tenor = rng.choice([1, 2, 3, 5, 7, 10], size=n_obs)
    years_operating = rng.integers(1, 30, size=n_obs)
    industry = rng.choice(industries, size=n_obs)

    # --- Default indicator via latent logistic model ---
    ic = ebitda / np.maximum(interest_expense, 1.0)  # interest coverage
    de = debt_ebitda_ratio

    log_odds = (
        -3.0
        + 0.50 * (de - de.mean()) / (de.std() + 1e-8)
        - 0.40 * (ic - ic.mean()) / (ic.std() + 1e-8)
        + 0.60 * prior_defaults
        - 0.30 * (credit_score - credit_score.mean()) / (credit_score.std() + 1e-8)
        + 0.20 * rng.standard_normal(n_obs)
    )
    pd_true = 1.0 / (1.0 + np.exp(-log_odds))
    default_indicator = (rng.uniform(size=n_obs) < pd_true).astype(int)

    return pd.DataFrame({
        'borrower_id': [f'B{i:04d}' for i in range(n_obs)],
        'default_indicator': default_indicator,
        'annual_revenue': annual_revenue,
        'ebitda': ebitda,
        'total_debt': total_debt,
        'cash_on_hand': cash_on_hand,
        'current_assets': current_assets,
        'current_liabilities': current_liabilities,
        'inventory': inventory,
        'accounts_receivable': accounts_receivable,
        'accounts_payable': accounts_payable,
        'interest_expense': interest_expense,
        'tax_rate': tax_rate,
        'industry': industry,
        'years_operating': years_operating.astype(float),
        'credit_score': credit_score,
        'prior_defaults': prior_defaults.astype(float),
        'loan_amount': loan_amount,
        'loan_tenor': loan_tenor.astype(float),
    })
