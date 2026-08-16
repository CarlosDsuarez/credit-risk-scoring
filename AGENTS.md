# AGENTS.md

## Cursor Cloud specific instructions

This repo is a **self-contained Python credit-risk scoring pipeline**. There is no web
server, database, or external service — the "application" is a script that runs a full
train/validate/report pipeline and writes plots to `outputs/`.

### Environment

- Dependencies are installed into a virtualenv at `.venv/` (they are **not** installed
  into the system Python). Run everything with `.venv/bin/python` (or activate the venv).
  The startup update script recreates/refreshes this venv from `requirements.txt`.

### Run / test / lint

- Run the pipeline (the application): `.venv/bin/python main.py`
  Prints a validation report to stdout and writes 4 PNGs to `outputs/`.
- Tests: `.venv/bin/python -m pytest tests/`
- No linter is configured in this repo. Use `.venv/bin/python -m py_compile *.py tests/*.py`
  as a quick syntax sanity check.

### Non-obvious gotchas

- The README's single-borrower `predict_default_probability` example uses **stale keys**
  (e.g. `total_liabilities`, `net_income`) that do not match the code. The real input
  schema is the raw dataset produced by `data_generator.generate_synthetic_data`, i.e.
  `compute_financial_ratios` needs: `annual_revenue`, `ebitda`, `total_debt`,
  `cash_on_hand`, `current_assets`, `current_liabilities`, `inventory`,
  `accounts_receivable`, `accounts_payable`, `interest_expense`, `tax_rate`, `industry`,
  `years_operating`, `credit_score`, `prior_defaults`. Industry values are capitalized
  (e.g. `"Manufacturing"`, `"Technology"`). To score a borrower, pass a dict matching that
  schema (see `tests/test_predict.py`, which uses a real dataset row).
- Installed scikit-learn is newer than when the code was written, so runs emit
  `FutureWarning: 'penalty' was deprecated` from `sklearn`. These are expected noise, not
  errors — the pipeline and all tests pass.
- The pipeline is fully deterministic (`seed=42`); rerunning `main.py` overwrites the PNGs
  in `outputs/` (which are checked into git). Avoid committing those regenerated binaries
  unless the plots meaningfully changed.
