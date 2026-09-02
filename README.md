# Credit Risk Scorecard & Model Validation Framework

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

An end-to-end, from-scratch credit risk scorecard built on Lending Club's full
historical loan portfolio (**1,268,686 loans**, 2007–2018), paired with an
independent **model validation framework** of the kind a bank's Model Risk
Management (MRM) function runs before a scorecard goes into production.

This isn't just a model that predicts default. It's the full lifecycle: feature
engineering via Weight-of-Evidence (WOE) and Information Value (IV), a
traditional points-based scorecard fit with logistic regression, and a
validation battery covering discrimination power, population stability,
calibration, fairness, and forward-looking loss forecasting.

## At a Glance

| Metric | Value |
|---|---|
| Portfolio size | 1,268,686 loans ($18.5B exposure) |
| Overall default rate | 19.55% |
| Vintages covered | 2007-06 to 2018-12 |
| Features engineered / selected | 17 evaluated → 7 selected by Information Value |
| Discrimination (OOT) | KS = 21.9, Gini = 0.309, AUC = 0.654 |
| Population stability (OOT) | PSI = 0.0172 (stable) |
| Fairness check (four-fifths rule) | Disparate impact ratio = 1.003 (pass) |
| Total expected loss | $2.28B (12.3% of exposure) |
| Automated validation checks passing | 5 / 6 (1 flagged for review — see [Known Limitations](#known-limitations--honest-caveats)) |

A full, pre-generated HTML validation report is included at
[`outputs/validation_report.html`](outputs/validation_report.html) — open it
directly in a browser to see every table and chart referenced below without
re-running anything.

## Table of Contents

- [Why This Project](#why-this-project)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [A Deliberate Design Decision: Excluding `grade` and `int_rate`](#a-deliberate-design-decision-excluding-grade-and-int_rate)
- [Known Limitations & Honest Caveats](#known-limitations--honest-caveats)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Skills Demonstrated](#skills-demonstrated)
- [License](#license)

## Why This Project

Underwriting a loan and *validating* an underwriting model are two different
disciplines. Most "build a credit model" tutorials stop at a trained
classifier. This project goes further, structured explicitly around the
questions a model risk / quantitative analytics team actually has to answer
before a scorecard is trusted:

- Does the model discriminate good loans from bad ones better than chance,
  out of sample and out of time?
- Is its coefficient behavior sane, right signs, no hidden multicollinearity?
- Has the scored population drifted since the model was built?
- Does the model treat groups disparately under a standard fair-lending test?
- What does this model imply about future portfolio losses?

Every one of those questions gets its own module, its own metrics, and its
own pass/fail judgment in the final report, not just a single accuracy
number.

## Repository Structure

```
├── src/
│   ├── data_prep.py         # Loads & cleans the raw Lending Club CSV; time-based train/test/OOT split
│   ├── woe_binning.py       # Weight-of-Evidence binning + Information Value feature ranking
│   ├── scorecard.py         # Logistic regression scorecard, PDO points scaling, VIF checks
│   ├── loss_forecast.py     # Vintage curves, roll-rate transition matrix, expected-loss (PD×LGD×EAD)
│   ├── validation.py        # KS, Gini/AUC, PSI, calibration, disparate impact (four-fifths rule)
│   ├── report.py            # Assembles every module's output into one HTML report
│   └── run_pipeline.py      # End-to-end orchestrator: data → scorecard → validation → report
├── data/                    # Not tracked in git — see Setup below
├── outputs/
│   └── validation_report.html   # Pre-generated report from the full 1.27M-loan run
├── requirements.txt
└── LICENSE
```

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/<your-username>/Credit-Risk-Scorecard-Model-Validation-Framework.git
cd Credit-Risk-Scorecard-Model-Validation-Framework

python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

**Dataset:** This project runs on Lending Club's public "accepted loans"
historical dataset (search "Lending Club Loan Data" on Kaggle). It's roughly
1.7 GB and is intentionally excluded from this repository via `.gitignore`
download it yourself and place it (rename to : loan.csv) at:

```
data/loan.csv
```

`data_prep.py` expects the standard Lending Club column set (`loan_amnt`,
`int_rate`, `grade`, `fico_range_low/high`, `issue_d`, `loan_status`, etc.) —
the accepted-loans CSV from Kaggle matches this out of the box.

## Usage

Each module can be run on its own to inspect that stage in isolation:

```bash
python src/data_prep.py       # sanity-check the load + time-based split
python src/woe_binning.py     # WOE bins + Information Value ranking
python src/scorecard.py       # fit the logistic regression scorecard
python src/loss_forecast.py   # vintage curves, roll-rates, expected loss (sampled for speed)
python src/validation.py      # discrimination, PSI, fairness metrics
```

Or run the full pipeline end to end, which reproduces everything in this
README and regenerates the HTML report:

```bash
python src/run_pipeline.py --data data/loan.csv
```

Optional flags:

```bash
python src/run_pipeline.py --data data/loan.csv --iv-threshold 0.02 --out outputs/validation_report.html
python src/run_pipeline.py --data data/loan.csv --include-underwriter-features   # see design decision below
```

> The loss-forecasting step simulates a monthly performance panel for the
> entire portfolio (1.27M loans) and can take several minutes — this is
> expected, not a hang.

## Methodology

1. **Data preparation** — the raw CSV is filtered to loans with a resolved
   outcome (Fully Paid / Charged Off; "Current" loans are excluded since
   they haven't matured), FICO range is averaged into a single score, and
   the portfolio is split **by origination date**, not randomly, into
   train (60%) / test (20%) / out-of-time or "OOT" (20%) — the same way a
   real credit risk team validates that a model still works on business
   it hasn't seen yet.
2. **Feature engineering (WOE/IV)** — every candidate feature is binned and
   converted to Weight-of-Evidence, with Information Value used to rank and
   filter down to the features carrying real univariate signal
   (IV ≥ 0.02).
3. **Scorecard fitting** — a `statsmodels` logistic regression is fit on the
   WOE-transformed features, regressing on P(good) so that positive
   coefficients and higher WOE both mean "safer," per the standard
   convention in Naeem Siddiqi's *Credit Risk Scorecards*. The fitted model
   is converted into a traditional points-based scorecard using the
   industry-standard PDO (points-to-double-odds) transform (pdo=20,
   base score=600 at 50:1 odds).
4. **Model design checks** — coefficient sign expectations and Variance
   Inflation Factors are computed automatically and flagged if they look
   wrong, mirroring a first-line model design review.
5. **Validation** — the fitted scorecard is scored on the test and OOT sets
   and run through five checks: discrimination (KS, Gini, AUC), calibration
   (predicted vs. observed default rate by decile), population stability
   (PSI), fairness (four-fifths / disparate impact rule), and reproducibility
   across the time-based holdouts.
6. **Loss forecasting** — a monthly performance panel is used to build
   vintage cumulative-default curves, a roll-rate transition matrix, and an
   Expected Loss (PD × LGD × EAD) projection by grade.
7. **Reporting** — every table and chart above is assembled into a single
   static HTML report (`outputs/validation_report.html`).

## Results

*(Full pipeline run on the complete 1,268,686-loan portfolio; see
`outputs/validation_report.html` for the underlying charts.)*

### Feature selection — Information Value

| Feature | IV | Strength |
|---|---|---|
| grade | 0.4802 | strong *(excluded — see below)* |
| int_rate | 0.4495 | strong *(excluded — see below)* |
| term | 0.2494 | medium |
| fico_score | 0.1188 | medium |
| dti | 0.0714 | weak |
| verification_status | 0.0483 | weak |
| loan_amnt | 0.0401 | weak |
| annual_inc | 0.0287 | weak |
| revol_util | 0.0222 | weak |
| home_ownership | 0.0199 | not predictive |
| purpose | 0.0158 | not predictive |
| credit_history_years | 0.0112 | not predictive |
| open_acc | 0.0091 | not predictive |
| delinq_2yrs | 0.0013 | not predictive |
| pub_rec | 0.0010 | not predictive |
| emp_length_years | 0.0005 | not predictive |
| total_acc | 0.0003 | not predictive |

**7 of 17** features clear the IV ≥ 0.02 threshold and are used in the final
scorecard: `term`, `fico_score`, `dti`, `verification_status`, `loan_amnt`,
`annual_inc`, `revol_util`.

### Scorecard — coefficients & multicollinearity

| Feature | Coefficient | p-value | Expected sign? | VIF |
|---|---|---|---|---|
| const | 1.5106 | 0.000 | ✅ | — |
| term | 0.9402 | 0.000 | ✅ | 1.24 |
| fico_score | 1.0440 | 0.000 | ✅ | 1.31 |
| dti | 0.6569 | 0.000 | ✅ | 1.12 |
| verification_status | 0.3875 | 0.000 | ✅ | 1.12 |
| loan_amnt | 0.3804 | 0.000 | ✅ | 1.52 |
| annual_inc | 1.1429 | 0.000 | ✅ | 1.30 |
| revol_util | -0.3796 | 0.000 | ⚠️ **flagged** | 1.33 |

VIF is clean across the board (max 1.52), so multicollinearity isn't
distorting these coefficients — `revol_util`'s flipped sign is a genuine
finding, discussed in [Known Limitations](#known-limitations--honest-caveats)
rather than hidden.

**Sample of the resulting points table** (higher points = lower risk):

| Feature | Bin | Points |
|---|---|---|
| term | 60 months | 56.0 |
| term | 36 months | 85.3 |
| fico_score | ≤ 667 | 65.3 |
| fico_score | 667–732 | 70.0 – 87.8 |
| fico_score | > 732 | 101.2 |

Score distribution on the test set (n=253,737): mean 535.9, std 19.3, range
478–593.

### Model validation summary

| Check | Test | Out-of-Time | Threshold |
|---|---|---|---|
| KS statistic | 24.5 | 21.9 | ≥ 20 acceptable |
| Gini coefficient | 0.348 | 0.309 | — |
| AUC | 0.674 | 0.654 | > 0.5 |
| PSI (vs. train) | 0.0100 | 0.0172 | < 0.10 stable |

**Fairness (four-fifths rule):** disparate impact ratio = **1.003** (passes
the 0.80 threshold) — see the caveat below on what this test can and can't
show with this dataset.

**Overall automated outcome: 5 of 6 checks PASS.** The one exception —
unexpected coefficient sign on `revol_util` — is flagged for review rather
than silently passed, which is the point of running this kind of gate in the
first place.

### Loss forecasting — expected loss by grade

Using the scorecard's own predicted PD (not Lending Club's realized
outcome) combined with a 65% loss-given-default assumption:

| Grade | Loans | Exposure | Avg. PD | Expected Loss | EL Rate |
|---|---|---|---|---|---|
| A | 222,420 | $3.12B | 9.2% | $185.8M | 6.0% |
| B | 370,187 | $4.95B | 14.5% | $474.2M | 9.6% |
| C | 359,332 | $5.16B | 19.4% | $683.4M | 13.2% |
| D | 188,548 | $2.92B | 23.0% | $461.9M | 15.8% |
| E | 88,731 | $1.58B | 28.5% | $304.6M | 19.2% |
| F | 30,606 | $591.3M | 32.3% | $127.3M | 21.5% |
| G | 8,862 | $182.8M | 33.7% | $40.1M | 21.9% |
| **Total** | **1,268,686** | **$18.51B** | **17.8%** | **$2.28B** | **12.3%** |

Notably, the model's predicted PD increases **monotonically** from grade A
through G — even though `grade` was never given to the model as an input.
That's meaningful external validation: an independently-built scorecard's
risk ordering agrees with Lending Club's own proprietary grading system
without having seen it.

## A Deliberate Design Decision: Excluding `grade` and `int_rate`

`grade` and `int_rate` show by far the highest Information Value of any
feature in this dataset (0.48 and 0.45, both "strong," vs. the next-best
feature at 0.25). It would be easy to build a scorecard around them and
report an artificially strong KS/AUC.

The problem: **`grade` and `int_rate` aren't applicant-provided data — they're
Lending Club's own underwriting model's output**, assigned to the loan at
origination. Including them doesn't produce an independent risk assessment;
it mostly re-derives Lending Club's own proprietary risk grade (they're also
collinear with each other, VIF ≈ 7–8, since `int_rate` is nearly
deterministic given `grade`).

This project excludes both by default (`UNDERWRITER_ASSIGNED_FEATURES` in
`scorecard.py`) so the resulting scorecard reflects genuine signal learned
from raw applicant/bureau attributes — while still reporting their IV for
context, and exposing `--include-underwriter-features` for anyone who wants
to benchmark this scorecard's discrimination power against Lending Club's
own grade on the same holdout.

## Known Limitations & Honest Caveats

A model validation framework is only credible if it's honest about what it
finds, including about itself. In that spirit:

- **`revol_util` has an unexpected coefficient sign.** It's monotonic and
  well-behaved on its own, and VIF (1.33) doesn't flag severe
  multicollinearity, but its partial effect flips once the other six
  features are controlled for — a genuine finding a real model design
  review would investigate further (e.g., its correlation with `loan_amnt`,
  `dti`, and `annual_inc` specifically), not a bug in the pipeline.
- **Discrimination is modest by design, not by mistake.** Lending Club's
  public dataset contains only *approved* loans — a "reject inference"
  problem. Any model trained on an already-approved population will show
  weaker discrimination than one trained on the full applicant pool, since
  the easiest-to-separate bad loans were filtered out before origination. A
  KS around 20–25 on an approved-only population is a reasonable, expected
  outcome.
- **The fairness test uses a synthetic protected-class proxy.** Lending
  Club's public file contains no race/ethnicity/sex field (real underwriting
  data never does, by law), so `data_prep.py` assigns a random placeholder
  group purely so the four-fifths rule test has something to run against.
  The passing 1.003 ratio demonstrates *how* to run and interpret a
  disparate-impact test — it is not evidence this model is fair in
  production, since there was no real signal for the test to detect bias in.
  A production deployment would use a proper BISG-derived proxy (or
  self-reported data where legally available) alongside legal/compliance
  review.
- **The roll-rate matrix is simulated, not observed.** Lending Club's public
  file records only each loan's final outcome, not a month-by-month payment
  history. `loss_forecast.py` reconstructs a plausible monthly delinquency
  path consistent with each loan's known outcome so the vintage/roll-rate
  machinery can be demonstrated — but this means the roll-rate matrix shows
  the *mechanics* of the calculation, not a real, empirically observed cure
  rate (e.g., real portfolios see 30-days-past-due loans "cure" back to
  current some of the time; this simulation does not model cures).
- **The earliest vintages (2007) are statistical noise.** Lending Club
  originated only a handful of loans per month when it launched — a single
  default in a 1–2 loan cohort shows up as a 100% cumulative default rate.
  These should be read as illustrative, not reliable, and would be trimmed
  or footnoted in a production report.

## Tech Stack

`pandas` · `numpy` · `statsmodels` (logistic regression with p-values/VIF) ·
`scikit-learn` (ROC/AUC) · `scipy` · `matplotlib` · `pytest`

## Skills Demonstrated

- Weight-of-Evidence / Information Value feature engineering
- Logistic regression scorecard design with industry-standard PDO scaling
- Multicollinearity diagnostics (VIF) and coefficient sign validation
- Out-of-time model validation methodology
- Discrimination metrics: KS statistic, Gini coefficient, AUC/ROC
- Population Stability Index (PSI) for model monitoring
- Fair-lending / disparate impact testing (four-fifths rule)
- Vintage analysis, roll-rate transition matrices, and Expected Loss
  (PD × LGD × EAD) projection
- Critical evaluation of feature legitimacy (recognizing and excluding a
  circular/leaky feature rather than reporting an inflated metric)
- Reproducible, tested, end-to-end pipeline design in Python

## License

MIT — see [LICENSE](LICENSE).

---

**Author:** Vishnu Muthyalu
