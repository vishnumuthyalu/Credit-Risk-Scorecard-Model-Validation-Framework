"""
woe_binning.py
==============
Weight-of-Evidence (WOE) binning and Information Value (IV) — the standard
feature-engineering technique used by bank credit-risk teams ahead of a
logistic-regression scorecard, instead of feeding raw/black-box features
into a model.

Why banks do it this way (and why it matters for validation):
  * WOE bins collapse noisy raw values into monotonic risk buckets, which
    makes the resulting logistic regression both more stable and directly
    interpretable ("each bin's WOE times its coefficient = its point
    contribution" — see scorecard.py).
  * IV gives a model-agnostic, pre-modeling measure of how predictive each
    feature is, which model-risk-management (MRM) teams use to justify
    which features were even considered (an early, well-documented feature
    selection step is itself part of "validating model design").

Conventions used here (standard in the credit industry):
    event  = "bad" = default = 1
    WOE_i  = ln( %good_i / %bad_i )   -> higher WOE = safer bin
    IV     = sum_i ( %good_i - %bad_i ) * WOE_i

IV interpretation (industry rule of thumb):
    < 0.02            not predictive
    0.02 - 0.10       weak
    0.10 - 0.30       medium
    0.30 - 0.50       strong
    > 0.50            suspiciously good -> check for leakage
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing count to avoid log(0) in sparse bins


@dataclass
class BinTable:
    feature: str
    kind: str  # "numeric" or "categorical"
    table: pd.DataFrame  # per-bin stats: label, count, bad, good, bad_rate, woe, iv
    iv: float
    edges: list = field(default_factory=list)          # numeric: bin edges
    category_map: dict = field(default_factory=dict)    # categorical: value -> bin label


def _woe_iv_from_counts(good: np.ndarray, bad: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Given per-bin good/bad counts, return (woe array, iv contribution array, total iv)."""
    total_good, total_bad = good.sum(), bad.sum()
    pct_good = (good + EPS) / (total_good + EPS * len(good))
    pct_bad = (bad + EPS) / (total_bad + EPS * len(good))
    woe = np.log(pct_good / pct_bad)
    iv_contrib = (pct_good - pct_bad) * woe
    return woe, iv_contrib, float(iv_contrib.sum())


class WOEBinner:
    """
    Fits monotonic WOE bins for numeric features and frequency-stabilized
    WOE bins for categorical features, then transforms raw features into
    their WOE values for use as logistic-regression inputs.
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        max_bins: int = 8,
        min_bin_pct: float = 0.05,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        self.min_bin_pct = min_bin_pct
        self.bins_: dict[str, BinTable] = {}

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEBinner":
        y = y.astype(int).values
        for col in self.numeric_features:
            self.bins_[col] = self._fit_numeric(X[col].values, y, col)
        for col in self.categorical_features:
            self.bins_[col] = self._fit_categorical(X[col].astype(str).values, y, col)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            if col not in self.bins_:
                continue
            bt = self.bins_[col]
            if bt.kind == "numeric":
                out[col] = self._apply_numeric(X[col].values, bt)
            else:
                out[col] = self._apply_categorical(X[col].astype(str).values, bt)
        return out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ #
    def iv_table(self) -> pd.DataFrame:
        """Feature-level IV summary, sorted descending — use this for feature selection."""
        rows = [{"feature": f, "iv": bt.iv, "n_bins": len(bt.table), "kind": bt.kind}
                for f, bt in self.bins_.items()]
        df = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

        def bucket(iv):
            if iv < 0.02: return "not predictive"
            if iv < 0.10: return "weak"
            if iv < 0.30: return "medium"
            if iv < 0.50: return "strong"
            return "suspicious (check leakage)"
        df["strength"] = df["iv"].apply(bucket)
        return df

    def bin_detail(self, feature: str) -> pd.DataFrame:
        """Full bin-level table for one feature — used in the validation report."""
        return self.bins_[feature].table

    # ------------------------------------------------------------------ #
    # Numeric: quantile init -> merge to enforce monotonic WOE + min size
    # ------------------------------------------------------------------ #
    def _fit_numeric(self, x: np.ndarray, y: np.ndarray, name: str) -> BinTable:
        mask = ~pd.isna(x)
        x, y = x[mask], y[mask]

        try:
            _, edges = pd.qcut(x, q=self.max_bins, retbins=True, duplicates="drop")
        except ValueError:
            edges = np.quantile(x, np.linspace(0, 1, self.max_bins + 1))
        edges = np.unique(edges)
        edges[0], edges[-1] = -np.inf, np.inf
        if len(edges) < 3:
            edges = np.array([-np.inf, np.median(x), np.inf])

        bin_idx = np.digitize(x, edges[1:-1], right=True)

        # Decide monotonic direction for WOE (not bad rate) as x increases.
        # corr(x, y) > 0 means higher x -> more defaults -> bad rate rises
        # -> WOE (ln(%good/%bad)) falls as x rises, so WOE direction is the
        # OPPOSITE sign of corr(x, y).
        direction = -1 if np.corrcoef(x, y)[0, 1] >= 0 else 1

        edges, bin_idx = self._merge_numeric(edges, bin_idx, y, direction)

        good = np.array([(y[bin_idx == i] == 0).sum() for i in range(len(edges) - 1)])
        bad = np.array([(y[bin_idx == i] == 1).sum() for i in range(len(edges) - 1)])
        woe, iv_contrib, iv = _woe_iv_from_counts(good, bad)

        labels = [self._interval_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        table = pd.DataFrame({
            "bin": labels, "count": good + bad, "good": good, "bad": bad,
            "bad_rate": bad / np.maximum(good + bad, 1), "woe": woe, "iv": iv_contrib,
        })
        return BinTable(feature=name, kind="numeric", table=table, iv=iv, edges=list(edges))

    def _merge_numeric(self, edges, bin_idx, y, direction):
        n = len(edges) - 1
        while n > 2:
            good = np.array([(y[bin_idx == i] == 0).sum() for i in range(n)])
            bad = np.array([(y[bin_idx == i] == 1).sum() for i in range(n)])
            total = good + bad
            woe, _, _ = _woe_iv_from_counts(good, bad)

            # Merge target 1: any bin under the minimum size threshold.
            frac = total / total.sum()
            small = np.where(frac < self.min_bin_pct)[0]

            # Merge target 2: first adjacent pair that breaks monotonicity.
            diffs = np.diff(woe)
            if direction == 1:
                violation = np.where(diffs < -1e-9)[0]
            else:
                violation = np.where(diffs > 1e-9)[0]

            if len(small) == 0 and len(violation) == 0:
                break

            merge_i = small[0] if len(small) else violation[0]
            merge_i = min(merge_i, n - 2)  # merge (merge_i, merge_i+1)

            edges = np.delete(edges, merge_i + 1)
            bin_idx = np.where(bin_idx > merge_i, bin_idx - 1, bin_idx)
            bin_idx = np.where(bin_idx == merge_i, merge_i, bin_idx)
            n -= 1
        return edges, bin_idx

    @staticmethod
    def _interval_label(lo, hi) -> str:
        lo_s = "-inf" if lo == -np.inf else f"{lo:.2f}"
        hi_s = "+inf" if hi == np.inf else f"{hi:.2f}"
        return f"({lo_s}, {hi_s}]"

    def _apply_numeric(self, x: np.ndarray, bt: BinTable) -> np.ndarray:
        edges = np.array(bt.edges)
        idx = np.digitize(x, edges[1:-1], right=True)
        idx = np.clip(idx, 0, len(bt.table) - 1)
        woe_lookup = bt.table["woe"].values
        result = np.where(pd.isna(x), np.nan, woe_lookup[idx])
        # Missing values get the WOE of the worst-risk (lowest-WOE) bin —
        # a conservative, documented convention, not silently dropped.
        if pd.isna(x).any():
            worst = woe_lookup.min()
            result = np.where(pd.isna(x), worst, result)
        return result.astype(float)

    # ------------------------------------------------------------------ #
    # Categorical: group rare categories, then WOE per remaining group
    # ------------------------------------------------------------------ #
    def _fit_categorical(self, x: np.ndarray, y: np.ndarray, name: str) -> BinTable:
        s = pd.Series(x)
        counts = s.value_counts(normalize=True)
        rare = set(counts[counts < self.min_bin_pct].index)
        mapped = s.where(~s.isin(rare), other="__OTHER__")

        cats = sorted(mapped.unique())
        good = np.array([((mapped == c) & (y == 0)).sum() for c in cats])
        bad = np.array([((mapped == c) & (y == 1)).sum() for c in cats])
        woe, iv_contrib, iv = _woe_iv_from_counts(good, bad)

        table = pd.DataFrame({
            "bin": cats, "count": good + bad, "good": good, "bad": bad,
            "bad_rate": bad / np.maximum(good + bad, 1), "woe": woe, "iv": iv_contrib,
        }).sort_values("woe").reset_index(drop=True)

        category_map = {orig: ("__OTHER__" if orig in rare else orig) for orig in s.unique()}
        return BinTable(feature=name, kind="categorical", table=table, iv=iv,
                         category_map=category_map)

    def _apply_categorical(self, x: np.ndarray, bt: BinTable) -> np.ndarray:
        mapped = pd.Series(x).map(lambda v: bt.category_map.get(v, "__OTHER__"))
        woe_by_bin = dict(zip(bt.table["bin"], bt.table["woe"]))
        default_woe = bt.table["woe"].min()
        return mapped.map(lambda b: woe_by_bin.get(b, default_woe)).astype(float).values


if __name__ == "__main__":
    from data_prep import load_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

    df = load_data()
    binner = WOEBinner(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    woe_df = binner.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET])
    print(binner.iv_table())
    print(binner.bin_detail("fico_score"))