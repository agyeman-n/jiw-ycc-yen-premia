#!/usr/bin/env python3
"""
YCC Impact on Yen Currency Premia — Econometric Analysis
=========================================================
Investigates the impact of the BOJ's September 2016 Yield Curve Control (YCC)
on JPY/USD currency premia using 3-month tenor data.

Sample: Dec 2008 – Feb 2026 (monthly, month-end)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import ruptures as rpt
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ============================================================
# Configuration
# ============================================================
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = f"{ROOT / 'data'}/"   # put the input files listed in README.md here
OUT_DIR = f"{ROOT / 'output'}/"
Path(OUT_DIR).mkdir(exist_ok=True)
START_DATE = pd.Timestamp("2008-10-01")
END_DATE = pd.Timestamp("2026-02-28")
YCC_DATE = pd.Timestamp("2016-09-01")
YCC_EXIT_DATE = pd.Timestamp("2024-03-01")  # BOJ formally ended YCC
POST_EXIT_START = pd.Timestamp("2024-04-01")  # start post-exit regime from next month
NW_LAGS = 4  # Newey-West truncation lag (Bartlett kernel)

# ============================================================
# STEP 1: Data Ingestion and Alignment
# ============================================================
print("=" * 70)
print("STEP 1: Data Ingestion and Alignment")
print("=" * 70)


def to_monthly(series, date_col, val_col, name):
    """Convert a daily/irregular series to month-end frequency."""
    df = series[[date_col, val_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna().set_index(date_col).sort_index()
    df = df.rename(columns={val_col: name})
    return df.resample("ME").last().dropna()


# --- 1a. Spot Rate (DEXJPUS: JPY per USD, daily) ---
print("  Loading spot rate...")
spot = pd.read_csv(f"{DATA_DIR}jpy_usd_spot.csv")
spot_m = to_monthly(spot, "observation_date", "DEXJPUS", "S")

# --- 1b. 3-month Forward Swap Points (xlsx, daily) ---
#     The xlsx contains daily forward *swap points* (in pips).
#     Outright forward: F = S + swap_points / 100
print("  Loading 3-month forward swap points...")
fwd_raw = pd.read_excel(f"{DATA_DIR}jpy_usd_3m.forward.xlsx", sheet_name="Sheet1")
fwd_m = to_monthly(fwd_raw, "Date", "Close", "fwd_points")

# Merge spot and forward at monthly level, then compute outright forward
sf = spot_m.join(fwd_m, how="inner")
sf["F"] = sf["S"] + sf["fwd_points"] / 100.0

# --- 1c. 3-month Exchange Rate Forecast ---
print("  Loading 3-month exchange rate forecast...")
fc_raw = pd.read_csv(
    f"{DATA_DIR}3m-ex-rate-forecast - Sheet1.csv",
    header=None,
    names=["date_str", "E_S"],
)
fc_raw = fc_raw.iloc[1:]  # skip text header row
fc_raw["date"] = pd.to_datetime(fc_raw["date_str"].str.strip(), format="%Y - %b")
fc_raw["E_S"] = pd.to_numeric(fc_raw["E_S"], errors="coerce")
forecast = fc_raw[["date", "E_S"]].dropna().set_index("date").sort_index()
forecast = forecast.resample("ME").last().dropna()

# --- 1d. Japan 3-month Interbank Rate (already monthly) ---
print("  Loading Japan 3-month interest rate...")
jpn_rate = pd.read_csv(f"{DATA_DIR}Japan_int.rate_3m.csv")
jpn_rate["observation_date"] = pd.to_datetime(jpn_rate["observation_date"])
jpn_rate["i_JPY_raw"] = pd.to_numeric(jpn_rate["IR3TIB01JPM156N"], errors="coerce")
jpn_rate = (
    jpn_rate[["observation_date", "i_JPY_raw"]]
    .dropna()
    .set_index("observation_date")
    .sort_index()
    .resample("ME")
    .last()
    .dropna()
)

# --- 1e. US 3-month T-Bill Rate (daily) ---
print("  Loading US 3-month interest rate...")
us_rate = pd.read_csv(f"{DATA_DIR}US_int.rate_3m.csv")
us_rate_m = to_monthly(us_rate, "observation_date", "DTB3", "i_USD_raw")

# --- 1f. Japan Corporate Bond Spreads (monthly) ---
print("  Loading Japan corporate bond spreads...")
jpn_spread = pd.read_csv(f"{DATA_DIR}Japan_corporate_bond_spreads.csv")
jpn_spread["date"] = pd.to_datetime(jpn_spread["date"], errors="coerce")
jpn_spread["JPN_spread"] = pd.to_numeric(
    jpn_spread["INJPNW-close"], errors="coerce"
)
jpn_spread = (
    jpn_spread[["date", "JPN_spread"]]
    .dropna()
    .set_index("date")
    .sort_index()
    .resample("ME")
    .last()
    .dropna()
)

# --- 1g. US Corporate Bond Spread — EBP proxy (daily) ---
print("  Loading US corporate bond spreads (EBP proxy)...")
us_spread = pd.read_csv(f"{DATA_DIR}US_corporate_bond_spreads.csv")
ebp_m = to_monthly(us_spread, "observation_date", "BAMLC0A0CM", "EBP")

# --- 1h. S&P 500 Equal-Weight Financials (monthly) ---
print("  Loading S&P 500 Financials index...")
spfin_raw = pd.read_csv(
    f'{DATA_DIR}S&P 500 EQUAL WEIGHTED Financials (Sector).csv'
)
spfin_raw["date"] = pd.to_datetime(
    spfin_raw["Date"].str.strip('"'), format="%m/%Y", errors="coerce"
)
spfin_raw["SPFin"] = pd.to_numeric(
    spfin_raw["Close"].astype(str).str.replace(",", "").str.strip('"'),
    errors="coerce",
)
spfin = (
    spfin_raw[["date", "SPFin"]]
    .dropna()
    .set_index("date")
    .sort_index()
    .resample("ME")
    .last()
    .dropna()
)

# --- 1i. VIX (daily) ---
print("  Loading VIX...")
vix_m = to_monthly(
    pd.read_csv(f"{DATA_DIR}Vix_data.csv"), "observation_date", "VIXCLS", "VIX"
)

# --- Merge into master DataFrame ---
print("\n  Merging all series on monthly dates...")
master = sf[["S", "F"]].join(
    [forecast, jpn_rate, us_rate_m, jpn_spread, ebp_m, spfin, vix_m],
    how="outer",
)
master = master.sort_index()
master = master.ffill()
master = master.loc[START_DATE:END_DATE]

pre_drop = len(master)
master = master.dropna()
print(f"  Observations before dropna: {pre_drop}, after: {len(master)}")
print(f"  Date range: {master.index[0]:%Y-%m} to {master.index[-1]:%Y-%m}")

# ============================================================
# STEP 2: Variable Construction
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Variable Construction")
print("=" * 70)

master["s"] = np.log(master["S"])
master["f"] = np.log(master["F"])
master["ln_E_S"] = np.log(master["E_S"])

master["i_JPY"] = master["i_JPY_raw"] / 100.0
master["i_USD"] = master["i_USD_raw"] / 100.0

# CIP Deviation (3-month): f_t − s_t − (i_JPY,t − i_USD,t)
master["CIP_Dev"] = master["f"] - master["s"] - (master["i_JPY"] - master["i_USD"])

# Expected UIP Deviation (3-month): ln(E[S_{t+3}]) − s_t − (i_JPY,t − i_USD,t)
master["EUIP_Dev"] = master["ln_E_S"] - master["s"] - (master["i_JPY"] - master["i_USD"])

# Lagged spot change: Δs_{t-1} = s_{t-1} − s_{t-2}
master["delta_s_lag1"] = master["s"].diff().shift(1)

# Drop rows with NaN from differencing
reg_vars = ["CIP_Dev", "EUIP_Dev", "delta_s_lag1", "EBP", "SPFin", "VIX"]
master = master.dropna(subset=reg_vars)

print(f"  Final sample: {len(master)} monthly observations")
print(f"  Date range:   {master.index[0]:%Y-%m} to {master.index[-1]:%Y-%m}\n")
print(master[["CIP_Dev", "EUIP_Dev", "EBP", "VIX", "SPFin"]].describe().round(6))

# ============================================================
# STEP 3: Structural Break Testing (Bai-Perron via ruptures)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Endogenous Structural Break Testing")
print("=" * 70)

signal = master["EUIP_Dev"].values
n_obs = len(signal)

# --- Single breakpoint (Bai-Perron via dynamic programming, L2 cost) ---
algo = rpt.Dynp(model="l2", min_size=12, jump=1).fit(signal)
bkpt_1 = algo.predict(n_bkps=1)
break_idx_1 = bkpt_1[0]
T_break_1 = master.index[break_idx_1]

print(f"\n  --- Single-break analysis ---")
print(f"  Estimated break index : {break_idx_1} / {n_obs}")
print(f"  Estimated break date  : {T_break_1:%Y-%m}")
print(f"  Expected (YCC)        : 2016-09")
months_diff_1 = (T_break_1.year - 2016) * 12 + (T_break_1.month - 9)
print(f"  Distance from YCC     : {months_diff_1:+d} months")

# --- Two breakpoints ---
bkpt_2 = algo.predict(n_bkps=2)
print(f"\n  --- Two-break analysis ---")
for i, bi in enumerate(bkpt_2[:-1]):
    dt = master.index[bi]
    md = (dt.year - 2016) * 12 + (dt.month - 9)
    print(f"  Break {i+1}: index {bi}, date {dt:%Y-%m}  (distance from YCC: {md:+d} months)")

# Use the break closest to YCC for regime analysis
all_breaks = [bkpt_1[0]] + list(bkpt_2[:-1])
all_dates = [master.index[b] for b in all_breaks]
closest_idx = np.argmin([abs((d.year - 2016) * 12 + (d.month - 9)) for d in all_dates])
T_break = all_dates[closest_idx]
break_idx = all_breaks[closest_idx]

months_diff = (T_break.year - 2016) * 12 + (T_break.month - 9)
print(f"\n  Break nearest to YCC date: {T_break:%Y-%m} ({months_diff:+d} months)")

# Chow test at the fixed YCC date (Sept 2016) as a robustness check
ycc_idx = master.index.get_indexer([YCC_DATE], method="nearest")[0]
pre_vals = signal[:ycc_idx]
post_vals = signal[ycc_idx:]
n1, n2 = len(pre_vals), len(post_vals)
k = 2  # parameters per regime (mean + variance)
rss_full = np.sum((signal - signal.mean()) ** 2)
rss_1 = np.sum((pre_vals - pre_vals.mean()) ** 2)
rss_2 = np.sum((post_vals - post_vals.mean()) ** 2)
chow_f = ((rss_full - rss_1 - rss_2) / k) / ((rss_1 + rss_2) / (n1 + n2 - 2 * k))
chow_p = 1 - stats.f.cdf(chow_f, k, n1 + n2 - 2 * k)
print(f"\n  Chow test at fixed Sept 2016:")
print(f"    F-statistic = {chow_f:.4f},  p-value = {chow_p:.4f}")
if chow_p < 0.05:
    print("    → Reject H₀ at 5%: structural break confirmed at Sept 2016.")
else:
    print("    → Cannot reject H₀ at 5% for a break at Sept 2016.")

# Regime statistics (split at YCC date for policy relevance)
pre_break = master.loc[master.index < YCC_DATE, "EUIP_Dev"]
post_break = master.loc[master.index >= YCC_DATE, "EUIP_Dev"]

print("\n  Regime descriptive statistics for EUIP_Dev (split at YCC date):")
print(f"  {'':30s} {'Pre-YCC':>12s} {'Post-YCC':>12s}")
print(f"  {'Mean (μ)':30s} {pre_break.mean():12.6f} {post_break.mean():12.6f}")
print(f"  {'Std. Dev. (σ)':30s} {pre_break.std():12.6f} {post_break.std():12.6f}")
print(f"  {'Observations':30s} {len(pre_break):12d} {len(post_break):12d}")

# Levene's test for equal variances
lev_stat, lev_p = stats.levene(pre_break, post_break)
print(f"\n  Levene's test for equal variances:")
print(f"    W-statistic = {lev_stat:.4f},  p-value = {lev_p:.4f}")
if lev_p < 0.05:
    print("    → Reject H₀ at 5%: volatility significantly DIFFERS across regimes.")
else:
    print("    → Cannot reject H₀: no significant difference in volatility.")

# Welch's t-test for difference in means
t_stat, t_p = stats.ttest_ind(pre_break, post_break, equal_var=False)
print(f"\n  Welch's t-test for equal means:")
print(f"    t-statistic = {t_stat:.4f},  p-value = {t_p:.4f}")
if t_p < 0.05:
    print("    → Reject H₀ at 5%: regime means are significantly different.")
else:
    print("    → Cannot reject H₀: no significant difference in regime means.")

# Also run F-test for variance ratio
var_ratio = pre_break.var() / post_break.var()
df1, df2 = len(pre_break) - 1, len(post_break) - 1
f_p = 2 * min(stats.f.cdf(var_ratio, df1, df2), 1 - stats.f.cdf(var_ratio, df1, df2))
print(f"\n  F-test for variance ratio:")
print(f"    F-statistic = {var_ratio:.4f} (df1={df1}, df2={df2}),  p-value = {f_p:.4f}")

# ============================================================
# STEP 4: Regime-Dependent Regression Analysis
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Regime-Dependent OLS Regressions (Newey-West SEs)")
print("=" * 70)

# Split using YCC_DATE (Sept 2016) and YCC exit (Mar 2024) for the regressions
pre_ycc = master.loc[master.index < YCC_DATE].copy()
post_ycc = master.loc[(master.index >= YCC_DATE) & (master.index < POST_EXIT_START)].copy()
post_exit_df = master.loc[(master.index >= POST_EXIT_START) & (master.index <= END_DATE)].copy()
print(f"\n  Pre-YCC sample  : {pre_ycc.index[0]:%Y-%m} to {pre_ycc.index[-1]:%Y-%m}  (n={len(pre_ycc)})")
print(f"  Post-YCC sample : {post_ycc.index[0]:%Y-%m} to {post_ycc.index[-1]:%Y-%m}  (n={len(post_ycc)})")
print(f"  Post-exit sample: {post_exit_df.index[0]:%Y-%m} to {post_exit_df.index[-1]:%Y-%m}  (n={len(post_exit_df)})")

rhs_vars = ["EBP", "SPFin", "delta_s_lag1", "VIX"]


def run_ols_nw(y, X_df, lags=NW_LAGS):
    """OLS with Newey-West HAC standard errors."""
    X = add_constant(X_df)
    model = OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return model


# Model 1: Pre-YCC
model1 = run_ols_nw(pre_ycc["CIP_Dev"], pre_ycc[rhs_vars])
# Model 2: Post-YCC
model2 = run_ols_nw(post_ycc["CIP_Dev"], post_ycc[rhs_vars])
# Model 3: Post-exit (Apr 2024 – Feb 2026)
model3 = run_ols_nw(post_exit_df["CIP_Dev"], post_exit_df[rhs_vars])

print("\n  Model 1 (Pre-YCC) — Dep. var: CIP_Dev")
print(model1.summary2().tables[1].to_string())
print(f"  R² = {model1.rsquared:.4f},  Adj. R² = {model1.rsquared_adj:.4f}, "
      f" DW = {durbin_watson(model1.resid):.3f}")

print("\n  Model 2 (Post-YCC) — Dep. var: CIP_Dev")
print(model2.summary2().tables[1].to_string())
print(f"  R² = {model2.rsquared:.4f},  Adj. R² = {model2.rsquared_adj:.4f}, "
      f" DW = {durbin_watson(model2.resid):.3f}")

print("\n  Model 3 (Post-exit) — Dep. var: CIP_Dev")
print(model3.summary2().tables[1].to_string())
print(f"  R² = {model3.rsquared:.4f},  Adj. R² = {model3.rsquared_adj:.4f}, "
      f" DW = {durbin_watson(model3.resid):.3f}")

# Post-exit correlation: CIP vs EBP (bivariate)
post_exit_corr = float(post_exit_df[["CIP_Dev", "EBP"]].corr().iloc[0, 1])
print(f"\n  Correlation (Post-exit Apr 2024–Feb 2026): corr(CIP_Dev, EBP) = {post_exit_corr:.3f}")

# --- Wald test via pooled interacted regression ---
print("\n  Wald test: H₀ — EBP and VIX coefficients equal across regimes")
pooled = master.copy()
pooled["D"] = (pooled.index >= YCC_DATE).astype(int)
pooled["D_EBP"] = pooled["D"] * pooled["EBP"]
pooled["D_SPFin"] = pooled["D"] * pooled["SPFin"]
pooled["D_delta_s"] = pooled["D"] * pooled["delta_s_lag1"]
pooled["D_VIX"] = pooled["D"] * pooled["VIX"]

pooled_rhs = rhs_vars + ["D", "D_EBP", "D_SPFin", "D_delta_s", "D_VIX"]
model_pooled = run_ols_nw(pooled["CIP_Dev"], pooled[pooled_rhs])

# Joint test: δ_EBP = 0  AND  δ_VIX = 0
r_matrix = np.zeros((2, len(model_pooled.params)))
param_names = list(model_pooled.params.index)
r_matrix[0, param_names.index("D_EBP")] = 1
r_matrix[1, param_names.index("D_VIX")] = 1

wald_result = model_pooled.wald_test(r_matrix, use_f=True)
wald_f = float(wald_result.statistic)
wald_p = float(wald_result.pvalue)
print(f"    F-statistic = {wald_f:.4f},  p-value = {wald_p:.4f}")
if wald_p < 0.05:
    print("    → Reject H₀ at 5%: coefficients on EBP and VIX differ significantly across regimes.")
else:
    print("    → Cannot reject H₀: no significant cross-regime difference in EBP/VIX loadings.")

# --- Formal 3-regime coefficient stability test (pre / post-YCC / post-exit) ---
print("\n  3-regime coefficient stability (pooled interactions, HAC)")
pooled3 = master.copy()
pooled3["D_post"] = ((pooled3.index >= YCC_DATE) & (pooled3.index < POST_EXIT_START)).astype(int)
pooled3["D_exit"] = (pooled3.index >= POST_EXIT_START).astype(int)

for v in rhs_vars:
    pooled3[f"D_post_{v}"] = pooled3["D_post"] * pooled3[v]
    pooled3[f"D_exit_{v}"] = pooled3["D_exit"] * pooled3[v]

pooled3_rhs = (
    rhs_vars
    + ["D_post", "D_exit"]
    + [f"D_post_{v}" for v in rhs_vars]
    + [f"D_exit_{v}" for v in rhs_vars]
)
model_pooled3 = run_ols_nw(pooled3["CIP_Dev"], pooled3[pooled3_rhs])

# Joint Wald: Post-exit vs Post-YCC slope differences on EBP and VIX
# H0: (D_exit_EBP - D_post_EBP) = 0 AND (D_exit_VIX - D_post_VIX) = 0
param3 = list(model_pooled3.params.index)
r3 = np.zeros((2, len(param3)))
r3[0, param3.index("D_exit_EBP")] = 1
r3[0, param3.index("D_post_EBP")] = -1
r3[1, param3.index("D_exit_VIX")] = 1
r3[1, param3.index("D_post_VIX")] = -1

wald_3 = model_pooled3.wald_test(r3, use_f=True)
wald3_f = float(wald_3.statistic)
wald3_p = float(wald_3.pvalue)
print("    H₀: Post-exit vs Post-YCC (EBP, VIX slopes) are equal")
print(f"    F-statistic = {wald3_f:.4f},  p-value = {wald3_p:.4f}")
if wald3_p < 0.05:
    print("    → Reject H₀ at 5%: EBP/VIX loadings changed from Post-YCC to Post-exit.")
else:
    print("    → Cannot reject H₀: no significant change in EBP/VIX loadings from Post-YCC to Post-exit.")

# ============================================================
# STEP 5: Output and Visualization
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Output Tables and Figures")
print("=" * 70)

# --- 5a. Academic regression table ---
table = summary_col(
    [model1, model2, model3],
    model_names=["Pre-YCC", "Post-YCC", "Post-exit"],
    stars=True,
    float_format="%.6f",
    regressor_order=["const", "EBP", "SPFin", "delta_s_lag1", "VIX"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "DW": lambda x: f"{durbin_watson(x.resid):.3f}",
    },
)
print("\n" + "=" * 70)
print("TABLE: Regime-Dependent OLS — CIP Deviation (3-month)")
print("       Newey-West HAC standard errors in parentheses")
print("=" * 70)
print(table)

# --- 5a.1 Write Stata-style regression output + narrative report ---
def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def stata_like_table(model, title: str, depvar: str, hac_lags: int) -> str:
    # statsmodels with HAC uses z-stats in summary2; we print t/z generically as "t"
    rows = []
    for name in model.params.index:
        b = float(model.params.loc[name])
        se = float(model.bse.loc[name])
        t = float(model.tvalues.loc[name])
        p = float(model.pvalues.loc[name])
        rows.append((name, b, se, t, p, _stars(p)))

    name_width = max(10, max(len(r[0]) for r in rows))
    header = (
        f"{title}\n"
        f"Dependent variable: {depvar}\n"
        f"Estimator: OLS with Newey-West HAC SEs (lags={hac_lags})\n"
        f"Number of obs = {int(model.nobs)}\n"
        f"R-squared     = {model.rsquared:.4f}\n"
        f"Adj R-squared = {model.rsquared_adj:.4f}\n"
        f"Durbin-Watson = {durbin_watson(model.resid):.3f}\n"
        "\n"
        "------------------------------------------------------------------------------\n"
        f"{'Variable':<{name_width}}  {'Coef.':>12}  {'Std. Err.':>12}  {'t':>9}  {'P>|t|':>9}\n"
        "------------------------------------------------------------------------------\n"
    )
    body = "\n".join(
        f"{nm:<{name_width}}  {b:>12.6f}{st:>3}  {se:>12.6f}  {t:>9.3f}  {p:>9.4f}"
        for (nm, b, se, t, p, st) in rows
    )
    footer = (
        "\n------------------------------------------------------------------------------\n"
        "Significance: * p<0.10, ** p<0.05, *** p<0.01\n"
    )
    return header + body + footer


stata_out = []
stata_out.append("STATA-STYLE OUTPUT (replicated in Python/statsmodels)")
stata_out.append("=" * 78)
stata_out.append("")
stata_out.append(". * Model 1: Pre-YCC")
stata_out.append(
    ". newey CIP_Dev EBP SPFin delta_s_lag1 VIX, lag(%d)" % NW_LAGS
)
stata_out.append("")
stata_out.append(stata_like_table(model1, "Model 1 (Pre-YCC)", "CIP_Dev", NW_LAGS))
stata_out.append("")
stata_out.append(". * Model 2: Post-YCC")
stata_out.append(
    ". newey CIP_Dev EBP SPFin delta_s_lag1 VIX, lag(%d)" % NW_LAGS
)
stata_out.append("")
stata_out.append(stata_like_table(model2, "Model 2 (Post-YCC)", "CIP_Dev", NW_LAGS))
stata_out.append("")
stata_out.append(". * Model 3: Post-exit (Apr 2024 – Feb 2026)")
stata_out.append(
    ". newey CIP_Dev EBP SPFin delta_s_lag1 VIX, lag(%d)" % NW_LAGS
)
stata_out.append("")
stata_out.append(stata_like_table(model3, "Model 3 (Post-exit)", "CIP_Dev", NW_LAGS))
stata_out.append("")
stata_out.append(". * Cross-regime Wald test (joint): H0: D_EBP = 0 and D_VIX = 0")
stata_out.append(". test (D_EBP = 0) (D_VIX = 0)")
stata_out.append(f"  F({int(wald_result.df_num)}, {int(wald_result.df_denom)}) = {wald_f:.4f}   Prob > F = {wald_p:.4f}")
stata_out.append("")
stata_out.append(". * 3-regime pooled-interaction stability test")
stata_out.append(". * H0: (D_exit_EBP - D_post_EBP) = 0 and (D_exit_VIX - D_post_VIX) = 0")
stata_out.append(". test (D_exit_EBP = D_post_EBP) (D_exit_VIX = D_post_VIX)")
stata_out.append(f"  F({int(wald_3.df_num)}, {int(wald_3.df_denom)}) = {wald3_f:.4f}   Prob > F = {wald3_p:.4f}")
stata_out.append("")

stata_path = f"{OUT_DIR}stata_style_regressions.txt"
with open(stata_path, "w", encoding="utf-8") as f:
    f.write("\n".join(stata_out))
print(f"\n  Saved: stata_style_regressions.txt")

# --- 5b. Figure 1: EUIP_Dev time series with structural break ---
fig1, ax1 = plt.subplots(figsize=(13, 6))

ax1.plot(
    master.index,
    master["EUIP_Dev"],
    color="#2c3e50",
    linewidth=1.0,
    label="$EUIP\\_Dev_t$ (3-month)",
)
ax1.axvline(
    YCC_DATE,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"YCC Implementation ({YCC_DATE:%b %Y})",
)

pre_mean = master.loc[master.index < YCC_DATE, "EUIP_Dev"].mean()
post_mean = master.loc[master.index >= YCC_DATE, "EUIP_Dev"].mean()

ax1.axhline(
    pre_mean,
    xmin=0,
    xmax=0.5,
    color="#e74c3c",
    linestyle=":",
    linewidth=1.2,
    alpha=0.8,
)
ax1.axhline(
    post_mean,
    xmin=0.5,
    xmax=1.0,
    color="#2980b9",
    linestyle=":",
    linewidth=1.2,
    alpha=0.8,
)

# Shade regimes
ax1.axvspan(master.index[0], YCC_DATE, alpha=0.04, color="red", label=None)
ax1.axvspan(YCC_DATE, master.index[-1], alpha=0.04, color="blue", label=None)

ax1.annotate(
    f"Pre-YCC mean = {pre_mean:.4f}",
    xy=(master.index[5], pre_mean),
    xytext=(15, 15),
    textcoords="offset points",
    fontsize=9,
    color="#e74c3c",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c", alpha=0.8),
)
ax1.annotate(
    f"Post-YCC mean = {post_mean:.4f}",
    xy=(master.index[-10], post_mean),
    xytext=(15, -25),
    textcoords="offset points",
    fontsize=9,
    color="#2980b9",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2980b9", alpha=0.8),
)

ax1.set_title(
    "Expected UIP Deviation (3-month): Pre- vs. Post-YCC Regimes",
    fontweight="bold",
)
ax1.set_xlabel("Date")
ax1.set_ylabel("$EUIP\\_Dev_t$")
ax1.legend(loc="upper right", framealpha=0.9)
ax1.xaxis.set_major_locator(mdates.YearLocator(2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}fig1_EUIP_Dev_timeseries.png", bbox_inches="tight")
print("\n  Saved: fig1_EUIP_Dev_timeseries.png")

# --- 5c. Figure 2: CIP_Dev and EBP dual-axis plot ---
fig2, ax2a = plt.subplots(figsize=(13, 6))
ax2b = ax2a.twinx()

ln1 = ax2a.plot(
    master.index,
    master["CIP_Dev"],
    color="#2c3e50",
    linewidth=1.0,
    label="$CIP\\_Dev_t$ (LHS)",
)
ln2 = ax2b.plot(
    master.index,
    master["EBP"],
    color="#e67e22",
    linewidth=1.0,
    alpha=0.85,
    label="$EBP_t$ — US Corp. Spread (RHS)",
)
ax2a.axvline(
    YCC_DATE,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label=f"YCC ({YCC_DATE:%b %Y})",
)
ax2a.axvline(
    YCC_EXIT_DATE,
    color="#8e44ad",
    linestyle="--",
    linewidth=1.5,
    label=f"YCC exit ({YCC_EXIT_DATE:%b %Y})",
)

# Correlation annotations
pre_corr = master.loc[master.index < YCC_DATE, ["CIP_Dev", "EBP"]].corr().iloc[0, 1]
post_corr = master.loc[(master.index >= YCC_DATE) & (master.index < POST_EXIT_START), ["CIP_Dev", "EBP"]].corr().iloc[0, 1]
post_exit_corr_plot = post_exit_df[["CIP_Dev", "EBP"]].corr().iloc[0, 1]

mid_pre = master.loc[master.index < YCC_DATE].index[len(pre_ycc) // 2]
mid_post = master.loc[(master.index >= YCC_DATE) & (master.index < POST_EXIT_START)].index[len(post_ycc) // 2]
mid_exit = post_exit_df.index[len(post_exit_df) // 2]

ax2a.annotate(
    f"Pre-YCC ρ = {pre_corr:.3f}",
    xy=(mid_pre, master.loc[mid_pre, "CIP_Dev"]),
    xytext=(0, 40),
    textcoords="offset points",
    fontsize=9,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#2c3e50", alpha=0.9),
)
ax2a.annotate(
    f"Post-YCC ρ = {post_corr:.3f}",
    xy=(mid_post, master.loc[mid_post, "CIP_Dev"]),
    xytext=(0, 40),
    textcoords="offset points",
    fontsize=9,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", edgecolor="#2c3e50", alpha=0.9),
)
ax2a.annotate(
    f"Post-exit ρ = {post_exit_corr_plot:.3f}",
    xy=(mid_exit, master.loc[mid_exit, "CIP_Dev"]),
    xytext=(0, -55),
    textcoords="offset points",
    fontsize=9,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5eef8", edgecolor="#8e44ad", alpha=0.95),
)

ax2a.set_xlabel("Date")
ax2a.set_ylabel("$CIP\\_Dev_t$", color="#2c3e50")
ax2b.set_ylabel("$EBP_t$ (US Corp. OAS, %)", color="#e67e22")
ax2a.tick_params(axis="y", labelcolor="#2c3e50")
ax2b.tick_params(axis="y", labelcolor="#e67e22")

lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax2a.legend(lns, labs, loc="upper left", framealpha=0.9)

ax2a.set_title(
    "CIP Deviation vs. Excess Bond Premium Proxy: Regime Comparison",
    fontweight="bold",
)
ax2a.xaxis.set_major_locator(mdates.YearLocator(2))
ax2a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}fig2_CIP_EBP_dual_axis.png", bbox_inches="tight")
print("  Saved: fig2_CIP_EBP_dual_axis.png")

# --- 5d. Figure 3 (bonus): CIP deviation with regime shading ---
fig3, ax3 = plt.subplots(figsize=(13, 5))
ax3.fill_between(
    master.index,
    master["CIP_Dev"],
    alpha=0.3,
    color="#3498db",
    label="$CIP\\_Dev_t$",
)
ax3.plot(master.index, master["CIP_Dev"], color="#2c3e50", linewidth=0.8)
ax3.axvline(YCC_DATE, color="red", linestyle="--", linewidth=1.5)
ax3.axhline(0, color="black", linewidth=0.5)
ax3.set_title("CIP Deviation (3-month): Full Sample", fontweight="bold")
ax3.set_ylabel("$CIP\\_Dev_t$")
ax3.set_xlabel("Date")
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax3.legend(loc="best", framealpha=0.9)
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}fig3_CIP_Dev_shaded.png", bbox_inches="tight")
print("  Saved: fig3_CIP_Dev_shaded.png")

plt.close("all")

print("\n" + "=" * 70)
print("Analysis complete. All figures saved to disk.")
print("=" * 70)
